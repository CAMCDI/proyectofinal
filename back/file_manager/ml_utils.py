import os
import re
import arff
import string
import email
import html.parser
import zipfile
import shutil
import nltk
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Cargar recursos de NLTK
try:
    from .visualizer import MLVisualizer
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from visualizer import MLVisualizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Utilidades Compartidas ---

class MLStripper(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html_content):
    s = MLStripper()
    s.feed(html_content)
    return s.get_data()

# --- Lógica de Detección de Spam ---

class EmailParser:
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    def parse(self, email_path):
        """Parsea un archivo de correo electrónico."""
        try:
            with open(email_path, 'r', errors='ignore') as e:
                msg = email.message_from_file(e)
            return None if not msg else self.get_email_content(msg)
        except Exception as e:
            return {"error": str(e)}

    def get_email_content(self, msg):
        subject = self.tokenize(msg['Subject']) if msg['Subject'] else []
        body = self.get_email_body(msg.get_payload(), msg.get_content_type())
        return {"subject": subject, "body": body, "content_type": msg.get_content_type()}

    def get_email_body(self, payload, content_type):
        body = []
        if isinstance(payload, str):
            if content_type == 'text/plain':
                return self.tokenize(payload)
            elif content_type == 'text/html':
                return self.tokenize(strip_tags(payload))
        elif isinstance(payload, list):
            for p in payload:
                body += self.get_email_body(p.get_payload(), p.get_content_type())
        return body

    def tokenize(self, text):
        """Limpia puntuación, tabulaciones y realiza stemming."""
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ").replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]

# --- Transformadores Personalizados ---

class DeleteNanRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.dropna()

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled_df[attr]
        return X_copy

class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._columns = None
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        if not X_cat.empty:
            self._oh.fit(X_cat)
            self._columns = list(self._oh.get_feature_names_out(X_cat.columns))
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        if X_cat.empty:
            return X_copy
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh_df = pd.DataFrame(X_cat_oh, columns=self._columns, index=X_copy.index)
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh_df)

class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._full_pipeline = None
        self._columns = None

    def fit(self, X, y=None):
        num_attribs = list(X.select_dtypes(exclude=['object']))
        cat_attribs = list(X.select_dtypes(include=['object']))
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('rbst_scaler', RobustScaler()),
        ])
        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoding(), cat_attribs),
        ])
        self._full_pipeline.fit(X)
        cat_transformer = self._full_pipeline.named_transformers_['cat']
        cat_cols = cat_transformer._columns if hasattr(cat_transformer, '_columns') and cat_transformer._columns else []
        self._columns = num_attribs + list(cat_cols)
        return self

    def transform(self, X, y=None):
        X_prep = self._full_pipeline.transform(X.copy())
        return pd.DataFrame(X_prep, columns=self._columns, index=X.index)

# --- Gestor de Procesamiento ML ---

class MLManager:
    @staticmethod
    def analyze_05_spam(file_path):
        """Procesa archivos ZIP con correos para detección de Spam."""
        try:
            if not zipfile.is_zipfile(file_path):
                return {"success": False, "error": "El archivo debe ser un ZIP"}
            
            extract_dir = os.path.join(os.path.dirname(file_path), "extracted_" + os.path.basename(file_path))
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            file_map = {}
            index_path = None
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    file_map[f] = os.path.join(root, f)
                    if f.lower() in ['index', 'index.txt']:
                        index_path = os.path.join(root, f)
            
            if not index_path:
                possible_paths = [
                    os.path.join(settings.BASE_DIR.parent.parent, "Datasets/datasets/trec07p/full/index.txt"),
                    "/home/aza/Documentos/simulacion/Datasets/datasets/trec07p/full/index.txt"
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        index_path = p
                        break
            
            if not index_path:
                shutil.rmtree(extract_dir, ignore_errors=True)
                return {"success": False, "error": "No se encontró el índice de correos"}
            
            with open(index_path, 'r', errors='ignore') as f:
                lines = f.readlines()
            
            X_raw, y_raw = [], []
            parser = EmailParser()
            for line in lines[:10000]:
                parts = line.strip().split(" ../")
                if len(parts) < 2: continue
                
                label, rel_path = parts[0], parts[1]
                email_path = file_map.get(os.path.basename(rel_path))
                if not email_path:
                    server_path = os.path.normpath(os.path.join(os.path.dirname(index_path), rel_path))
                    if os.path.exists(server_path): email_path = server_path
                
                if email_path and os.path.exists(email_path):
                    content = parser.parse(email_path)
                    if content and "error" not in content and (content['subject'] or content['body']):
                        X_raw.append(" ".join(content['subject']) + " " + " ".join(content['body']))
                        y_raw.append(label)
            
            shutil.rmtree(extract_dir, ignore_errors=True)
            if not X_raw: return {"success": False, "error": "No se pudieron procesar los correos"}
            
            # Entrenamiento y evaluación
            split_idx = int(len(X_raw) * 0.8)
            vectorizer = CountVectorizer()
            X_train_vec = vectorizer.fit_transform(X_raw[:split_idx])
            X_test_vec = vectorizer.transform(X_raw[split_idx:])
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_vec, y_raw[:split_idx])
            test_acc = accuracy_score(y_raw[split_idx:], clf.predict(X_test_vec))
            
            # Preparación de resultados para el frontend
            vocab = vectorizer.get_feature_names_out()
            cv_html = pd.DataFrame(X_train_vec[:5, :10].toarray(), columns=vocab[:10]).to_html(classes='table table-sm table-hover small', border=0)
            
            words = X_raw[0].split()[:10]
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe_encoded = ohe.fit_transform([[w] for w in words])
            ohe_html = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out()).to_html(classes='table table-sm table-hover small', border=0)
            
            pred_df = pd.DataFrame({'Real': y_raw[split_idx:split_idx+10], 'Predicho': clf.predict(X_test_vec)[:10]})
            pred_html = pred_df.to_html(classes='table table-sm table-striped small', border=0, index=False)
            
            return {
                "success": True,
                "task": "05: Detección de Spam",
                "result": f"Modelo entrenado. Exactitud: {test_acc:.4f}",
                "metrics": [
                    {"label": "Accuracy", "value": f"{test_acc:.4f}"},
                    {"label": "Total Correos", "value": len(X_raw)},
                    {"label": "SPAM", "value": y_raw.count('spam')},
                    {"label": "HAM", "value": y_raw.count('ham')}
                ],
                "tables": [
                    {"title": "1. Muestra CountVectorizer", "content": cv_html},
                    {"title": "2. Muestra OneHotEncoder", "content": ohe_html},
                    {"title": "3. Comparativa de Predicciones", "content": pred_html}
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _load_arff_to_df(file_path):
        """Carga y normaliza un archivo ARFF."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Corregir formato de atributos nominales
        pattern = re.compile(r"(@ATTRIBUTE\s+.*?){\s*(.*?)\s*}", re.IGNORECASE)
        fixed_content = pattern.sub(lambda m: f"{m.group(1)}{{{re.sub(r's*,s*', ',', m.group(2).strip())}}}", content)
        
        dataset = arff.loads(fixed_content)
        return pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

    @staticmethod
    def analyze_06_viz(file_path):
        """Genera visualizaciones y estadísticas del dataset."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            # Muestreo para mayor velocidad
            df_stats = df.sample(10000) if len(df) > 10000 else df
            df_plot = df.sample(1000) if len(df) > 1000 else df
            
            head_html = df.head(10).to_html(classes='table table-sm table-hover table-bordered small', border=0)
            describe_html = df_stats.describe().to_html(classes='table table-sm table-hover table-bordered small', border=0)
            
            graphics = []
            if 'protocol_type' in df.columns:
                graphics.append({"title": "Protocol Type", "image": MLVisualizer.plot_categorical_count(df_stats, 'protocol_type')})
            
            numeric_df = df_plot.select_dtypes(include=[np.number])
            graphics.append({"title": "Histogramas (Top 12)", "image": MLVisualizer.plot_all_histograms(numeric_df[numeric_df.columns[:12]])})
            graphics.append({"title": "Matriz de Correlación", "image": MLVisualizer.plot_correlation_matrix(numeric_df)})
            
            return {
                "success": True,
                "task": "06: Visualización",
                "metrics": [{"label": "Registros", "value": len(df)}, {"label": "Atributos", "value": len(df.columns)}],
                "tables": [
                    {"title": "Primeros 10 registros", "content": head_html},
                    {"title": "Estadísticas descriptivas", "content": describe_html}
                ],
                "graphics": graphics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_07_split(file_path):
        """Divide el dataset en conjuntos de entrenamiento, validación y prueba (60/20/20)."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            strat = 'protocol_type' if 'protocol_type' in df.columns else None
            
            train, test_val = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat] if strat else None)
            val, test = train_test_split(test_val, test_size=0.5, random_state=42, stratify=test_val[strat] if strat else None)
            
            graphics = []
            if strat:
                graphics.append({"title": "Distribución Original", "image": MLVisualizer.plot_categorical_count(df, strat)})
                graphics.append({"title": "Train Set", "image": MLVisualizer.plot_categorical_count(train, strat)})
                graphics.append({"title": "Test Set", "image": MLVisualizer.plot_categorical_count(test, strat)})

            return {
                "success": True,
                "task": "07: División de Datos",
                "metrics": [
                    {"label": "Training Set", "value": len(train)},
                    {"label": "Validation Set", "value": len(val)},
                    {"label": "Test Set", "value": len(test)}
                ],
                "graphics": graphics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_08_prep(file_path):
        """Preparación: Limpieza, imputación y escalado."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            strat = 'protocol_type' if 'protocol_type' in df.columns else None
            train, _ = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat] if strat else None)
            X_train = train.drop('class', axis=1, errors='ignore').copy()

            # Introducción artificial de nulos para demostración
            if 'src_bytes' in X_train.columns:
                X_train.loc[(X_train['src_bytes'] > 400) & (X_train['src_bytes'] < 800), 'src_bytes'] = np.nan
            
            # Imputación
            X_num = X_train.select_dtypes(include=[np.number])
            imputer = SimpleImputer(strategy='median')
            df_imputed = pd.DataFrame(imputer.fit_transform(X_num), columns=X_num.columns, index=X_num.index)
            
            # Codificación
            X_cat = X_train.select_dtypes(exclude=[np.number])
            encoded_html = pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(X_cat)).head(10).to_html(classes='table table-sm', border=0) if not X_cat.empty else "<p>Sin categóricos</p>"
            
            return {
                "success": True,
                "task": "08: Preparación",
                "metrics": [{"label": "Nulos detectados", "value": int(X_train.isnull().any(axis=1).sum())}],
                "tables": [
                    {"title": "1. Filas con nulos", "content": X_train[X_train.isnull().any(axis=1)].head(10).to_html(classes='table table-sm', border=0)},
                    {"title": "2. Tras Imputación", "content": df_imputed.head(10).to_html(classes='table table-sm', border=0)},
                    {"title": "3. Tras Codificación", "content": encoded_html}
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_09_pipelines(file_path):
        """Uso de Pipelines personalizados para transformaciones."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            strat = 'protocol_type' if 'protocol_type' in df.columns else None
            train, _ = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat] if strat else None)
            X_train = train.drop('class', axis=1, errors='ignore').copy()

            # Pipeline full
            prep = DataFramePreparer()
            X_final = prep.fit_transform(X_train)
            
            return {
                "success": True,
                "task": "09: Pipelines",
                "metrics": [{"label": "Atributos finales", "value": X_final.shape[1]}],
                "tables": [{"title": "Resultado del Pipeline", "content": X_final.head(10).to_html(classes='table table-sm', border=0)}]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_10_eval(file_path):
        """Evaluación final del modelo de Regresión Logística."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            strat = 'protocol_type' if 'protocol_type' in df.columns else None
            train, rest = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat] if strat else None)
            val, test = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest[strat] if strat else None)
            
            X_train = train.drop("class", axis=1, errors='ignore')
            y_train = train["class"] if "class" in train.columns else None
            X_val = val.drop("class", axis=1, errors='ignore')
            y_val = val["class"] if "class" in val.columns else None

            # Procesamiento y entrenamiento
            prep = DataFramePreparer()
            X_train_prep = prep.fit_transform(X_train)
            X_val_prep = prep.transform(X_val)
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_prep, y_train)
            
            # Resultados
            y_pred = clf.predict(X_val_prep)
            comp_df = pd.DataFrame({'Real': y_val.values[:10], 'Predicho': y_pred[:10]})
            
            return {
                "success": True,
                "task": "10: Evaluación",
                "result": f"Exactitud Validación: {clf.score(X_val_prep, y_val):.4f}",
                "metrics": [
                    {"label": "Train Acc", "value": f"{clf.score(X_train_prep, y_train):.4f}"},
                    {"label": "Val Acc", "value": f"{clf.score(X_val_prep, y_val):.4f}"},
                    {"label": "Train Set", "value": len(X_train)},
                    {"label": "Val Set", "value": len(X_val)},
                    {"label": "Test Set", "value": len(test)}
                ],
                "tables": [
                    {"title": "1. Vista inicial Dataset", "content": df.head(10).to_html(classes='table table-sm', border=0)},
                    {"title": "2. Atributos de Entrenamiento", "content": X_train.head(10).to_html(classes='table table-sm', border=0)},
                    {"title": "3. Comparativa de Predicciones", "content": comp_df.to_html(classes='table table-sm', border=0, index=False)}
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
