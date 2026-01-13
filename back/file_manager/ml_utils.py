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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
            for line in lines[:2000]: # Reducido de 10000 a 2000 para velocidad
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
            
            y_test_real = y_raw[split_idx:]
            y_test_pred = clf.predict(X_test_vec)
            
            # Cálculo de las 7 métricas estándar
            acc = accuracy_score(y_test_real, y_test_pred)
            prec = precision_score(y_test_real, y_test_pred, pos_label='spam', average='binary', zero_division=0)
            rec = recall_score(y_test_real, y_test_pred, pos_label='spam', average='binary', zero_division=0)
            f1 = f1_score(y_test_real, y_test_pred, pos_label='spam', average='binary', zero_division=0)
            
            total = len(y_test_real)
            correct = int((np.array(y_test_real) == np.array(y_test_pred)).sum())
            incorrect = total - correct
            
            # Preparación de resultados
            raw_data_df = pd.DataFrame({'Texto': X_raw[:10], 'Etiqueta': y_raw[:10]})
            raw_html = raw_data_df.to_html(classes='table table-sm table-hover small', border=0, index=False)
            
            vocab = vectorizer.get_feature_names_out()
            cv_html = pd.DataFrame(X_train_vec[:5, :10].toarray(), columns=vocab[:10]).to_html(classes='table table-sm table-hover small', border=0)
            
            # Paso intermedio: Mostrar tokenización de un correo real
            sample_email_content = X_raw[0]
            sample_tokens = sample_email_content.split()[:20]
            token_df = pd.DataFrame({'Token Original': sample_tokens})
            # El parser ya hizo stemming y limpieza, así que mostramos lo que hay en X_raw
            token_html = token_df.to_html(classes='table table-sm table-dark small', border=0)

            words = X_raw[0].split()[:10]
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe_encoded = ohe.fit_transform([[w] for w in words])
            ohe_html = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out()).to_html(classes='table table-sm table-hover small', border=0)
            
            pred_df = pd.DataFrame({'Real': y_raw[split_idx:split_idx+10], 'Predicho': clf.predict(X_test_vec)[:10]})
            pred_html = pred_df.to_html(classes='table table-sm table-striped small', border=0, index=False)
            
            return {
                "success": True,
                "task": "05: Detección de Spam",
                "result": f"Modelo entrenado con éxito. Accuracy: {acc:.4f}",
                "metrics": [
                    {"label": "ACCURACY", "value": f"{acc*100:.2f}%"},
                    {"label": "PRECISION", "value": f"{prec*100:.2f}%"},
                    {"label": "RECALL", "value": f"{rec*100:.2f}%"},
                    {"label": "F1-SCORE", "value": f"{f1*100:.2f}%"},
                    {"label": "TOTAL PREDICCIONES", "value": total},
                    {"label": "CORRECTAS", "value": correct},
                    {"label": "INCORRECTAS", "value": incorrect}
                ],
                "tables": [
                    {"title": "1. Muestra de Datos Originales (Primeras 10 filas)", "content": raw_html},
                    {"title": "2. Pre-procesamiento: Tokens del primer correo (Muestra)", "content": token_html},
                    {"title": "3. Vectorización: CountVectorizer (Frecuencias)", "content": cv_html},
                    {"title": "4. Ejemplo OneHotEncoder sobre palabras", "content": ohe_html},
                    {"title": "5. Evaluación: Comparativa de Predicciones", "content": pred_html}
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _load_arff_to_df(file_path, max_rows=None):
        """Carga y normaliza un archivo ARFF con muestreo opcional."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Corregir formato de atributos nominales
        pattern = re.compile(r"(@ATTRIBUTE\s+.*?){\s*(.*?)\s*}", re.IGNORECASE)
        fixed_content = pattern.sub(lambda m: f"{m.group(1)}{{{re.sub(r's*,s*', ',', m.group(2).strip())}}}", content)
        
        dataset = arff.loads(fixed_content)
        df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
        
        if max_rows and len(df) > max_rows:
            return df.sample(max_rows, random_state=42)
        return df

    @staticmethod
    def analyze_06_viz(file_path):
        """Genera visualizaciones y estadísticas del dataset."""
        try:
            # Muestreo agresivo para visualización (8,000 para rapidez total)
            df = MLManager._load_arff_to_df(file_path, max_rows=8000)
            
            # Codificación de la clase para permitir correlación
            from sklearn.preprocessing import LabelEncoder
            le_class = LabelEncoder()
            if 'class' in df.columns:
                df['class_encoded'] = le_class.fit_transform(df['class'])
            
            # Muestreo adicional para gráficos pesados
            df_stats = df.sample(4000) if len(df) > 4000 else df
            df_plot = df.sample(1000) if len(df) > 1000 else df
            
            head_html = df.head(10).to_html(classes='table table-sm table-hover table-bordered small', border=0)
            describe_html = df_stats.describe().to_html(classes='table table-sm table-hover table-bordered small', border=0)
            
            graphics = []
            if 'protocol_type' in df.columns:
                graphics.append({"title": "Protocol Type", "image": MLVisualizer.plot_categorical_count(df_stats, 'protocol_type')})
            
            numeric_df = df_plot.select_dtypes(include=[np.number])
            # 1. Mosaico de Histogramas (Limitado a 12 para velocidad)
            top_cols = numeric_df.columns[:12]
            graphics.append({"title": "Mosaico de Histogramas (Top 12 Atributos)", "image": MLVisualizer.plot_all_histograms(numeric_df[top_cols])})
            
            # 2. Matriz de Dispersión (Atributos clave del notebook)
            key_attrs = ["same_srv_rate", "dst_host_srv_count", "dst_host_same_srv_rate"]
            if 'class_encoded' in numeric_df.columns:
                key_attrs.append('class_encoded')
            
            present_attrs = [a for a in key_attrs if a in numeric_df.columns]
            if len(present_attrs) > 1:
                graphics.append({"title": "Matriz de Dispersión (Correlaciones Visuales)", "image": MLVisualizer.plot_scatter_matrix(numeric_df, present_attrs)})
            
            graphics.append({"title": "Matriz de Correlación General", "image": MLVisualizer.plot_correlation_matrix(numeric_df)})
            
            null_counts = df.isnull().sum()
            null_df = pd.DataFrame({'Atributo': null_counts.index, 'Valores Nulos': null_counts.values})
            null_html = null_df[null_df['Valores Nulos'] > 0].to_html(classes='table table-sm table-warning small', border=0, index=False)
            if null_df['Valores Nulos'].sum() == 0:
                null_html = "<p class='text-success'>No se detectaron valores nulos en el dataset.</p>"
            
            protocol_counts = df['protocol_type'].value_counts()
            protocol_df = pd.DataFrame({'Protocolo': protocol_counts.index, 'Frecuencia': protocol_counts.values})
            protocol_html = protocol_df.to_html(classes='table table-sm table-info small', border=0, index=False)

            # 3. Correlación con el Target (Como en el notebook)
            corr_target_html = ""
            if 'class_encoded' in df.columns:
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                corr_target = corr_matrix['class_encoded'].sort_values(ascending=False).to_frame()
                corr_target.columns = ['Correlación con la Clase']
                corr_target_html = corr_target.to_html(classes='table table-sm table-dark small', border=0)

            return {
                "success": True,
                "task": "06: Visualización",
                "metrics": [
                    {"label": "Total Registros", "value": len(df)},
                    {"label": "Atributos", "value": len(df.columns)},
                    {"label": "Valores Nulos", "value": int(df.isnull().sum().sum())},
                    {"label": "Filas Duplicadas", "value": int(df.duplicated().sum())},
                    {"label": "Memoria (MB)", "value": f"{df.memory_usage().sum() / 1024 / 1024:.2f}"},
                    {"label": "Clases Únicas", "value": len(df['class'].unique()) if 'class' in df.columns else 0},
                    {"label": "Balance Clase (%)", "value": f"{df['class'].value_counts(normalize=True).iloc[0]*100:.1f}" if 'class' in df.columns else "N/A"}
                ],
                "tables": [
                    {"title": "1. Muestra de los primeros 10 registros", "content": head_html},
                    {"title": "2. Estadísticas Descriptivas (Numéricas)", "content": describe_html},
                    {"title": "3. Análisis de Correlación con la Clase (Target)", "content": corr_target_html},
                    {"title": "4. Detección de Valores Nulos", "content": null_html},
                    {"title": "5. Distribución Exacta de Protocolos", "content": protocol_html}
                ],
                "graphics": graphics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_07_split(file_path):
        """Divide el dataset en conjuntos de entrenamiento, validación y prueba."""
        try:
            df = MLManager._load_arff_to_df(file_path, max_rows=8000)
            strat = 'protocol_type' if 'protocol_type' in df.columns else None
            
            train, test_val = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat] if strat else None)
            val, test = train_test_split(test_val, test_size=0.5, random_state=42, stratify=test_val[strat] if strat else None)
            
            graphics = []
            if strat:
                graphics.append({"title": "Distribución: Conjunto Original", "image": MLVisualizer.plot_categorical_count(df, strat, "Dataset Completo")})
                graphics.append({"title": "Distribución: Conjunto Entrenamiento (Train)", "image": MLVisualizer.plot_categorical_count(train, strat, "Train Set")})
                graphics.append({"title": "Distribución: Conjunto Validación (Val)", "image": MLVisualizer.plot_categorical_count(val, strat, "Val Set")})
                graphics.append({"title": "Distribución: Conjunto Prueba (Test)", "image": MLVisualizer.plot_categorical_count(test, strat, "Test Set")})

            split_summary = pd.DataFrame({
                'Conjunto': ['Entrenamiento (Train)', 'Validación (Val)', 'Prueba (Test)', 'Total'],
                'Registros': [len(train), len(val), len(test), len(df)],
                'Porcentaje': [f"{(len(train)/len(df))*100:.1f}%", f"{(len(val)/len(df))*100:.1f}%", f"{(len(test)/len(df))*100:.1f}%", '100%']
            })
            summary_html = split_summary.to_html(classes='table table-sm table-bordered table-hover small', border=0, index=False)

            return {
                "success": True,
                "task": "07: División de Datos",
                "result": f"Dataset dividido exitosamente aplicando estratificación sobre '{strat or 'N/A'}'.",
                "metrics": [
                    {"label": "Total Registros", "value": len(df)},
                    {"label": "Train Set", "value": len(train)},
                    {"label": "Val Set", "value": len(val)},
                    {"label": "Test Set", "value": len(test)},
                    {"label": "Train %", "value": f"{(len(train)/len(df))*100:.1f}%"},
                    {"label": "Val %", "value": f"{(len(val)/len(df))*100:.1f}%"},
                    {"label": "Test %", "value": f"{(len(test)/len(df))*100:.1f}%"}
                ],
                "tables": [
                    {"title": "1. Resumen de la División (60/20/20)", "content": summary_html},
                    {"title": "2. Muestra: Conjunto Entrenamiento (Train)", "content": train.head(10).to_html(classes='table table-sm small', border=0)},
                    {"title": "3. Muestra: Conjunto Prueba (Test)", "content": test.head(10).to_html(classes='table table-sm small', border=0)}
                ],
                "graphics": graphics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_08_prep(file_path):
        """Preparación: Limpieza, imputación y escalado."""
        try:
            df = MLManager._load_arff_to_df(file_path, max_rows=8000)
            strat = 'protocol_type' if 'protocol_type' in df.columns else None
            train, _ = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat] if strat else None)
            X_train = train.drop('class', axis=1, errors='ignore').copy()
            
            # Usar una muestra de entrenamiento más pequeña para pre-ajustar
            X_train_small = X_train.sample(4000) if len(X_train) > 4000 else X_train

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
            
            # Escalado de demostración
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns, index=df_imputed.index)
            
            # Preparación de resultados para comparación de imputación
            if 'src_bytes' in X_train.columns:
                null_indices = X_train[X_train['src_bytes'].isnull()].index[:5]
                compare_imputation = pd.DataFrame({
                    'Original (con Nulos)': X_train.loc[null_indices, 'src_bytes'],
                    'Tras Imputación': df_imputed.loc[null_indices, 'src_bytes']
                })
                compare_html = compare_imputation.to_html(classes='table table-sm table-dark small', border=0)
            else:
                compare_html = "<p>No hay datos suficientes para mostrar comparación de imputación.</p>"

            return {
                "success": True,
                "task": "08: Preparación",
                "metrics": [
                    {"label": "Total Registros", "value": len(df)},
                    {"label": "Nulos Detectados", "value": int(X_train.isnull().any(axis=1).sum())},
                    {"label": "Nulos Imputados", "value": int(X_train.isnull().sum().sum())},
                    {"label": "Atrib. Numéricos", "value": len(X_num.columns)},
                    {"label": "Atrib. Categóricos", "value": len(X_cat.columns)},
                    {"label": "Columnas Finales", "value": X_scaled.shape[1]},
                    {"label": "Calidad Datos (%)", "value": f"{(1 - X_train.isnull().sum().sum() / X_train.size)*100:.1f}%"}
                ],
                "tables": [
                    {"title": "1. Filas con valores nulos (Muestra original)", "content": X_train[X_train.isnull().any(axis=1)].head(10).to_html(classes='table table-sm small', border=0)},
                    {"title": "2. Comparativa de Imputación (Antes vs Después)", "content": compare_html},
                    {"title": "3. Ejemplo de Codificación OneHot (Protocolos/Servicios)", "content": encoded_html},
                    {"title": "4. Resultado final tras Escalado (RobustScaler)", "content": X_scaled.head(10).to_html(classes='table table-sm small', border=0)}
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_09_pipelines(file_path):
        """Uso de Pipelines personalizados para transformaciones."""
        try:
            df = MLManager._load_arff_to_df(file_path, max_rows=20000)
            strat = 'protocol_type' if 'protocol_type' in df.columns else None
            train, _ = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat] if strat else None)
            X_train = train.drop('class', axis=1, errors='ignore').copy()

            # Pipeline full
            prep = DataFramePreparer()
            X_final = prep.fit_transform(X_train)
            
            # Entrenamiento ultra-rápido para obtener métricas
            y_train = train['class']
            clf = LogisticRegression(max_iter=300, solver='liblinear') 
            clf.fit(X_final, y_train)
            
            y_pred = clf.predict(X_final)
            acc = accuracy_score(y_train, y_pred)
            prec = precision_score(y_train, y_pred, pos_label='anomaly', average='binary', zero_division=0)
            rec = recall_score(y_train, y_pred, pos_label='anomaly', average='binary', zero_division=0)
            f1 = f1_score(y_train, y_pred, pos_label='anomaly', average='binary', zero_division=0)
            
            total = len(y_train)
            correct = int((y_train == y_pred).sum())
            incorrect = total - correct
            
            # Detalle de la estructura del Pipeline
            pipeline_steps = pd.DataFrame({
                'Paso': ['Imputer', 'Scaler', 'OneHotEncoder', 'Final Assembler'],
                'Descripción': [
                    'Sustitución de valores nulos por la mediana',
                    'Escalado robusto para manejar outliers',
                    'Codificación de variables categóricas',
                    'Combinación de todas las columnas procesadas'
                ]
            })
            pipeline_html = pipeline_steps.to_html(classes='table table-sm table-info', border=0, index=False)

            return {
                "success": True,
                "task": "09: Pipelines",
                "result": "Pipeline de transformación aplicado exitosamente con evaluación de entrenamiento.",
                "metrics": [
                    {"label": "ACCURACY", "value": f"{acc*100:.2f}%"},
                    {"label": "PRECISION", "value": f"{prec*100:.2f}%"},
                    {"label": "RECALL", "value": f"{rec*100:.2f}%"},
                    {"label": "F1-SCORE", "value": f"{f1*100:.2f}%"},
                    {"label": "TOTAL PREDICCIONES", "value": total},
                    {"label": "CORRECTAS", "value": correct},
                    {"label": "INCORRECTAS", "value": incorrect}
                ],
                "tables": [
                    {"title": "1. Estructura y Configuración del Pipeline", "content": pipeline_html},
                    {"title": "2. Datos de Entrada (X_train - Primeras 10 filas)", "content": X_train.head(10).to_html(classes='table table-sm small', border=0)},
                    {"title": "3. Resultado Final tras Pipeline Completo (Consolidado)", "content": X_final.head(10).to_html(classes='table table-sm table-striped small', border=0)}
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_10_eval(file_path):
        """Evaluación final del modelo de Regresión Logística."""
        try:
            df = MLManager._load_arff_to_df(file_path, max_rows=20000)
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
            
            clf = LogisticRegression(max_iter=300, solver='liblinear') 
            clf.fit(X_train_prep, y_train)
            
            # Resultados
            y_pred = clf.predict(X_val_prep)
            y_proba = clf.predict_proba(X_val_prep)[:, 1] # Probabilidad de la clase positiva
            comp_df = pd.DataFrame({'Real': y_val.values[:10], 'Predicho': y_pred[:10]})
            
            # Codificación binaria para métricas (si es necesario)
            pos_label = 'anomaly' 
            y_val_bin = (y_val == pos_label).astype(int)
            
            # Métricas extendidas
            precision = precision_score(y_val, y_pred, pos_label='anomaly', average='binary')
            recall = recall_score(y_val, y_pred, pos_label='anomaly', average='binary')
            f1 = f1_score(y_val, y_pred, pos_label='anomaly', average='binary')
            accuracy = accuracy_score(y_val, y_pred)
            
            total = len(y_val)
            correct = int((y_val == y_pred).sum())
            incorrect = total - correct

            metrics_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision (Anomaly)', 'Recall (Anomaly)', 'F1-Score (Anomaly)'],
                'Valor': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
            })
            metrics_html = metrics_df.to_html(classes='table table-sm table-success', border=0, index=False)
            
            # Gráficos
            graphics = []
            labels = sorted(y_val.unique())
            cm = confusion_matrix(y_val, y_pred, labels=labels)
            graphics.append({"title": "Matriz de Confusión (Set de Validación)", "image": MLVisualizer.plot_confusion_matrix(cm, labels)})
            
            # ROC Curve
            graphics.append({"title": "Curva ROC (Receiver Operating Characteristic)", "image": MLVisualizer.plot_roc_curve(y_val_bin, y_proba)})

            return {
                "success": True,
                "task": "10: Evaluación",
                "result": f"Modelo evaluado exitosamente. Accuracy final: {accuracy:.4f}",
                "metrics": [
                    {"label": "ACCURACY", "value": f"{accuracy*100:.2f}%"},
                    {"label": "PRECISION", "value": f"{precision*100:.2f}%"},
                    {"label": "RECALL", "value": f"{recall*100:.2f}%"},
                    {"label": "F1-SCORE", "value": f"{f1*100:.2f}%"},
                    {"label": "TOTAL PREDICCIONES", "value": total},
                    {"label": "CORRECTAS", "value": correct},
                    {"label": "INCORRECTAS", "value": incorrect}
                ],
                "tables": [
                    {"title": "1. Tabla de Métricas Detalladas", "content": metrics_html},
                    {"title": "2. Vista inicial del Dataset (Muestra)", "content": df.head(10).to_html(classes='table table-sm small', border=0)},
                    {"title": "3. Comparativa de Predicciones sobre Validación", "content": comp_df.to_html(classes='table table-sm table-hover small', border=0, index=False)}
                ],
                "graphics": graphics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
