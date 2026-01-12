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

try:
    from .visualizer import MLVisualizer
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from visualizer import MLVisualizer

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Shared Utilities ---

class MLStripper(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html_content):
    s = MLStripper()
    s.feed(html_content)
    return s.get_data()

# --- Spam Detection Logic (Notebook 05) ---

class EmailParser:
    def __init__(self):
        self.stemmer = nltk.PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = list(string.punctuation)

    def parse(self, email_path):
        """Parse an email."""
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
        """Transform a text string in tokens. perform two main actions,
        clean the punctuation symbols and do stemming of the text."""
        for c in self.punctuation:
            text = text.replace(c, "")
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")
        tokens = list(filter(None, text.split(" ")))
        # Stemming of the tokens
        return [self.stemmer.stem(w) for w in tokens if w not in self.stopwords]

# --- NSL-KDD Pipelines (Notebooks 08/09) ---

# --- Custom Transformers (Notebook 09) ---

class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
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
        
        # Consistent with Notebook 10
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('rbst_scaler', RobustScaler()),
        ])

        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoding(), cat_attribs),
        ])
        self._full_pipeline.fit(X)
        
        # Optimized column name retrieval
        cat_transformer = self._full_pipeline.named_transformers_['cat']
        cat_cols = cat_transformer._columns if hasattr(cat_transformer, '_columns') and cat_transformer._columns else []
        self._columns = num_attribs + list(cat_cols)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_prep = self._full_pipeline.transform(X_copy)
        return pd.DataFrame(X_prep, columns=self._columns, index=X_copy.index)

# --- ML Manager for Proyecto Final ---

class MLManager:
    @staticmethod
    def analyze_05_spam(file_path):
        """Logic from Notebook 05: Regresión Logística (Spam) - ZIP Processing"""
        try:
            if not zipfile.is_zipfile(file_path):
                return {"success": False, "error": "El archivo debe ser un ZIP que contenga la carpeta data/ con emails y el archivo index"}
            
            # Extract ZIP
            extract_dir = os.path.join(os.path.dirname(file_path), "extracted_" + os.path.basename(file_path))
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # 1. Map all files in ZIP for O(1) access by filename
            file_map = {}
            index_path = None
            found_files = []
            
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    found_files.append(f)
                    file_map[f] = os.path.join(root, f) # key: filename, value: full_path
                    if f.lower() in ['index', 'index.txt']:
                        index_path = os.path.join(root, f)
            
            # 2. Server-side Index Fallback
            if not index_path:
                # Ruta absoluta detectada: /home/aza/Documentos/simulacion/Datasets/datasets/trec07p/full/index.txt
                # Intentamos construirla dinamicamente o usar hardcode si es necesario en este entorno
                
                # Opcion 1: Subir 3 niveles desde back (back -> result -> simulacion -> Datasets)
                # BASE_DIR es .../proyectoFinal/back
                possible_paths = [
                    os.path.join(settings.BASE_DIR.parent.parent, "Datasets/datasets/trec07p/full/index.txt"), # .../simulacion/Datasets...
                    "/home/aza/Documentos/simulacion/Datasets/datasets/trec07p/full/index.txt" # Hardcoded fallback backup
                ]
                
                server_index_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        server_index_path = p
                        index_path = p
                        print(f"[DEBUG] Usando index del servidor: {index_path}")
                        break
            
            if not index_path:
                shutil.rmtree(extract_dir, ignore_errors=True)
                return {
                    "success": False, 
                    "error": f"No se encontró 'index' en el ZIP ni en el servidor. Buscado en {[p for p in possible_paths]}. Archivos recibidos: {found_files[:5]}..."
                }
            
            # 3. Read index lines
            with open(index_path, 'r', errors='ignore') as f:
                lines = f.readlines()
            
            # 4. Process emails
            X_raw = []
            y_raw = []
            parser = EmailParser()
            processed = 0
            errors = 0
            
            for line in lines[:10000]:  # Limit 10k
                line = line.strip()
                if not line: continue
                
                parts = line.split(" ../")
                if len(parts) < 2: continue
                
                label = parts[0]
                rel_path = parts[1] # e.g. data/inmail.1
                target_filename = os.path.basename(rel_path) # inmail.1
                
                # BUSCAR ARCHIVO: Primero en el mapa del ZIP (lo que subió el usuario)
                email_path = file_map.get(target_filename)
                
                # Si no está en el ZIP, intentar ruta absoluta (servidor)
                if not email_path:
                     # ../data/inmail.1 relative to full/index.txt -> trec07p/data/inmail.1
                     server_file_path = os.path.normpath(os.path.join(os.path.dirname(index_path), rel_path))
                     if os.path.exists(server_file_path):
                         email_path = server_file_path
                
                if email_path and os.path.exists(email_path):
                    content = parser.parse(email_path)
                    if content and "error" not in content and (content['subject'] or content['body']):
                        text = " ".join(content['subject']) + " " + " ".join(content['body'])
                        X_raw.append(text)
                        y_raw.append(label)
                        processed += 1
                else:
                    errors += 1
            
            # Cleanup
            shutil.rmtree(extract_dir, ignore_errors=True)
            
            if not X_raw:
                return {"success": False, "error": f"No se pudieron procesar emails. Index encontrado pero no coinciden archivos. Procesados: {processed}, Errores: {errors}"}
            
            # 5. ML Pipeline
            split_idx = int(len(X_raw) * 0.8)
            X_train_raw = X_raw[:split_idx]
            y_train = y_raw[:split_idx]
            X_test_raw = X_raw[split_idx:]
            y_test = y_raw[split_idx:]
            
            # Vectorize
            vectorizer = CountVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train_raw)
            X_test_vec = vectorizer.transform(X_test_raw)
            
            # Train Model
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train_vec, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_vec)
            test_acc = accuracy_score(y_test, y_pred)
            
            # Tables para frontend
            vocab = vectorizer.get_feature_names_out()
            
            # Tabla 1: CountVectorizer sample
            cv_df = pd.DataFrame(
                X_train_vec[:5, :10].toarray(),
                columns=vocab[:10]
            )
            cv_html = cv_df.to_html(classes='table table-sm table-hover small', border=0)
            
            # Tabla 2: OneHotEncoder sample
            words = X_raw[0].split()[:10]
            ohe = OneHotEncoder(handle_unknown='ignore')
            words_reshaped = [[w] for w in words]
            ohe.fit(words_reshaped)
            ohe_encoded = ohe.transform(words_reshaped).toarray()
            ohe_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out())
            ohe_html = ohe_df.to_html(classes='table table-sm table-hover small', border=0)
            
            # Tabla 3: Predicciones
            pred_df = pd.DataFrame({
                'Real': y_test[:10],
                'Predicho': y_pred[:10]
            })
            pred_html = pred_df.to_html(classes='table table-sm table-striped small', border=0, index=False)
            
            spam_count = y_raw.count('spam')
            ham_count = y_raw.count('ham')
            
            return {
                "success": True,
                "task": "05: Regresión Logística (Spam)",
                "result": f"Modelo entrenado con Accuracy: {test_acc:.4f}",
                "details": f"Se procesaron {len(X_raw)} emails ({len(X_train_raw)} train, {len(X_test_raw)} test). Siguiendo la lógica del Notebook 05.",
                "metrics": [
                    {"label": "Accuracy", "value": f"{test_acc:.4f}"},
                    {"label": "Total Correos", "value": len(X_raw)},
                    {"label": "SPAM", "value": spam_count},
                    {"label": "HAM", "value": ham_count},
                    {"label": "Vocabulario", "value": len(vocab)}
                ],
                "tables": [
                    {"title": "1. CountVectorizer (Primeros 10 tokens)", "content": cv_html},
                    {"title": "2. OneHotEncoder (Muestra)", "content": ohe_html},
                    {"title": "3. Predicciones vs Real (Test Set)", "content": pred_html}
                ],
                "graphics": []
            }
            
        except Exception as e:
            import traceback
            return {"success": False, "error": f"{str(e)}\n{traceback.format_exc()}"}


    @staticmethod
    def _load_arff_to_df(file_path):
        """Helper to load ARFF with normalization for nominal attribute spaces."""
        print(f"[DEBUG] Leyendo archivo físico: {file_path}")
        with open(file_path, 'r') as f:
            content = f.read()
        
        print(f"[DEBUG] Normalizando contenido ARFF (Regex)...")
        # Regex to normalize nominal attribute declarations:
        def clean_braces(match):
            attr_part = match.group(1)
            braces_content = match.group(2)
            cleaned = re.sub(r"\s*,\s*", ",", braces_content.strip())
            return f"{attr_part}{{{cleaned}}}"

        pattern = re.compile(r"(@ATTRIBUTE\s+.*?){\s*(.*?)\s*}", re.IGNORECASE)
        fixed_content = pattern.sub(clean_braces, content)
        
        print(f"[DEBUG] Parseando con liac-arff (esto puede tardar si el archivo es grande)...")
        dataset = arff.loads(fixed_content)
        attributes = [attr[0] for attr in dataset['attributes']]
        print(f"[DEBUG] Creando DataFrame de pandas...")
        return pd.DataFrame(dataset['data'], columns=attributes)

    @staticmethod
    def analyze_06_viz(file_path):
        """Logic from Notebook 06: Visualización de DataSet."""
        import time
        t0 = time.time()
        try:
            from sklearn.preprocessing import LabelEncoder
            print("[DEBUG] Cargando ARFF...")
            df = MLManager._load_arff_to_df(file_path)
            print(f"[DEBUG] ARFF cargado en {time.time()-t0:.2f}s. Filas: {len(df)}")
            
            # --- PUNTOS DE OPTIMIZACIÓN CRÍTICOS ---
            # Para las tablas (describe/head), usamos una muestra de 10k si es muy grande
            df_stats = df.sample(10000) if len(df) > 10000 else df
            # Para los gráficos, usamos solo 1000 puntos para que matplotlib sea instantáneo
            df_plot_sample = df.sample(1000) if len(df) > 1000 else df
            
            # 1. Statistical Tables (Notebook 06)
            print("[DEBUG] Renderizando tablas...")
            head_html = df.head(10).to_html(classes='table table-sm table-hover table-bordered small', border=0)
            describe_html = df_stats.describe().to_html(classes='table table-sm table-hover table-bordered small', border=0)
            
            counts_html = ""
            if 'protocol_type' in df.columns:
                counts_html = df['protocol_type'].value_counts().to_frame().to_html(classes='table table-sm table-hover table-bordered small', border=0)
            
            # 2. Pre-processing for correlation
            df_plot = df_plot_sample.copy()
            le = LabelEncoder()
            if 'class' in df_plot.columns:
                df_plot['class'] = le.fit_transform(df_plot['class'].astype(str))
            
            # 3. Graphics
            print("[DEBUG] Iniciando generación de gráficos rápidos...")
            numeric_df = df_plot.select_dtypes(include=[np.number])
            
            graphics = []
            # Categorical Count
            if 'protocol_type' in df.columns:
                graphics.append({"title": "Distribución de Protocol Type", "image": MLVisualizer.plot_categorical_count(df_stats, 'protocol_type')})
            
            # Histogramas: limitamos a las primeras 12 columnas numéricas para no saturar
            print("[DEBUG] -> Generando Histogramas (Top 12)...")
            cols_to_hist = numeric_df.columns[:12]
            graphics.append({"title": "Histogramas (Primeras 12 variables)", "image": MLVisualizer.plot_all_histograms(numeric_df[cols_to_hist])})
            
            # Matrix: con 1000 filas es muy rápido
            print("[DEBUG] -> Generando Matriz de Correlación...")
            graphics.append({"title": "Matriz de Correlación (Heatmap)", "image": MLVisualizer.plot_correlation_matrix(numeric_df)})
            
            # Scatter Matrix (specific attributes)
            scatter_attrs = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]
            existing_attrs = [a for a in scatter_attrs if a in df_plot.columns]
            if len(existing_attrs) > 1:
                print("[DEBUG] -> Generando Matriz de Dispersión...")
                graphics.append({"title": "Matriz de Dispersión", "image": MLVisualizer.plot_scatter_matrix(df_plot, existing_attrs)})
            
            print("[DEBUG] Análisis finalizado correctamente.")
            return {
                "success": True,
                "task": "06: Visualización de DataSet",
                "result": f"Análisis completo para {len(df)} registros.",
                "details": "Se han generado estadísticas descriptivas y visualizaciones dinámicas según el cuaderno 06.",
                "metrics": [
                    {"label": "Registros Total", "value": len(df)},
                    {"label": "Protocolos", "value": len(df['protocol_type'].unique())},
                    {"label": "Atributos", "value": len(df.columns)}
                ],
                "tables": [
                    {"title": "Primeros 10 Registros (df.head)", "content": head_html},
                    {"title": "Estadísticas Descriptivas (df.describe)", "content": describe_html},
                    {"title": "Conteo por Protocolo (value_counts)", "content": counts_html}
                ],
                "graphics": graphics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_07_split(file_path):
        """Logic from Notebook 07: División del DataSet (60/20/20 Split)."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            
            # Notebook 07: Split 60/40 then split the 40 into 50/50 (results in 60/20/20)
            # Seed 42 and stratification on protocol_type
            stratify_col = 'protocol_type' if 'protocol_type' in df.columns else None
            
            # Initial Split (60% Train, 40% Test)
            train_set, test_val_set = train_test_split(
                df, test_size=0.4, random_state=42, 
                stratify=df[stratify_col] if stratify_col else None
            )
            
            # Secondary Split (Split the 40% into 20% Val, 20% Test)
            val_set, test_set = train_test_split(
                test_val_set, test_size=0.5, random_state=42,
                stratify=test_val_set[stratify_col] if stratify_col else None
            )
            
            # Verificación Visual (Histograms of protocol_type distribution)
            graphics = []
            if stratify_col:
                graphics.append({"title": f"Distribución Original ({len(df)} recs)", "image": MLVisualizer.plot_categorical_count(df, stratify_col, f"Original: {stratify_col}")})
                graphics.append({"title": f"Train Set ({len(train_set)} recs)", "image": MLVisualizer.plot_categorical_count(train_set, stratify_col, f"Train: {stratify_col}")})
                graphics.append({"title": f"Validation Set ({len(val_set)} recs)", "image": MLVisualizer.plot_categorical_count(val_set, stratify_col, f"Validation: {stratify_col}")})
                graphics.append({"title": f"Test Set ({len(test_set)} recs)", "image": MLVisualizer.plot_categorical_count(test_set, stratify_col, f"Test: {stratify_col}")})

            return {
                "success": True,
                "task": "07: División del DataSet",
                "result": f"Datos divididos en 60/20/20 (Train/Val/Test).",
                "details": f"Particionado completo con semilla 42 y estratificación por '{stratify_col}'.",
                "metrics": [
                    {"label": "Longitud Training Set", "value": len(train_set)},
                    {"label": "Longitud Validación Set", "value": len(val_set)},
                    {"label": "Longitud Test Set", "value": len(test_set)},
                    {"label": "Total Registros", "value": len(df)}
                ],
                "graphics": graphics
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_08_prep(file_path):
        """Logic from Notebook 08: Preparación del DataSet."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            
            # 1. 60/20/20 Split
            stratify_col = 'protocol_type' if 'protocol_type' in df.columns else None
            train_set, test_val_set = train_test_split(
                df, test_size=0.4, random_state=42, 
                stratify=df[stratify_col] if stratify_col else None
            )
            val_set, test_set = train_test_split(
                test_val_set, test_size=0.5, random_state=42,
                stratify=test_val_set[stratify_col] if stratify_col else None
            )

            # 2. X/y Separation
            X_train = train_set.drop('class', axis=1, errors='ignore').copy()

            # 3. Artificial Introduction of Nulls
            if 'src_bytes' in X_train.columns and 'dst_bytes' in X_train.columns:
                X_train.loc[(X_train['src_bytes'] > 400) & (X_train['src_bytes'] < 800), 'src_bytes'] = np.nan
                X_train.loc[(X_train['dst_bytes'] > 500) & (X_train['dst_bytes'] < 2000), 'dst_bytes'] = np.nan

            # Table 1: Rows with Nulls
            null_rows = X_train[X_train.isnull().any(axis=1)].head(10)
            null_html = null_rows.to_html(classes='table table-sm table-hover small', border=0)

            # 4. Imputation
            X_train_num = X_train.select_dtypes(include=[np.number])
            imputer = SimpleImputer(strategy='median')
            X_train_num_imputed = imputer.fit_transform(X_train_num)
            df_imputed = pd.DataFrame(X_train_num_imputed, columns=X_train_num.columns, index=X_train_num.index)
            imputed_html = df_imputed.head(10).to_html(classes='table table-sm table-hover small', border=0)

            # 5. Encoding
            X_train_cat = X_train.select_dtypes(exclude=[np.number])
            oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            if not X_train_cat.empty:
                X_encoded = oh_encoder.fit_transform(X_train_cat)
                df_encoded = pd.DataFrame(X_encoded, columns=oh_encoder.get_feature_names_out(), index=X_train_cat.index)
                encoded_html = df_encoded.head(10).to_html(classes='table table-sm table-hover small', border=0)
            else:
                encoded_html = "<p>No hay atributos categóricos.</p>"

            # 6. Scaling
            scale_cols = [c for c in ['src_bytes', 'dst_bytes'] if c in X_train_num.columns]
            if scale_cols:
                robust_scaler = RobustScaler()
                X_scaled = robust_scaler.fit_transform(df_imputed[scale_cols])
                df_scaled = pd.DataFrame(X_scaled, columns=scale_cols, index=df_imputed.index)
                scaled_html = df_scaled.head(10).to_html(classes='table table-sm table-hover small', border=0)
            else:
                scaled_html = "<p>Columnas src_bytes/dst_bytes no encontradas.</p>"

            return {
                "success": True,
                "task": "08: Preparación del DataSet",
                "result": "Preparación completa según Cuaderno 08.",
                "details": f"Procesados {len(df)} registros: división, imputación, codificación y escalado.",
                "metrics": [
                    {"label": "Train Set", "value": len(train_set)},
                    {"label": "Valores Nulos Detectados", "value": X_train.isnull().any(axis=1).sum()}
                ],
                "tables": [
                    {"title": "1. Filas con Valores Nulos Encontrados", "content": null_html},
                    {"title": "2. Tras Imputación (Median)", "content": imputed_html},
                    {"title": "3. Tras Codificación (OneHot)", "content": encoded_html},
                    {"title": "4. Tras Escalado (RobustScaler)", "content": scaled_html}
                ],
                "graphics": []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_09_pipelines(file_path):
        """Logic from Notebook 09: Pipelines Personalizados."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            
            # 1. Split and X/y Separation (Notebook 09 flow)
            stratify_col = 'protocol_type' if 'protocol_type' in df.columns else None
            train_set, test_val_set = train_test_split(
                df, test_size=0.4, random_state=42, 
                stratify=df[stratify_col] if stratify_col else None
            )
            X_train = train_set.drop('class', axis=1, errors='ignore').copy()

            # Artificial Nulls for demonstration (if present)
            if 'src_bytes' in X_train.columns and 'dst_bytes' in X_train.columns:
                X_train.loc[(X_train['src_bytes'] > 400) & (X_train['src_bytes'] < 800), 'src_bytes'] = np.nan
                X_train.loc[(X_train['dst_bytes'] > 500) & (X_train['dst_bytes'] < 2000), 'dst_bytes'] = np.nan

            # table_results list to store transformation stages
            tables = []

            # Stage 1: DeleteNanRows
            delete_nan = DeleteNanRows()
            X_prep_1 = delete_nan.fit_transform(X_train)
            tables.append({"title": "1. Custom: DeleteNanRows (Post-dropna)", "content": X_prep_1.head(10).to_html(classes='table table-sm table-hover small', border=0)})

            # Stage 2: CustomScaler
            scale_cols = [c for c in ['src_bytes', 'dst_bytes'] if c in X_prep_1.columns]
            if scale_cols:
                custom_scaler = CustomScaler(scale_cols)
                X_prep_2 = custom_scaler.fit_transform(X_prep_1)
                tables.append({"title": f"2. Custom: CustomScaler ({', '.join(scale_cols)})", "content": X_prep_2.head(10)[scale_cols].to_html(classes='table table-sm table-hover small', border=0)})
            else:
                X_prep_2 = X_prep_1

            # Stage 3: Full Pipeline (Professional Integration)
            num_attribs = list(X_train.select_dtypes(exclude=['object']))
            cat_attribs = list(X_train.select_dtypes(include=['object']))
            
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('rbst_scaler', RobustScaler()),
            ])

            full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_attribs),
            ])

            X_final = full_pipeline.fit_transform(X_train)
            
            # Reconstruct DataFrame for display
            try:
                # Attempt to get dummies style column names
                cat_encoder = full_pipeline.named_transformers_['cat']
                cat_cols = list(cat_encoder.get_feature_names_out(cat_attribs))
                final_cols = num_attribs + cat_cols
                df_final = pd.DataFrame(X_final, columns=final_cols, index=X_train.index)
                tables.append({"title": "3. Full Pipeline: ColumnTransformer Kết quả", "content": df_final.head(10).to_html(classes='table table-sm table-hover small', border=0)})
            except:
                tables.append({"title": "3. Full Pipeline: Kết quả (Raw Array)", "content": f"<p>Array shape: {X_final.shape}</p>"})

            return {
                "success": True,
                "task": "09: Pipelines Personalizados",
                "result": "Pipeline de transformación diseñado con éxito.",
                "details": "Se han implementado transformadores personalizados y una integración con ColumnTransformer.",
                "metrics": [
                    {"label": "Atributos Numéricos", "value": len(num_attribs)},
                    {"label": "Atributos Categóricos", "value": len(cat_attribs)},
                    {"label": "Total Features Final", "value": X_final.shape[1]}
                ],
                "tables": tables,
                "graphics": []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def analyze_10_eval(file_path):
        """Logic from Notebook 10: Evaluación de Resultados."""
        try:
            df = MLManager._load_arff_to_df(file_path)
            
            # 1. Stratified Split (60/20/20)
            stratify_col = 'protocol_type' if 'protocol_type' in df.columns else None
            train_set, test_val_set = train_test_split(
                df, test_size=0.4, random_state=42, 
                stratify=df[stratify_col] if stratify_col else None
            )
            val_set, test_set = train_test_split(
                test_val_set, test_size=0.5, random_state=42,
                stratify=test_val_set[stratify_col] if stratify_col else None
            )

            # 2. X/y Separation for Train and Validation
            X_train = train_set.drop("class", axis=1, errors='ignore')
            y_train = train_set["class"].copy() if "class" in train_set.columns else None
            X_val = val_set.drop("class", axis=1, errors='ignore')
            y_val = val_set["class"].copy() if "class" in val_set.columns else None
            
            X_df = df.drop("class", axis=1, errors='ignore')

            # 3. Transform Data
            data_preparer = DataFramePreparer()
            data_preparer.fit(X_train) # Fit ONLY on training data (ML Best Practice & Faster)
            
            X_train_prep = data_preparer.transform(X_train)
            X_val_prep = data_preparer.transform(X_val)

            # 4. Train Logistic Regression
            clf = LogisticRegression(max_iter=1000, solver='lbfgs') # Reduced max_iter for speed
            clf.fit(X_train_prep, y_train)

            # 5. Metrics
            train_acc = clf.score(X_train_prep, y_train)
            val_acc = clf.score(X_val_prep, y_val)
            
            # 6. Prediction Comparison Table (Sample of 10)
            y_val_pred = clf.predict(X_val_prep)
            comparison_df = pd.DataFrame({
                'Real': y_val.values[:10] if y_val is not None else [],
                'Predicho': y_val_pred[:10]
            })
            comparison_html = comparison_df.to_html(classes='table table-sm table-hover table-striped small', border=0, index=False)

            return {
                "success": True,
                "task": "10: Evaluación de Resultados",
                "result": f"Modelo Logístico entrenado (Accuracy Val: {val_acc:.4f})",
                "details": "Entrenamiento de regresión logística con max_iter=5000 sobre conjunto de datos estratificado.",
                "metrics": [
                    {"label": "Training Accuracy", "value": f"{train_acc:.4f}"},
                    {"label": "Validation Accuracy", "value": f"{val_acc:.4f}"},
                    {"label": "Registros Entrenamiento", "value": len(X_train_prep)},
                    {"label": "Registros Validación", "value": len(X_val_prep)}
                ],
                "tables": [
                    {"title": "Comparativa de Predicciones (Acrúa vs Predicho - Muestra 10)", "content": comparison_html}
                ],
                "graphics": []
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
