import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

def get_base64_plot():
    """Convert the current matplotlib plot to a base64 string."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

class MLVisualizer:
    @staticmethod
    def plot_correlation_matrix(df):
        """Generate a heatmap of the correlation matrix."""
        df_numeric = df.select_dtypes(include=['number'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_numeric.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title("Matriz de Correlación")
        return get_base64_plot()

    @staticmethod
    def plot_feature_distribution(df, column):
        """Generate a histogram/distribution plot for a feature."""
        plt.figure(figsize=(8, 5))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribución de {column}")
        return get_base64_plot()

    @staticmethod
    def plot_clusters(df, x_col, y_col, labels):
        """Generate a scatter plot for clusters."""
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col], c=labels, cmap='viridis', alpha=0.5)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Visualización de Clústeres (K-Means)")
        plt.colorbar(label='Cluster ID')
        return get_base64_plot()

    @staticmethod
    def plot_confusion_matrix(cm, labels):
        """Generate a heatmap for a confusion matrix."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.title("Matriz de Confusión")
        return get_base64_plot()
    @staticmethod
    def plot_scatter_matrix(df, attributes):
        """Generate a scatter matrix for specific attributes."""
        plt.figure(figsize=(12, 8))
        scatter_matrix(df[attributes], figsize=(12, 8), alpha=0.5)
        return get_base64_plot()

    @staticmethod
    def plot_all_histograms(df):
        """Generate a grid of histograms for all numeric columns."""
        df.hist(bins=50, figsize=(20, 15))
        plt.tight_layout()
        return get_base64_plot()

    @staticmethod
    def plot_categorical_count(df, column, title=None):
        """Generate a histogram for a categorical column."""
        plt.figure(figsize=(8, 5))
        df[column].value_counts().plot(kind='bar')
        plt.title(title or f"Distribución de {column}")
        plt.grid(axis='y', alpha=0.3)
        return get_base64_plot()
