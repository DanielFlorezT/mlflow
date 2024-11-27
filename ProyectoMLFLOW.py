#!/usr/bin/env python
# -*- coding: utf-8 -*- Proyecto para pruebas MLFLOW
import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Configuración del bucket y archivo en S3
bucket_name = 'proyecto-dvcstore-dsa-team4'
file_key = 'files/md5/94/0b416bb13a9b24bb5c9e1589284005'

# Configurar cliente S3 usando credenciales preconfiguradas
s3 = boto3.client('s3')

# Cargar datos desde S3
def cargar_datos():
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(obj['Body'])
        return df
    except Exception as e:
        print(f"Error al cargar datos desde S3: {e}")
        raise

# Cargar y preparar datos
df = cargar_datos()
columnas_relevantes = ['LIMIT_BAL', 'AGE', 'PAY_0', 'SEX', 'EDUCATION', 'MARRIAGE']
valores_por_defecto = {
    'LIMIT_BAL': df['LIMIT_BAL'].mean(),
    'AGE': df['AGE'].mean(),
    'PAY_0': 0,
    'SEX': 2,
    'EDUCATION': 2,
    'MARRIAGE': 2
}

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df[columnas_relevantes], 
    df['default.payment.next.month'], 
    test_size=0.2, 
    random_state=42
)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo
modelo = LogisticRegression(max_iter=500, penalty='l2', C=1.0, solver='saga')
modelo.fit(X_train, y_train)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)[:, 1]

precision_modelo = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Configurar MLflow
mlflow.set_tracking_uri("http://34.203.247.28:8080/")  
mlflow.set_experiment("Predicción de Riesgo de Impago")

# Registrar el modelo y métricas
with mlflow.start_run():
    mlflow.log_param("max_iter", 500)
    mlflow.log_param("penalty", "l2")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "saga")
    
    mlflow.log_metric("precision", precision_modelo)
    mlflow.log_metric("auc_roc", auc_roc)
    
    # Registrar el modelo
    mlflow.sklearn.log_model(modelo, "modelo_riesgo")


