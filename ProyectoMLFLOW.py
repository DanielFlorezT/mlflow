#!/usr/bin/env python
# -*- coding: utf-8 -*- Proyecto para pruebas MLFLOW
import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import boto3
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


# se configura el bucket de S3 donde se encuentra la data

bucket_name = 'proyecto-dvcstore-dsa-team4'
file_key = 'files/md5/94/0b416bb13a9b24bb5c9e1589284005'

s3 = boto3.client('s3')

# Cargar datos desde S3
def cargar_datos():
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(obj['Body'])
    return df
    
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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[columnas_relevantes], 
    df['default.payment.next.month'], 
    test_size=0.2, 
    random_state=42
)

# Normalizar los datos (escalar solo después de dividir)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo - parámetros
modelo = LogisticRegression(max_iter=500, penalty='l2', C=1.0, solver='saga')
modelo.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = modelo.predict(X_test)
y_pred_proba = modelo.predict_proba(X_test)[:, 1]

precision_modelo = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Configurar MLflow
mlflow.set_tracking_uri("http://44.211.153.243:5000")  
mlflow.set_experiment("Predicción de Riesgo de Impago")

# Registrar el modelo y las métricas
with mlflow.start_run():
    mlflow.log_param("max_iter", 500)
    mlflow.log_param("penalty", "l2")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "saga")
    
    mlflow.log_metric("precision", precision_modelo)
    mlflow.log_metric("auc_roc", auc_roc)
    
    # Registrar el modelo
    mlflow.sklearn.log_model(modelo, "modelo_riesgo")

# Función para predecir
def predecir(edad=None, limite=None, genero=None, educacion=None, estado=None, pay0=None):
    nueva_data = pd.DataFrame([[
        limite or valores_por_defecto['LIMIT_BAL'],
        edad or valores_por_defecto['AGE'],
        pay0 or valores_por_defecto['PAY_0'],
        genero or valores_por_defecto['SEX'],
        educacion or valores_por_defecto['EDUCATION'],
        estado or valores_por_defecto['MARRIAGE']
    ]], columns=columnas_relevantes)

    # Normalización y encoding
    nueva_data = scaler.transform(nueva_data)
    probabilidad = modelo.predict_proba(nueva_data)[0][1]
    if probabilidad <= 0.35:
        riesgo = "BAJO"
    elif 0.35 < probabilidad <= 0.65:
        riesgo = "MEDIO"
    else:
        riesgo = "ALTO"
    return probabilidad, riesgo

# Configurar dashboard en Dash
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Predicción de probabilidad del riesgo de incumplimiento de pago en clientes de tarjetas de crédito"),
    html.H2("Grupo 4"),
    html.H2("Despliegue de soluciones analíticas"),
    html.Div([
        html.P(
            "En el presente dashboard, puede calcularse el riesgo de que un cliente incumpla "
            "con sus obligaciones de tarjeta de crédito. Explore diferentes combinaciones "
            "de variables para identificar patrones y factores que más influyen en el riesgo."
        ),
    ]),
    html.Div([
        html.H3("Descripción de Variables"),
        html.P("Edad: Edad del cliente."),
        html.P("Límite de crédito: Monto máximo aprobado para el cliente."),
        html.P("Género: 1 para masculino, 2 para femenino."),
        html.P("Educación: Nivel de educación (1=Postgrado, 2=Universitario, etc.)."),
        html.P("Estado Civil: 1=Casado, 2=Soltero, etc."),
        html.P("PAY_0: Estado del pago en septiembre de 2005 (-1=pagó a tiempo, 1=atraso de 1 mes, 2=atraso de 2 meses, ..., 9=atraso de 9 meses o más).")
    ]),
    html.Div([
        html.H3("Panel de Entrada"),
        dcc.Input(id="input-edad", type="number", placeholder="Edad"),
        dcc.Input(id="input-limite", type="number", placeholder="Límite de crédito"),
        dcc.Input(id="input-genero", type="number", placeholder="Género"),
        dcc.Input(id="input-educacion", type="number", placeholder="Nivel de Educación"),
        dcc.Input(id="input-estado", type="number", placeholder="Estado Civil"),
        dcc.Input(id="input-pay0", type="number", placeholder="Historial de pagos"),
        html.Button("Predecir", id="btn-prediccion"),
    ]),
    html.Div([
        html.H3("Resultado de Predicción"),
        html.Div(id="indicadores-modelo", style={"marginBottom": "20px"}),
        dcc.Graph(id="roc-curve"),
        html.Div(id="resultado-prediccion", style={"fontSize": "18px", "marginTop": "20px"})
    ]),
    html.Div([
        html.H3("Factores de Influencia"),
        dcc.Graph(id="factores-influencia"),
    ]),
    html.Div([
        html.H3("Recomendación"),
        html.P(id="recomendacion")
    ]),
    html.Div([
        html.H4("Oscar Ardila - Guillermo Ariza - Paola Cifuentes - Daniel Flórez Thomas / Grupo 4"),
        html.P("Despliegue de Soluciones Analíticas"),
        html.P("Universidad de los Andes - Maestría en Inteligencia Analítica de Datos")
    ])
])

@app.callback(
    Output("indicadores-modelo", "children"),
    Input("btn-prediccion", "n_clicks")
)
def mostrar_indicadores(n_clicks):
    return f"Precisión del modelo: {precision_modelo:.2f}. AUC-ROC: {auc_roc:.2f}."

@app.callback(
    Output("resultado-prediccion", "children"),
    Input("btn-prediccion", "n_clicks"),
    State("input-edad", "value"),
    State("input-limite", "value"),
    State("input-genero", "value"),
    State("input-educacion", "value"),
    State("input-estado", "value"),
    State("input-pay0", "value")
)
def actualizar_prediccion(n_clicks, edad, limite, genero, educacion, estado, pay0):
    if n_clicks:
        probabilidad, riesgo = predecir(edad, limite, genero, educacion, estado, pay0)
        return f"Probabilidad de incumplimiento: {probabilidad:.2f}. Riesgo: {riesgo}."

@app.callback(
    Output("roc-curve", "figure"),
    Input("btn-prediccion", "n_clicks")
)
def graficar_roc(n_clicks):
    fpr, tpr, _ = roc_curve(y, modelo.predict_proba(X)[:, 1])
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode="lines"))
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return fig

@app.callback(
    Output("factores-influencia", "figure"),
    Input("btn-prediccion", "n_clicks")
)
def mostrar_factores(n_clicks):
    importancia = modelo.coef_[0]
    factores = pd.DataFrame({"Variable": columnas_modelo, "Importancia": importancia}).nlargest(3, "Importancia")
    fig = go.Figure(data=[go.Bar(x=factores["Variable"], y=factores["Importancia"])])
    fig.update_layout(title="Factores de Influencia", xaxis_title="Variables", yaxis_title="Importancia")
    return fig

@app.callback(
    Output("recomendacion", "children"),
    Input("btn-prediccion", "n_clicks")
)
def generar_recomendacion(n_clicks):
    return "Se recomienda establecer alertas tempranas y ajustar políticas para clientes con alto riesgo. Como puede verse los factores más incidentes en el riesgo son el estado de pago en el mes más reciente a la fecha de corte, seguidos de la edad y el sexo en una menor proporción."

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8060)


