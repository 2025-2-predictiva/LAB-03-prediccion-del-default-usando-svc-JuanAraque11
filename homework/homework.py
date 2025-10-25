# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import json, gzip, os, pickle, zipfile
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Variables globales - Importantes
RUTA_BASE_INPUT = Path("files/input")
RUTA_BASE_OUTPUT = Path("files")
TRAIN_ZIP_PATH = RUTA_BASE_INPUT / "train_data.csv.zip"
TEST_ZIP_PATH = RUTA_BASE_INPUT / "test_data.csv.zip"
NOMBRE_TRAIN = "train_default_of_credit_card_clients.csv"
NOMBRE_TEST = "test_default_of_credit_card_clients.csv"
MODEL_PATH = RUTA_BASE_OUTPUT / "models/model.pkl.gz"
METRICS_PATH = RUTA_BASE_OUTPUT / "output/metrics.json"

CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERICAL_COLS = [
    "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]
TARGET_COL = "default"


def cargar_csv_desde_zip(ruta_zip: Path, nombre_interno: str) -> pd.DataFrame:
    with zipfile.ZipFile(ruta_zip) as zf, zf.open(nombre_interno) as f:
        return pd.read_csv(f)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.drop(columns="ID")
          .rename(columns={"default payment next month": TARGET_COL})
          .dropna()
    )
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df


def crear_modelo_y_busqueda() -> GridSearchCV:
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", StandardScaler(), NUMERICAL_COLS)
    ], remainder="passthrough")

    pipe = Pipeline([
        ("prep", preproc),
        ("pca", PCA()),
        ("kbest", SelectKBest(score_func=f_classif)),
        ("svc", SVC(kernel="rbf", random_state=42))
    ])

    grid = {
        "pca__n_components": [20, 21],
        "kbest__k": [12],
        "svc__gamma": [0.099]
    }

    return GridSearchCV(pipe, grid, cv=10, scoring="balanced_accuracy",
                        refit=True, verbose=1, return_train_score=False)


def calcular_metricas(nombre: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics", "dataset": nombre,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }


def calcular_matriz_confusion(nombre: str, y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix", "dataset": nombre,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)}
    }


def guardar_modelo(objeto) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(MODEL_PATH, "wb") as f:
        pickle.dump(objeto, f)


def guardar_resultados_jsonl(resultados: list[dict]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in resultados)


def main():

    df_train = limpiar_datos(cargar_csv_desde_zip(TRAIN_ZIP_PATH, NOMBRE_TRAIN))
    df_test = limpiar_datos(cargar_csv_desde_zip(TEST_ZIP_PATH, NOMBRE_TEST))

    X_tr, y_tr = df_train.drop(TARGET_COL, axis=1), df_train[TARGET_COL]
    X_te, y_te = df_test.drop(TARGET_COL, axis=1), df_test[TARGET_COL]

    search = crear_modelo_y_busqueda()
    search.fit(X_tr, y_tr)
    guardar_modelo(search)

    y_tr_pred, y_te_pred = search.predict(X_tr), search.predict(X_te)
    resultados = [
        calcular_metricas("train", y_tr, y_tr_pred),
        calcular_metricas("test", y_te, y_te_pred),
        calcular_matriz_confusion("train", y_tr, y_tr_pred),
        calcular_matriz_confusion("test", y_te, y_te_pred)
    ]

    guardar_resultados_jsonl(resultados)

if __name__ == "__main__":
    main()
