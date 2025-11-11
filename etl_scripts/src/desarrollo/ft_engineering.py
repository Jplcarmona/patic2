import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ruta_datos = r'C:\Users\Juan\Downloads\Pyhton_ETL - copia\etl_scripts\src\desarrollo\BD_creditos.xlsx'

def cargar_datos():
    try:
        df = pd.read_excel(ruta_datos)
        print(f"Datos cargados desde {ruta_datos}")
        return df
    except FileNotFoundError:
        print(f"No se encontró el archivo en {ruta_datos}")
        return None
    
def limpiar_tendencia(valor):
    if pd.isna(valor):
        return np.nan

    # Si ya está como categoría válida, mantenerlo
    if str(valor).strip().lower() in ["creciente", "decreciente", "estable"]:
        return str(valor).capitalize()

    try:
        num = float(valor)
        if num == 0:
            return "Estable"
        elif num > 0:
            return "Creciente"
        else:
            return "Decreciente"
    except (ValueError, TypeError):
        return np.nan # En caso de algo raro 
    
def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    
    if "capital_prestado" in df.columns and "salario_cliente" in df.columns:
        df["relacion_deuda_ingreso"] = df["capital_prestado"] / (df["salario_cliente"])
        
    df["tendencia_ingresos"] = df["tendencia_ingresos"].apply(limpiar_tendencia)
    
    df.loc[(df["edad_cliente"] < 18) | (df["edad_cliente"] > 74), "edad_cliente"] = np.nan
    df.loc[(df["puntaje_datacredito"] < 150) | (df["puntaje_datacredito"] > 950), "puntaje_datacredito"] = np.nan
    df.loc[df["plazo_meses"] <= 0, "plazo_meses"] = np.nan
    df.loc[df["huella_consulta"] < 0, "huella_consulta"] = np.nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    columnas_eliminar = ["fecha_prestamo", "salario_cliente", "cuota_pactada", "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal " , "puntaje", "saldo_principal", "saldo_mora", "saldo_mora_codeudor"]
    df = df.drop(columns=[c for c in columnas_eliminar if c in df.columns], axis = 1)
    
    if "tipo_credito" in df.columns:
        df = df[(df["tipo_credito"] != 68) & (df["tipo_credito"] != 7)]
        
    if "tipo_credito" in df.columns:
        df["tipo_credito"] = df["tipo_credito"].astype("category")
        
    if "tipo_laboral" in df.columns:
        df["tipo_laboral"] = df["tipo_laboral"].astype("category")
        
    if "tendencia_ingresos" in df.columns:
        df["tendencia_ingresos"] = df["tendencia_ingresos"].astype("category")
    
    if "Pago_atiempo" in df.columns:
        df["Pago_atiempo"] = df["Pago_atiempo"].astype("category")
        
    return df

def crear_pipelines():
    print("\n -- Iniciando Feature Engineering --")
    
    df = cargar_datos()
    if df is None:
        return
    
    df = preparar_dataset(df)
    
    X = df.drop("Pago_atiempo", axis = 1)
    y = df["Pago_atiempo"]
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y)
    
    nominal_features = ["tipo_credito", "tipo_laboral", ]
    numeric_features = ["capital_prestado", "plazo_meses", "edad_cliente", "total_otros_prestamos", "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta", "saldo_total", "promedio_ingresos_datacredito", "relacion_deuda_ingreso"]
    ordinal_features = ["tendencia_ingresos"]
    
    numeric_transformer = Pipeline(steps= [
        ("Imputer", SimpleImputer(strategy="median")),
        ("Scaler", StandardScaler())
    ])
    
    categorical_nominal = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="most_frequent")),
        ("Onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
    ])
    
    categorical_ordinal = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="most_frequent")),
        ("Ordinal", OrdinalEncoder(categories=[["Decreciente", "Estable", "Creciente"]]))
    ])
    
    
    preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat_nom", categorical_nominal, nominal_features),
            ("cat_ord", categorical_ordinal, ordinal_features)
        ])
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return preprocessor, X_train, X_test, y_train, y_test, X_train_processed, X_test_processed

if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test, X_train_processed, X_test_processed  = crear_pipelines()

    print("Shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    assert X_train.shape[0] == y_train.shape[0], "Error: X_train e y_train no coinciden"
    assert X_test.shape[0] == y_test.shape[0], "Error: X_test e y_test no coinciden"
    print(" Feature Engineering funcionando correctamente.")
    
    
    
    
    
    
    
    


