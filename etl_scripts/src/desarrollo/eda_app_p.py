import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
#from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
#from prophet import Prophet


# Configuración general
st.set_page_config(page_title="EDA y Predicción - Restaurant Sales", layout="wide")
st.title("🍽️ Restaurant Sales Analytics & Prediction Dashboard")

# CARGA DE DATOS

@st.cache_data
def load_data():
    try:
        path_data = r'etl_scripts\src\desarrollo\restaurant_sales_data.csv'
        df = pd.read_csv(path_data)
        print(f"Datos cargados desde {path_data}")
        return df
    except FileNotFoundError:
        print(f"No se encontró el archivo en {path_data}")
        return None

# FUNCIONES DE TRANSFORMACIÓN 

def add_time_features(df):
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    df["day_of_week"] = df["date"].dt.isocalendar().day
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["sales"] = df["quantity_sold"] * df["actual_selling_price"]

    def get_season(date):
        year = date.year
        if (date.month == 12) or (date.month == 1 and date.day <= 6):
            return "Navidad"

        san_valentin = pd.Timestamp(year=year, month=2, day=14)
        if san_valentin - pd.Timedelta(days=7) <= date <= san_valentin + pd.Timedelta(days=7):
            return "San Valentín"

        c = calendar.Calendar(firstweekday=calendar.MONDAY)
        may_days = [d for d in c.itermonthdates(year, 5) if d.month == 5 and d.weekday() == 6]
        dia_madre = pd.Timestamp(may_days[1])
        if dia_madre - pd.Timedelta(days=7) <= date <= dia_madre + pd.Timedelta(days=7):
            return "Día de la Madre"

        june_days = [d for d in c.itermonthdates(year, 6) if d.month == 6 and d.weekday() == 6]
        dia_padre = pd.Timestamp(june_days[2])
        if dia_padre - pd.Timedelta(days=7) <= date <= dia_padre + pd.Timedelta(days=7):
            return "Día del Padre"

        sept_days = [d for d in c.itermonthdates(year, 9) if d.month == 9 and d.weekday() == 5]
        amor_amistad = pd.Timestamp(sept_days[2])
        if amor_amistad - pd.Timedelta(days=7) <= date <= amor_amistad + pd.Timedelta(days=7):
            return "Amor y Amistad"

        return "Normal"

    df["season"] = df["date"].apply(get_season)

    #df = df.drop(columns=['date'])

    return df

def eliminar_variables(df):
    cols_drop = ['key_ingredients_tags','restaurant_id', 'restaurant_type',
                 'observed_market_price', 'typical_ingredient_cost']
    return df.drop(columns=[c for c in cols_drop if c in df.columns], errors='ignore')

def replace_missing_values(df):
    missing_values = ["", " ", "NA", "N/A", "NULL", "None","Desconocido", "null", "none", "na", "n/a", "desconocido"]
    df.replace(missing_values, np.nan, inplace=True)
    return df

def transformaciones_ciclicas(df):
    for col in ['day_of_week', 'day_of_month', 'month']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

    return df.drop(columns=['day_of_week','day_of_month','month'], errors='ignore')


# Pestañas

tab1, tab2= st.tabs(["📈 Dashboard", "🤖 Predicción de Ventas Total"])

with tab1:

    df = load_data()
    
    if df.empty:
        st.stop()
        
    st.sidebar.success("✅ Datos cargados correctamente")
    st.sidebar.write(f"Registros totales: {df.shape[0]:,}")

    df = add_time_features(df)

    # FILTROS
    st.sidebar.header("🎚️ Filtros de Exploración")

    # --- Filtros de tiempo ---
    st.sidebar.subheader("🗓️ Tiempo")

    selected_month = st.sidebar.multiselect("Seleccionar mes:", sorted(df["month"].unique()), default=sorted(df["month"].unique()))
    selected_quarter = st.sidebar.multiselect("Trimestre:",sorted(df["quarter"].unique()), default=sorted(df["quarter"].unique()))

    # --- Filtros de condiciones ---
    st.sidebar.subheader("🌦️ Condiciones y Eventos")

    selected_weather = st.sidebar.multiselect("Condición del clima:", df["weather_condition"].unique(), default=df["weather_condition"].unique())
    selected_promo = st.sidebar.multiselect("Promocion:", df["has_promotion"].unique(), default=df["has_promotion"].unique())
    selected_event = st.sidebar.multiselect("Evento especial:", df["special_event"].unique(), default=df["special_event"].unique())
    selected_season = st.sidebar.multiselect("Temporada/Festividad:", df["season"].unique(), default=df["season"].unique())

    # --- Filtros de producto ---
    st.sidebar.subheader("🍽️ Menú")

    selected_item = st.sidebar.multiselect("Item del menú:", sorted(df["menu_item_name"].unique()), default=sorted(df["menu_item_name"].unique()))

    # --- Filtro de tipo de comida ---
    st.sidebar.subheader("🍱 Tipo de comida")

    selected_meal = st.sidebar.multiselect("Tipo de comida:", df["meal_type"].unique(), default=df["meal_type"].unique())

    # --- Aplicar filtros ---
    df_filtered = df.copy()

    if "month" in df.columns and selected_month:
        df_filtered = df_filtered[df_filtered["month"].isin(selected_month)]
        
    if "quarter" in df.columns and selected_quarter:
        df_filtered = df_filtered[df_filtered["quarter"].isin(selected_quarter)]

    if "weather_condition" in df.columns and selected_weather:
        df_filtered = df_filtered[df_filtered["weather_condition"].isin(selected_weather)]

    if "has_promotion" in df.columns and selected_promo:
        df_filtered = df_filtered[df_filtered["has_promotion"].isin(selected_promo)]

    if "special_event" in df.columns and selected_event:
        df_filtered = df_filtered[df_filtered["special_event"].isin(selected_event)]

    if "season" in df.columns and selected_season:
        df_filtered = df_filtered[df_filtered["season"].isin(selected_season)]

    if "meal_type" in df.columns and selected_meal:
        df_filtered = df_filtered[df_filtered["meal_type"].isin(selected_meal)]

    if "menu_item_name" in df.columns and selected_item:
        df_filtered = df_filtered[df_filtered["menu_item_name"].isin(selected_item)]

    if df_filtered.empty:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")
        st.stop()

    # GRÁFICOS
    day_mapping = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturday', 7:'Sunday'}

    # --- ventas por Día y Clima ---
    if "weather_condition" in df_filtered and not df_filtered["weather_condition"].isna().all():
        st.subheader("Ventas por Día de la Semana y Clima")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=df_filtered,
            x=df_filtered["day_of_week"].map(day_mapping),
            y='sales',
            hue='weather_condition',
            estimator="sum",
            errorbar=None,
            palette='Set2',
            ax=ax
        )
        ax.set_title("Ventas por Día y Clima", fontsize=14)
        st.pyplot(fig)

    # --- Ventas por tipo de comida ---
    if not df_filtered["meal_type"].isna().all():
        st.subheader("Ventas Totales por Tipo de Comida")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='meal_type', y='sales', data=df_filtered, estimator="sum", palette='viridis', errorbar=None, ax=ax)
        st.pyplot(fig)

    # --- Ventas por promociones ---
    if not df_filtered["has_promotion"].isna().all():
        st.subheader("Ventas Totales por Promociones")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='has_promotion', y='sales', data=df_filtered, estimator="sum", palette='viridis', errorbar=None, ax=ax)
        st.pyplot(fig)
        
    # --- Ventas por eventos especiales ---
    if not df_filtered["special_event"].isna().all():
        st.subheader("Ventas Totales por eventos especiales")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='special_event', y='sales', data=df_filtered, estimator="sum", palette='viridis', errorbar=None, ax=ax)
        st.pyplot(fig)

    # --- Ventas por temporada ---
    if not df_filtered["season"].isna().all():
        st.subheader("Ventas por Temporada / Evento Especial")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='season', y='sales', data=df_filtered, estimator="sum", palette='viridis', errorbar=None, ax=ax)
        st.pyplot(fig)

    # --- Ventas por item del menu ---
    if not df_filtered["menu_item_name"].isna().all():
        st.subheader("🍽️ Ventas Totales por Ítem del Menú")
        fig, ax = plt.subplots(figsize=(25,10))
        sns.barplot(x='menu_item_name', y='sales', data=df_filtered, estimator="sum", palette='magma', errorbar=None, ax=ax)
        st.pyplot(fig)

    # --- Ventas a lo largo del tiempo ---
    if not df_filtered["date"].isna().all():
        st.subheader("Tendencia de Ventas a lo Largo del Tiempo")
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(x='date', y='sales', data=df_filtered, color='skyblue', linewidth=2, marker='o', ax=ax)
        ax.set_title("Tendencia de Ventas y Regresion Lineal", fontsize=14)
        st.pyplot(fig)

    # RESUMEN
    st.subheader("📌 Resumen de Ventas")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ventas Totales", f"${df_filtered['sales'].sum():,.0f}")
    col2.metric("Pedidos Totales", f"{df_filtered['quantity_sold'].sum():,.0f}")
    col3.metric("Precio Promedio", f"${df_filtered['actual_selling_price'].mean():,.2f}")
    
with tab2:
    st.header("🔮 Predicción de Ventas Totales del Restaurante")

    df = load_data()
    df = add_time_features(df)
    df = transformaciones_ciclicas(df)
    df = eliminar_variables(df)
    

    cyclic_cols = ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
               'day_of_month_sin', 'day_of_month_cos']

    # Agregar ventas totales por día
    df_total = df.groupby("date")[["sales"]+ cyclic_cols].sum().reset_index()

    # Fecha futura a pronosticar
    future_date = st.date_input("Selecciona una fecha futura para pronosticar", value=pd.to_datetime("2025-12-24"))

    # Aseguramos que la serie esté ordenada 
    df_total = df_total.sort_values("date")

    # Dividimos en train hasta la última fecha conocida
    last_date = df_total["date"].max()
    forecast_days = (pd.Timestamp(future_date) - last_date).days

    # =========================
    # 1️⃣ Entrenamiento del modelo SARIMAX
    # =========================
    st.write("🧩 Entrenando modelo SARIMAX...")

    exog_vars = df_total[['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos']]

    # Ajuste automático básico del modelo (parámetros p,d,q y estacionales)
    model = SARIMAX(df_total["sales"], exog= exog_vars, order=(0, 0, 2), seasonal_order=(1, 1, 2, 7), enforce_stationarity=False, enforce_invertibility=False # ciclo semanal (7 días)
    )

    results = model.fit(disp=False)

    # Pronóstico futuro
    forecast_index = pd.date_range(start=last_date, periods=forecast_days + 1, freq='D')[1:]
    
    future_df = pd.DataFrame({"date": forecast_index})
    future_df["day_of_week"] = future_df["date"].dt.dayofweek
    future_df["day_of_month"] = future_df["date"].dt.day
    future_df["month"] = future_df["date"].dt.month
    future_df = transformaciones_ciclicas(future_df)
    
    exog_future = future_df[['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                         'day_of_month_sin', 'day_of_month_cos']]
    
    
    forecast = results.get_forecast(steps=forecast_days, exog=exog_future)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Combinar resultados históricos y pronóstico
    forecast_df = pd.DataFrame({
        "date": forecast_index,
        "forecast": forecast_mean
    })

    df_plot = pd.concat([
        df_total[["date", "sales"]].rename(columns={"sales": "value"}),
        forecast_df.rename(columns={"forecast": "value"})
    ])

    # Gráfico
    st.subheader(f"📊 Pronóstico Total de Ventas hasta {future_date.strftime('%d %B %Y')}")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_total["date"], df_total["sales"], label="Histórico", color="skyblue")
    ax.plot(forecast_index, forecast_mean, label="Pronóstico", color="orange", linestyle="--")
    ax.fill_between(forecast_index, forecast_ci["lower sales"], forecast_ci["upper sales"], color="orange", alpha=0.2)
    ax.set_title(f"Pronóstico Total de Ventas hasta {future_date.strftime('%d %B %Y')}", fontsize=14)
    ax.legend()
    st.pyplot(fig)

    # Métricas resumidas
    total_forecasted = forecast_mean.sum()
    st.metric("📈 Ventas Totales Proyectadas", f"${total_forecasted:,.0f}")
    
    # Simulador de múltiples productos según la proyección
    
    st.subheader(f"🏆 Simulador de Ventas para {future_date.strftime('%d %B %Y')}")
    
    # Promedio histórico de participación por producto
    product_share = (
        df.groupby("menu_item_name")["sales"].sum() / df["sales"].sum()
    )

    # Predicción total para la fecha seleccionada
    forecast_value = forecast_df[forecast_df["date"] == pd.Timestamp(future_date)]["forecast"].values
    if len(forecast_value) == 0:
        st.warning("⚠️ La fecha seleccionada está fuera del rango proyectado.")
    else:
        total_pred = forecast_value[0]
        product_list = sorted(product_share.index.tolist())

        # Inicializar sesión de productos
        if "productos" not in st.session_state:
            st.session_state["productos"] = [{"producto": product_list[0], "precio": 1.0}]

        # Botón para agregar producto
        if st.button("➕ Agregar producto"):
            st.session_state["productos"].append({"producto": product_list[0], "precio": 1.0})

        # Mostrar lista de productos con precio
        st.markdown("### 🧾 Productos seleccionados:")
        for i, p in enumerate(st.session_state["productos"]):
            cols = st.columns([3, 2, 1])
            st.session_state["productos"][i]["producto"] = cols[0].selectbox(
                f"Producto {i+1}",
                product_list,
                index=product_list.index(p["producto"]),
                key=f"producto_{i}"
            )
            st.session_state["productos"][i]["precio"] = cols[1].number_input(
                f"💲 Precio {i+1}",
                min_value=0.1,
                value=float(p["precio"]),
                step=500.0,
                key=f"precio_{i}"
            )

            # Botón eliminar producto individual
            if cols[2].button("❌", key=f"eliminar_{i}"):
                st.session_state["productos"].pop(i)
                st.experimental_rerun()

        st.markdown("---")

        # Botón para predecir
        if st.button("🔮 Predecir Ventas"):
            resultados = []
            total_ventas = 0
            total_unidades = 0

            for p in st.session_state["productos"]:
                producto = p["producto"]
                precio = p["precio"]
                participacion = product_share[producto]
                
                ventas_proyectadas = total_pred * participacion
                unidades_proyectadas = int(ventas_proyectadas / precio)

                resultados.append({
                    "Producto": producto,
                    "Precio_Ingresado": precio,
                    "Participación": f"{participacion:.2%}",
                    "Ventas_Proyectadas": ventas_proyectadas,
                    "Unidades_Proyectadas": unidades_proyectadas
                })

                total_ventas += ventas_proyectadas
                total_unidades += unidades_proyectadas

            # Mostrar resultados
            resultados_df = pd.DataFrame(resultados)
            st.markdown("### 📊 Resultados de la simulación")
            st.dataframe(
                resultados_df.style.format({
                    "Precio_Ingresado": "${:,.0f}",
                    "Ventas_Proyectadas": "${:,.0f}"
                })
            )

            st.success(
                f"📅 Fecha simulada: {future_date.strftime('%d %B %Y')}\n\n"
                f"💰 **Total de ventas proyectadas:** ${total_ventas:,.0f}\n"
                f"📦 **Total de unidades estimadas:** {total_unidades:,}"
            )
            
        #Cambiar el tipo de seleccion de fecha poner desde donde hasta donde quiere la prediccion
        