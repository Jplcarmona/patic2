import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import numpy as np


# Configuración general
st.set_page_config(page_title="EDA - Restaurant Sales", layout="wide")
st.title("📊 Exploratory Data Analysis - Restaurant Sales Report")

# CARGA DE DATOS
path_data = r'etl_scripts\src\desarrollo\restaurant_sales_data.csv'
 
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(path_data)
        print(f"Datos cargados desde {path_data}")
        return df
    except FileNotFoundError:
        print(f"No se encontró el archivo en {path_data}")
        return None

df = load_data()
if df.empty:
    st.stop()
    
st.sidebar.success("✅ Datos cargados correctamente")
st.sidebar.write(f"Registros totales: {df.shape[0]:,}")

# PREPROCESAMIENTO
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
df["day_of_week"] = df["date"].dt.isocalendar().day
df["day_of_month"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["sales"] = df["actual_selling_price"] * df["quantity_sold"]

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

# =============================
# 📊 RESUMEN
# =============================
st.subheader("📌 Resumen de Ventas")
col1, col2, col3 = st.columns(3)
col1.metric("Ventas Totales", f"${df_filtered['sales'].sum():,.0f}")
col2.metric("Pedidos Totales", f"{df_filtered['quantity_sold'].sum():,.0f}")
col3.metric("Precio Promedio", f"${df_filtered['actual_selling_price'].mean():,.2f}")