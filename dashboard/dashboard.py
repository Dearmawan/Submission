import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sns.set(style='dark')

# Fungsi untuk styling background
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #e0f7fa, #80deea);
            color: #1e3a8a;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(to bottom, #f8f9fa, #e2e8f0);
            border-radius: 10px;
            padding: 10px;
        }
        h1, h2, h3 {
            color: #1e3a8a;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            border-radius: 5px;
        }
        .stMetric {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Helper functions
def create_daily_rentals_df(df):
    daily_rentals_df = df.copy()
    daily_rentals_df.rename(columns={"cnt": "rental_count"}, inplace=True)
    return daily_rentals_df

def create_byweather_df(df):
    byweather_df = df.groupby("weathersit")["rental_count"].mean().reset_index()
    byweather_df.rename(columns={"rental_count": "avg_rental_count"}, inplace=True)
    return byweather_df

def create_byseason_df(df):
    byseason_df = df.groupby("season")["rental_count"].mean().reset_index()
    byseason_df.rename(columns={"rental_count": "avg_rental_count"}, inplace=True)
    return byseason_df

def create_byholiday_df(df):
    byholiday_df = df.groupby("holiday")["rental_count"].mean().reset_index()
    holiday_map = {0: 'Bukan Hari Libur', 1: 'Hari Libur'}
    byholiday_df['holiday'] = byholiday_df['holiday'].map(holiday_map)
    return byholiday_df

def create_byweekday_df(df):
    byweekday_df = df.groupby("weekday")["rental_count"].mean().reset_index()
    weekday_map = {0: 'Minggu', 1: 'Senin', 2: 'Selasa', 3: 'Rabu', 4: 'Kamis', 5: 'Jumat', 6: 'Sabtu'}
    byweekday_df['weekday'] = byweekday_df['weekday'].map(weekday_map)
    return byweekday_df

# Load data
day_df = pd.read_csv("dashboard\\day.csv")
day_df['dteday'] = pd.to_datetime(day_df['dteday'])

# Cek kolom yang ada
print("Kolom yang ada di day_df:", day_df.columns)

datetime_columns = ["dteday"]
day_df.sort_values(by="dteday", inplace=True)
day_df.reset_index(drop=True, inplace=True)

for column in datetime_columns:
    day_df[column] = pd.to_datetime(day_df[column])

min_date = day_df["dteday"].min()
max_date = day_df["dteday"].max()

set_background()

with st.sidebar:
    st.image("dashboard\\sepeda.png", width=200)
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = day_df[(day_df["dteday"] >= str(start_date)) & 
                 (day_df["dteday"] <= str(end_date))]
main_df = main_df.rename(columns={"cnt": "rental_count"})

# Menyiapkan dataframe
daily_rentals_df = create_daily_rentals_df(main_df)
byweather_df = create_byweather_df(main_df)
byseason_df = create_byseason_df(main_df)
byholiday_df = create_byholiday_df(main_df)
byweekday_df = create_byweekday_df(main_df)

# Kluster
features = ['rental_count', 'temp', 'hum', 'windspeed', 'season', 'weathersit']
cluster_df = main_df[features].copy()
bins = [0, 2000, 5000, cluster_df['rental_count'].max()]
labels = ['Rendah', 'Sedang', 'Tinggi']
cluster_df['cluster'] = pd.cut(cluster_df['rental_count'], bins=bins, labels=labels, include_lowest=True)

# Dashboard
st.title('🚲 Bike Sharing Dashboard')
st.markdown("**Analisis Penyewaan Sepeda Berdasarkan Data Harian**")

col1, col2 = st.columns(2)
with col1:
    total_rentals = daily_rentals_df['rental_count'].sum()
    st.metric("Total Penyewaan", value=f"{total_rentals:,}")
with col2:
    avg_rentals = daily_rentals_df['rental_count'].mean()
    st.metric("Rata-rata Penyewaan Harian", value=f"{avg_rentals:.2f}")

# Tren Penyewaan Harian
st.subheader("Tren Penyewaan Harian")
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(daily_rentals_df["dteday"], daily_rentals_df["rental_count"], marker='o', linewidth=2, color="#1e3a8a")
ax.set_title("Jumlah Penyewaan Harian", fontsize=20)
ax.set_xlabel("Tanggal", fontsize=15)
ax.set_ylabel("Jumlah Penyewaan", fontsize=15)
ax.tick_params(axis='both', labelsize=12)
st.pyplot(fig)

# Regresi Linier
st.subheader("Prediksi Penyewaan dengan Regresi Linier")
X = main_df[['temp', 'hum', 'windspeed', 'weathersit', 'holiday']]
y = main_df['rental_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, color='#1e3a8a', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Aktual', fontsize=15)
ax.set_ylabel('Prediksi', fontsize=15)
ax.set_title('Aktual vs Prediksi Penyewaan', fontsize=20)
st.pyplot(fig)
st.write(f"R² Score: {r2:.4f}")
st.write("Koefisien Model:")
for feature, coef in zip(['Suhu', 'Kelembapan', 'Kecepatan Angin', 'Kondisi Cuaca', 'Hari Libur'], model.coef_):
    st.write(f"{feature}: {coef:.2f}")

# Penyewaan Berdasarkan Hari Libur
st.subheader("Penyewaan Berdasarkan Status Hari Libur")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='holiday', y='rental_count', hue='holiday', data=byholiday_df, palette='muted', ax=ax, legend=False)
ax.set_title('Rata-rata Penyewaan: Hari Libur vs Bukan Hari Libur', fontsize=20)
ax.set_xlabel('Status Hari', fontsize=15)
ax.set_ylabel('Rata-rata Penyewaan', fontsize=15)
st.pyplot(fig)

# Penyewaan Berdasarkan Kondisi Cuaca
st.subheader("Penyewaan Berdasarkan Kondisi Cuaca")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='weathersit', y='avg_rental_count', hue='weathersit', data=byweather_df, palette='viridis', ax=ax, legend=False)
ax.set_title('Rata-rata Penyewaan Berdasarkan Cuaca', fontsize=20)
ax.set_xlabel('Kondisi Cuaca (1:Clear, 2:Misty, 3:Light Rain)', fontsize=15)
ax.set_ylabel('Rata-rata Penyewaan', fontsize=15)
st.pyplot(fig)

# Penyewaan Berdasarkan Musim dan Hari
st.subheader("Penyewaan Berdasarkan Musim dan Hari")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='season', y='avg_rental_count', hue='season', data=byseason_df, palette='coolwarm', ax=ax, legend=False)
ax.set_title('Rata-rata Penyewaan per Musim', fontsize=20)
ax.set_xlabel('Musim (1:Spring, 2:Summer, 3:Fall, 4:Winter)', fontsize=15)
ax.set_ylabel('Rata-rata Penyewaan', fontsize=15)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='weekday', y='rental_count', hue='weekday', data=byweekday_df, palette='coolwarm', ax=ax, legend=False)
ax.set_title('Rata-rata Penyewaan per Hari dalam Seminggu', fontsize=20)
ax.set_xlabel('Hari', fontsize=15)
ax.set_ylabel('Rata-rata Penyewaan', fontsize=15)
st.pyplot(fig)

# Analisis Kluster
st.subheader("Analisis Kluster Penyewaan")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='cluster', y='rental_count', hue='cluster', data=cluster_df, palette='Set2', ax=ax, legend=False)
ax.set_title('Distribusi Penyewaan per Kluster', fontsize=20)
ax.set_xlabel('Kluster', fontsize=15)
ax.set_ylabel('Jumlah Penyewaan', fontsize=15)
st.pyplot(fig)

# Korelasi Antar Fitur
st.subheader("Korelasi Antar Fitur")
correlation = main_df[['rental_count', 'temp', 'hum', 'windspeed', 'weathersit', 'holiday']].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Korelasi Antar Variabel', fontsize=20)
st.pyplot(fig)

st.caption('Copyright © Dearmawan 2025')