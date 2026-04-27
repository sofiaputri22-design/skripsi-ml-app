import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Prediksi Kecelakaan", layout="wide")

st.title("🚗 Estimasi Tingkat Keparahan Kecelakaan")

# INPUT
st.sidebar.header("Input Data")

age = st.sidebar.slider("Umur", 0, 100, 25)
jenis_kendaraan = st.sidebar.selectbox("Jenis Kendaraan", ["Motor", "Mobil", "Truk"])
kepemilikan_sim = st.sidebar.selectbox("Kepemilikan SIM", ["Ada", "Tidak Ada"])
atribut = st.sidebar.selectbox("Atribut Keselamatan", ["Safety Belt", "Tidak"])
kecamatan = st.sidebar.selectbox("Kecamatan", ["Dukun", "Manyar", "Gresik"])
tipe_laka = st.sidebar.selectbox("Tipe Laka", ["Tabrak", "Tergelincir"])
direction = st.sidebar.selectbox("Arah", ["Utara", "Selatan"])
kecepatan = st.sidebar.slider("Kecepatan", 0, 120, 40)
tipe_jalan = st.sidebar.selectbox("Tipe Jalan", ["2/2", "4/2"])
fungsi_jalan = st.sidebar.selectbox("Fungsi Jalan", ["Arteri", "Kolektor"])

# buat dataframe input
input_dict = {
    "Age": age,
    "Kecepatan": kecepatan
}

input_df = pd.DataFrame([input_dict])

# sesuaikan kolom
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# prediksi
if st.button("Prediksi"):
    proba = model.predict_proba(input_df)[0]
    labels = model.classes_

    result = dict(zip(labels, proba))

    st.subheader("Hasil Prediksi")
    for k, v in result.items():
        st.write(f"{k}: {v*100:.2f}%")

    # pie chart
    chart_data = pd.DataFrame({
        "kelas": labels,
        "prob": proba
    })

    st.bar_chart(chart_data.set_index("kelas"))