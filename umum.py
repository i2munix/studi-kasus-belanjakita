"""Fungsi bersama untuk aplikasi studi kasus ulasan PT BelanjaKita Digital (data dummy)."""
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA = Path(__file__).parent / "data"
LABEL = ["negatif", "positif"]

KATA_UMUM = {"dan", "di", "yang", "ini", "itu", "tapi", "saya", "sudah", "jadi", "sekali", "banget", "ya",
             "kok", "gini", "dulu", "ada", "padahal", "terus", "sini", "dalam", "untuk", "mau", "saja", "cuma",
             "tidak", "malah", "lagi", "semua", "bikin", "kecewa", "parah", "jujur", "aduh", "sayang", "nyesel",
             "belanja", "tolong", "jangan", "semoga", "bintang", "satu", "hari", "nunggu", "respon", "cepat",
             "diperbaiki", "dibenahi", "uninstall", "komplain", "order", "pesanan", "kedua", "kemarin", "pakai",
             "sejak", "metode", "cod", "beli", "pesan", "jam", "malam", "barang", "sekaligus", "sepatu", "hadiah",
             "bisa", "dipakai", "ramah", "mudah", "sesuai", "foto", "datang", "tertulis", "membalas", "membantu"}


def samarkan(teks: str) -> str:
    """Menyamarkan data pribadi: nomor telepon dan nama setelah kata 'saya'."""
    teks = re.sub(r"(\+62|0)8\d{1,2}[- ]?\d{3,4}[- ]?\d{3,4}", "[NOMOR_HP]", teks)
    return re.sub(r"\b(saya|Saya) [A-Z][a-z]+", r"\1 [NAMA]", teks)


def bersihkan(teks: str) -> str:
    """Huruf kecil, buang data pribadi yang disamarkan, angka, tanda baca, dan emoji."""
    teks = samarkan(teks).lower()
    teks = re.sub(r"\[nomor_hp\]|\[nama\]", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    return re.sub(r"\s+", " ", teks).strip()


@st.cache_data
def data_mentah() -> pd.DataFrame:
    return pd.read_csv(DATA / "ulasan_belanjakita.csv", keep_default_na=False)


@st.cache_data
def data_bersih() -> pd.DataFrame:
    df = data_mentah()
    df = df[df["teks_ulasan"].str.strip() != ""].drop_duplicates().reset_index(drop=True)
    df["teks_bersih"] = df["teks_ulasan"].map(bersihkan)
    return df


@st.cache_data
def data_baru() -> pd.DataFrame:
    df = pd.read_csv(DATA / "ulasan_baru.csv", keep_default_na=False)
    df["teks_bersih"] = df["teks_ulasan"].map(bersihkan)
    return df


@st.cache_resource
def latih_model():
    """Pipeline sama dengan notebook: split 80:20 (random_state=42, stratify), TF-IDF bawaan, MultinomialNB."""
    df = data_bersih()
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        df["teks_bersih"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    tfidf = TfidfVectorizer()
    model = MultinomialNB().fit(tfidf.fit_transform(X_latih), y_latih)
    tebak = model.predict(tfidf.transform(X_uji))
    hasil = dict(n_latih=len(X_latih), n_uji=len(X_uji), fitur=len(tfidf.vocabulary_),
                 akurasi=accuracy_score(y_uji, tebak), cm=confusion_matrix(y_uji, tebak, labels=LABEL))
    return model, tfidf, hasil


def prediksi(teks: str) -> str:
    model, tfidf, _ = latih_model()
    return model.predict(tfidf.transform([bersihkan(teks)]))[0]


def top_kata_negatif(n=10) -> pd.DataFrame:
    df = data_bersih()
    neg = df.loc[df["label"] == "negatif", "teks_bersih"]
    c = Counter(k for t in neg for k in t.split() if k not in KATA_UMUM and len(k) > 2)
    return pd.DataFrame(c.most_common(n), columns=["kata", "jumlah"])


def tabel_cm(cm) -> pd.DataFrame:
    return pd.DataFrame(cm, index=["Asli negatif", "Asli positif"], columns=["Ditebak negatif", "Ditebak positif"])


def batang_label(vc: pd.DataFrame):
    import altair as alt
    warna = alt.Scale(domain=LABEL, range=["#C0392B", "#2E8B57"])
    dasar = alt.Chart(vc).encode(x=alt.X("label:N", title=None, sort=LABEL, axis=alt.Axis(labelAngle=0)),
                                 y=alt.Y("jumlah:Q", title="Jumlah ulasan"))
    return (dasar.mark_bar().encode(color=alt.Color("label:N", scale=warna, legend=None))
            + dasar.mark_text(dy=-8).encode(text="jumlah:Q")).properties(height=260)


def batang_h(df: pd.DataFrame, x: str, y: str, warna="#C0392B", judul_x="Jumlah", fmt=None, maks=None):
    import altair as alt
    sb = alt.Scale(domain=[0, maks]) if maks else alt.Undefined
    dasar = alt.Chart(df).encode(y=alt.Y(f"{y}:N", sort=None, title=None),
                                 x=alt.X(f"{x}:Q", title=judul_x, scale=sb))
    teks = dasar.mark_text(align="left", dx=4).encode(text=alt.Text(f"{x}:Q", format=fmt) if fmt else f"{x}:Q")
    return (dasar.mark_bar(color=warna) + teks).properties(height=36 * len(df) + 20)


def persen(x, d=1):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def kepala(judul: str, kp: str, asesor: str):
    st.set_page_config(page_title=judul, page_icon="📊", layout="wide")
    st.title(judul)
    st.caption(f"Jawaban Studi Kasus FR.IA.04A · {kp} · Skema Insinyur Pembelajaran Mesin (Machine Learning Engineer) · "
               f"Asesor: {asesor}")
    st.info("Data rekaan (dummy) untuk uji kompetensi. Nama dan nomor telepon fiktif, tetapi tetap diperlakukan "
            "sebagai Data Pribadi.", icon="ℹ️")


def ringkasan_data():
    raw = data_mentah()
    df = data_bersih()
    kosong = int((raw["teks_ulasan"].str.strip() == "").sum())
    dup = int(raw.duplicated().sum())
    a, b, c, d = st.columns(4)
    a.metric("Baris awal", f"{len(raw):,}".replace(",", "."))
    b.metric("Baris kosong", kosong)
    c.metric("Baris duplikat", dup)
    d.metric("Siap diolah", len(df))
    kiri, kanan = st.columns([2, 1])
    with kiri:
        st.write("5 baris pertama data mentah:")
        st.dataframe(raw.head(), use_container_width=True, hide_index=True)
    with kanan:
        vc = df["label"].value_counts().reindex(LABEL).rename_axis("label").reset_index(name="jumlah")
        st.altair_chart(batang_label(vc), use_container_width=True)
        st.caption(f"Negatif {vc.jumlah[0]} ({vc.jumlah[0] / len(df):.0%}) · Positif {vc.jumlah[1]} "
                   f"({vc.jumlah[1] / len(df):.0%})")
    return raw, df
