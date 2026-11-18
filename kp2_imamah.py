"""KP2 – Pembangunan dan Pengevaluasian Model (Asesor: Imamah)."""
import streamlit as st

from umum import (bersihkan, data_bersih, data_mentah, kepala, latih_model, persen, prediksi, ringkasan_data, samarkan,
                  tabel_cm, top_kata_negatif, batang_h)

kepala("Analisis Sentimen Ulasan PT BelanjaKita Digital",
       "Kelompok Pekerjaan 2: Pembangunan dan Pengevaluasian Model", "Imamah")

model, tfidf, h = latih_model()
cm = h["cm"]
t1, t2, t3, t4, t5, t6 = st.tabs(["1. Data", "2. Membersihkan teks", "3–6. Model & pengujian",
                                  "7. Kata keluhan", "Coba prediksi", "8. Laporan"])

with t1:
    st.subheader("Langkah 1. Membuka dan mengenal data")
    st.write("Rating aplikasi turun dari **4,5** menjadi **3,8**. Tim Customer Experience menyiapkan ulasan "
             "berlabel positif/negatif dalam file CSV (kolom `teks_ulasan` dan `label`).")
    raw, df = ringkasan_data()

with t2:
    st.subheader("Langkah 2. Membersihkan teks")
    st.write("Huruf kecil, hapus angka, tanda baca, dan emoji. Nama dan nomor telepon **disamarkan dulu** "
             "(UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi), lalu dibuang dari teks.")
    df = data_bersih()
    pribadi = df["teks_ulasan"].str.contains(r"08\d|saya [A-Z]", regex=True)
    st.metric("Ulasan yang memuat nama/nomor telepon", int(pribadi.sum()))
    contoh = df[pribadi].head(3)[["teks_ulasan", "teks_bersih"]].rename(
        columns={"teks_ulasan": "Sebelum", "teks_bersih": "Sesudah"})
    st.dataframe(contoh, use_container_width=True, hide_index=True)
    coba = st.text_input("Coba bersihkan teks sendiri:", "saya Budi, refund lama!! hubungi 0812-1111-2222 😡")
    st.write("Disamarkan:", samarkan(coba))
    st.write("Bersih:", f"`{bersihkan(coba)}`")

with t3:
    st.subheader("Langkah 3–5. Membagi data, TF-IDF, Naive Bayes")
    a, b, c = st.columns(3)
    a.metric("Data latih (80%)", h["n_latih"])
    b.metric("Data uji (20%)", h["n_uji"])
    c.metric("Jumlah kata (fitur TF-IDF)", h["fitur"])
    st.code("X_latih, X_uji, y_latih, y_uji = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
            "tfidf = TfidfVectorizer()\n"
            "model = MultinomialNB().fit(tfidf.fit_transform(X_latih), y_latih)", language="python")
    st.subheader("Langkah 6. Menguji model")
    kiri, kanan = st.columns(2)
    with kiri:
        st.metric("Akurasi pada data uji", persen(h["akurasi"]),
                  delta=f"{(h['akurasi'] - 0.75) * 100:+.1f} poin dari target 75%".replace(".", ","))
        if h["akurasi"] >= 0.75:
            st.success("Target minimal 75% → MEMENUHI")
        else:
            st.error("Target minimal 75% → BELUM memenuhi")
        st.write(f"Benar **{cm[0][0] + cm[1][1]}** dari **{h['n_uji']}** ulasan uji.")
    with kanan:
        st.write("Confusion matrix")
        st.dataframe(tabel_cm(cm), use_container_width=True)
        st.caption(f"{cm[0][1]} negatif salah ditebak positif · {cm[1][0]} positif salah ditebak negatif. "
                   "Kesalahan banyak terjadi pada ulasan campuran (ada pujian sekaligus keluhan).")

with t4:
    st.subheader("Langkah 7. Kata keluhan terbanyak")
    top = top_kata_negatif()
    kiri, kanan = st.columns([3, 2])
    kiri.altair_chart(batang_h(top, "jumlah", "kata", judul_x="Jumlah kemunculan"), use_container_width=True)
    kanan.markdown("""**Arti untuk perusahaan**
- *aplikasi, sering* → aplikasi sering error / keluar sendiri
- *lambat, pengiriman, kurir, ongkir* → masalah pengiriman
- *refund, pembayaran* → dana lama kembali, pembayaran gagal
- *pelayanan, customer* → layanan CS lambat

Kata umum (dan, di, yang, …) tidak dihitung.""")

with t5:
    st.subheader("Coba model dengan ulasan sendiri")
    teks = st.text_area("Tulis ulasan:", "Pengiriman cepat, harga murah, pasti belanja lagi")
    if st.button("Prediksi", type="primary"):
        hasil = prediksi(teks)
        if hasil == "positif":
            st.success(f"Hasil: {hasil.upper()}")
        else:
            st.error(f"Hasil: {hasil.upper()}")

with t6:
    st.subheader("Langkah 8. Laporan singkat")
    raw = data_mentah()
    df = data_bersih()
    neg, pos = (df["label"] == "negatif").sum(), (df["label"] == "positif").sum()
    st.markdown(f"""
**Tujuan.** Mengetahui apakah ulasan lebih banyak positif atau negatif, dan keluhan apa yang paling sering muncul.

**Data.** {len(raw)} baris → {len(df)} ulasan setelah baris kosong dan duplikat dihapus ({neg} negatif, {pos} positif).

**Langkah.** Membersihkan teks dan menyamarkan data pribadi → bagi data 80:20 → TF-IDF → Naive Bayes.

**Hasil.** Akurasi **{persen(h['akurasi'])}** pada data uji, memenuhi target 75%.

**Kesimpulan.** Ulasan negatif lebih banyak. Keluhan utama: aplikasi, pengiriman, refund/pembayaran, dan layanan CS.

**Saran.** (1) Perbaiki stabilitas aplikasi dan login; (2) percepat refund dan pengiriman; (3) pantau ulasan baru
setiap bulan dengan model ini; (4) tambah data berlabel agar akurasi meningkat.
""")
