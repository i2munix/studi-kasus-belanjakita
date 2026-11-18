"""KP3 – Pelaksanaan Deployment dan Pemeliharaan Model (Asesor: Firdaus)."""
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from umum import LABEL, batang_h, data_baru, data_bersih, kepala, latih_model, persen, prediksi, samarkan, tabel_cm

kepala("Cek Sentimen Ulasan – PT BelanjaKita Digital",
       "Kelompok Pekerjaan 3: Pelaksanaan Deployment dan Pemeliharaan Model", "Firdaus")

model, tfidf, h = latih_model()
t1, t2, t3, t4, t5, t6 = st.tabs(["Aplikasi (deployment)", "5. Uji aplikasi", "6. Kualitas model",
                                  "7. Tindak lanjut", "4. Data pribadi", "8. Laporan"])

with t1:
    st.subheader("Cek sentimen ulasan")
    st.write("Tim Customer Experience cukup menempelkan teks ulasan, lalu klik **Prediksi**.")
    teks = st.text_area("Tulis atau tempel teks ulasan:", "saya Rina, refund lama sekali, hubungi saya di 0812-3456-7890")
    if st.button("Prediksi", type="primary"):
        if not teks.strip():
            st.warning("Teks ulasan masih kosong.")
        else:
            hasil = prediksi(teks)
            if hasil == "positif":
                st.success(f"Hasil: {hasil.upper()}")
            else:
                st.error(f"Hasil: {hasil.upper()}")
            st.caption("Teks yang diproses (data pribadi disamarkan): " + samarkan(teks))
            print(f"log: hasil={hasil} panjang={len(teks)}")  # log tanpa teks lengkap
    with st.expander("Langkah 1–3: bagaimana aplikasi ini dibuat"):
        st.markdown("""1. Model Naive Bayes + TF-IDF dilatih dari `ulasan_belanjakita.csv` (80% data latih).
2. Library dan versinya dicatat di `requirements.txt` (streamlit, scikit-learn, pandas).
3. Aplikasi dibuat dengan Streamlit dan di-*deploy* ke Streamlit Community Cloud, sehingga cukup dibuka lewat link
   (tidak perlu `localhost`).""")
        st.code('teks = st.text_area("Tulis atau tempel teks ulasan:")\n'
                'if st.button("Prediksi"):\n'
                '    hasil = model.predict(tfidf.transform([bersihkan(teks)]))[0]', language="python")

with t2:
    st.subheader("Langkah 5. Uji aplikasi dengan 5 contoh ulasan")
    contoh = ["Barang sampai rusak dan cs lambat membalas", "Pengiriman cepat, kemasan aman dan rapi",
              "saya Rina, refund lama sekali, hubungi saya di 0812-3456-7890", "Harga murah, banyak promo menarik 😍",
              "Tidak bisa login, aplikasi sering error!!"]
    st.dataframe(pd.DataFrame({"Ulasan (disamarkan)": [samarkan(t) for t in contoh],
                               "Hasil": [prediksi(t) for t in contoh]}), use_container_width=True, hide_index=True)

baru = data_baru()
tebak = model.predict(tfidf.transform(baru["teks_bersih"]))
acc_baru = accuracy_score(baru["label"], tebak)

with t3:
    st.subheader("Langkah 6. Memeriksa kualitas model dengan 100 ulasan baru")
    a, b = st.columns(2)
    with a:
        st.metric("Akurasi pada ulasan baru", persen(acc_baru, 0),
                  delta=f"{(acc_baru - 0.75) * 100:+.0f} poin dari target 75%")
        if acc_baru >= 0.75:
            st.success("Memenuhi target 75%")
        else:
            st.error("Di bawah target 75% → perlu tindak lanjut")
        st.write("Confusion matrix (100 ulasan baru)")
        st.dataframe(tabel_cm(confusion_matrix(baru["label"], tebak, labels=LABEL)), use_container_width=True)
    with b:
        banding = pd.DataFrame({"data": ["Data uji awal", "Ulasan baru"],
                                "akurasi": [round(h["akurasi"] * 100, 1), round(acc_baru * 100, 1)]})
        st.altair_chart(batang_h(banding, "akurasi", "data", warna="#2F6690", judul_x="Akurasi (%)", maks=100),
                        use_container_width=True)
        st.caption("Penyebab turun: ulasan setelah update aplikasi memakai kata yang belum dikenal model, dan "
                   "sebagian label manusia keliru.")
    st.write("Contoh ulasan baru yang salah ditebak:")
    st.dataframe(baru[baru["label"] != tebak][["teks_ulasan", "label"]].head(8), use_container_width=True,
                 hide_index=True)

with t4:
    st.subheader("Langkah 7. Tingkat masalah dan tindak lanjut")
    st.table(pd.DataFrame([
        ["Ringan", "Aplikasi sesekali lambat, hasil prediksi masih benar", "Catat dan pantau"],
        ["Sedang", "Akurasi pada data baru di bawah/dekat 75%", "Latih ulang model dengan data baru"],
        ["Berat", "Aplikasi mati, banyak prediksi salah, atau data pribadi bocor", "Hentikan, perbaiki, laporkan"],
    ], columns=["Tingkat", "Contoh masalah", "Tindak lanjut"]))
    st.write(f"Kasus ini: akurasi {persen(acc_baru, 0)} → **SEDANG** → latih ulang dengan data lama + 50 ulasan baru, "
             "uji pada 50 ulasan baru lainnya.")
    lama = data_bersih()
    b_latih, b_uji = train_test_split(baru, test_size=0.5, random_state=42, stratify=baru["label"])
    gab = pd.concat([lama[["teks_bersih", "label"]], b_latih[["teks_bersih", "label"]]])
    tf2 = TfidfVectorizer(); m2 = MultinomialNB().fit(tf2.fit_transform(gab["teks_bersih"]), gab["label"])
    a1 = accuracy_score(b_uji["label"], model.predict(tfidf.transform(b_uji["teks_bersih"])))
    a2 = accuracy_score(b_uji["label"], m2.predict(tf2.transform(b_uji["teks_bersih"])))
    c1, c2 = st.columns(2)
    c1.metric("Model lama (50 ulasan uji)", persen(a1, 0))
    c2.metric("Model latih ulang (50 ulasan uji)", persen(a2, 0), delta=f"{(a2 - a1) * 100:+.0f} poin")

with t5:
    st.subheader("Langkah 4. Melindungi data pribadi di aplikasi")
    st.markdown("""- Nama → `[NAMA]` dan nomor HP → `[NOMOR_HP]` **sebelum** teks diproses.
- Log hanya mencatat **hasil** dan **panjang teks**, bukan teks ulasan lengkap.
- Aplikasi hanya untuk tim Customer Experience; log hanya dibuka petugas berwenang dan dihapus berkala.
- Bila data bocor: hentikan akses, lapor petugas PDP, beri tahu pelanggan paling lambat 3 × 24 jam.""")
    coba = st.text_input("Coba penyamaran:", "saya Andi, paket rusak, hubungi saya di 0857-1234-5678")
    st.write("Hasil:", samarkan(coba))

with t6:
    st.subheader("Langkah 8. Laporan singkat deployment dan pemeliharaan")
    st.markdown(f"""
**Tujuan.** Model sentimen dapat dipakai tim Customer Experience lewat aplikasi web dan diperiksa berkala.

**Deployment.** Model + TF-IDF → `requirements.txt` → aplikasi Streamlit → di-*deploy* ke Streamlit Community Cloud.

**Uji aplikasi.** 5 contoh ulasan memberi hasil yang benar dan sama dengan notebook.

**Kualitas.** Akurasi pada 100 ulasan baru **{persen(acc_baru, 0)}** (di bawah 75%) → masalah **sedang**.

**Tindak lanjut.** Latih ulang: akurasi pada 50 ulasan uji naik dari {persen(a1, 0)} menjadi **{persen(a2, 0)}**.

**Saran.** Periksa akurasi setiap bulan, latih ulang bila < 75% (terutama setelah update aplikasi), simpan versi model,
dan pakai laporan ini untuk deployment di kanal lain (App Store).
""")
