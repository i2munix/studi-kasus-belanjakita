"""KP1 – Penentuan dan Pembuatan Proyek Data Science (Asesor: Fika)."""
import altair as alt
import pandas as pd
import streamlit as st

from umum import data_bersih, kepala, ringkasan_data, samarkan

kepala("Tujuan Teknis dan Rencana Proyek Analisis Sentimen Ulasan",
       "Kelompok Pekerjaan 1: Penentuan dan Pembuatan Proyek Data Science", "Fika")

t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(["1. Masalah bisnis", "2. Data", "3. Tujuan teknis", "4. Kriteria",
                                          "5–6. Rencana & jadwal", "Risiko", "7. Data pribadi", "8. Laporan"])

with t1:
    st.subheader("Langkah 1. Memahami masalah bisnis")
    a, b = st.columns([1, 2])
    a.metric("Rating aplikasi di Play Store", "3,8", delta="-0,7 dari 4,5", delta_color="inverse")
    b.markdown("""**Masalah:** rating aplikasi PT BelanjaKita Digital turun dari 4,5 menjadi 3,8.

**Tujuan bisnis:**
1. Mengetahui apakah ulasan pelanggan lebih banyak positif atau negatif.
2. Mengetahui keluhan yang paling sering muncul, supaya perusahaan tahu bagian mana yang diperbaiki lebih dulu.""")

with t2:
    st.subheader("Langkah 2. Mengenal data")
    raw, df = ringkasan_data()
    st.caption(f"Rata-rata panjang ulasan: {df['teks_ulasan'].str.split().str.len().mean():.1f} kata".replace(".", ","))

with t3:
    st.subheader("Langkah 3. Menentukan tujuan teknis")
    st.table(pd.DataFrame({
        "Tujuan bisnis": ["Mengetahui ulasan lebih banyak positif atau negatif",
                          "Mengetahui keluhan yang paling sering muncul"],
        "Tujuan teknis": ["Membuat model klasifikasi teks 2 kelas (positif/negatif) dari kolom teks_ulasan",
                          "Menampilkan 10 kata terbanyak pada ulasan negatif dalam diagram batang"]}))

with t4:
    st.subheader("Langkah 4. Menentukan kriteria kesuksesan")
    st.table(pd.DataFrame([
        ["Ketepatan model", "Akurasi pada data uji", "≥ 75%", "accuracy_score pada 20% data uji"],
        ["Ketepatan model", "Confusion matrix", "Ditampilkan", "Melihat jenis kesalahan positif/negatif"],
        ["Hasil bisnis", "Kata keluhan terbanyak", "10 kata", "Diagram batang dari ulasan negatif"],
        ["Kinerja", "Waktu memproses 1.000 ulasan", "< 1 menit", "Diukur saat menjalankan model"],
        ["Data pribadi", "Nama & nomor HP disamarkan", "100% ulasan", "Pencarian pola nomor HP/nama"],
    ], columns=["Aspek", "Ukuran", "Target", "Cara mengukur"]))
    st.caption("Kriteria dibuat jelas dan bisa diukur, lalu disetujui manajer Customer Experience. "
               "Target 75% berarti 3 dari 4 tebakan benar: cukup dipercaya dan realistis untuk model sederhana.")

with t5:
    st.subheader("Langkah 5. Tahapan proyek")
    st.write("Memahami data → Membersihkan data → Membuat model → Menguji model → Membuat laporan. "
             "Jika akurasi < 75% pada tahap 4, kembali ke tahap 3 (iterasi).")
    st.subheader("Langkah 6. Jadwal dan sumber daya")
    jadwal = pd.DataFrame([
        [1, "Memahami data", 2, "Data Scientist", "Python, pandas, Google Colab"],
        [2, "Membersihkan data", 3, "Data Scientist", "Python, pandas, regex"],
        [3, "Membuat model", 3, "ML Engineer", "scikit-learn (TF-IDF, Naive Bayes)"],
        [4, "Menguji model", 2, "ML Engineer", "scikit-learn (akurasi, confusion matrix)"],
        [5, "Membuat laporan", 2, "Ketua proyek", "Microsoft Word / PowerPoint"],
    ], columns=["No", "Tahapan", "Lama (hari)", "Penanggung jawab", "Alat"])
    jadwal["Mulai"] = jadwal["Lama (hari)"].cumsum() - jadwal["Lama (hari)"]
    jadwal["Selesai"] = jadwal["Lama (hari)"].cumsum()
    st.dataframe(jadwal.drop(columns=["Mulai", "Selesai"]), use_container_width=True, hide_index=True)
    gantt = alt.Chart(jadwal).mark_bar(color="#2F6690", cornerRadius=3).encode(
        y=alt.Y("Tahapan:N", sort=None, title=None),
        x=alt.X("Mulai:Q", title="Hari kerja", scale=alt.Scale(domain=[0, 14])), x2="Selesai:Q",
        tooltip=["Tahapan", "Lama (hari)", "Penanggung jawab", "Alat"])
    cadangan = alt.Chart(pd.DataFrame({"a": [12], "b": [14]})).mark_rect(color="#FDF3DC").encode(x="a:Q", x2="b:Q")
    st.altair_chart((cadangan + gantt).properties(height=230), use_container_width=True)
    st.success("Total: 12 hari kerja + 2 hari cadangan (± 2,5 minggu)")

with t6:
    st.subheader("Risiko dan penanganannya")
    st.table(pd.DataFrame([
        ["Data dari tim CX terlambat", "Mulai dengan sampel data; pakai 2 hari cadangan"],
        ["Akurasi model < 75%", "Tambah data berlabel atau coba algoritma lain (misalnya Logistic Regression)"],
        ["Data pribadi ikut terbawa", "Samarkan nama & nomor HP sebelum data dibagikan ke tim"],
        ["Satu tahap terlambat 2 hari", "Pakai hari cadangan, beri tahu manajer CX, perbarui jadwal"],
    ], columns=["Risiko", "Penanganan"]))

with t7:
    st.subheader("Langkah 7. Aturan data pribadi")
    st.markdown("""1. Nama dan nomor telepon pelanggan **disamarkan** sebelum data dibagikan ke tim.
2. Data disimpan di folder **berkata sandi**; hanya anggota tim proyek yang boleh membuka.
3. Data tidak boleh disalin ke laptop/flashdisk pribadi.
4. Data dihapus setelah proyek selesai.
5. Dasar aturan: UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi.""")
    df = data_bersih()
    pribadi = df["teks_ulasan"].str.contains(r"08\d|saya [A-Z]", regex=True)
    st.metric("Ulasan yang memuat nama/nomor telepon", int(pribadi.sum()))
    contoh = df[pribadi].head(3)["teks_ulasan"]
    st.dataframe(pd.DataFrame({"Asli": contoh, "Disamarkan": contoh.map(samarkan)}), use_container_width=True,
                 hide_index=True)
    coba = st.text_input("Coba samarkan teks sendiri:", "saya Siti, paket telat, hubungi saya di 0813-2222-3333")
    st.write("Hasil:", samarkan(coba))

with t8:
    st.subheader("Langkah 8. Laporan singkat rencana proyek")
    df = data_bersih()
    st.markdown(f"""
**Masalah bisnis.** Rating aplikasi turun dari 4,5 menjadi 3,8.

**Data.** {len(df)} ulasan siap diolah ({(df.label == 'negatif').sum()} negatif, {(df.label == 'positif').sum()} positif).

**Tujuan teknis.** Model klasifikasi teks 2 kelas + 10 kata keluhan terbanyak.

**Kriteria kesuksesan.** Akurasi ≥ 75% pada data uji, confusion matrix, 10 kata keluhan, < 1 menit untuk 1.000 ulasan,
seluruh data pribadi disamarkan.

**Rencana.** 5 tahapan, 12 hari kerja + 2 hari cadangan.

**Kesimpulan.** Data cukup untuk model 2 kelas; rencana perlu disetujui manajer Customer Experience sebelum dimulai.
""")
