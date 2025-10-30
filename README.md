# 🪄 Auto Pas Foto Generator (Background Merah + Resize 3x4)

Sistem ini membantu kamu mengedit hasil foto siswa dari photoshoot menjadi **ukuran pas foto 3x4** dengan **background merah otomatis**, serta **menyamakan tone warna** agar hasilnya lebih rapi dan seragam.  
Selain itu sistem ini juga mendukung **auto rename file menggunakan NIS (Nomor Induk Siswa)** dan **batch log** agar proses dokumentasi lebih efisien.

---

## ✨ Fitur Utama
- 🖼️ **Auto Remove Background** (pakai AI)  
- 🟥 Ganti background otomatis ke warna merah (kode warna bisa disesuaikan)  
- 🧑‍🎓 Resize ke ukuran pas foto standar 3x4 cm  
- 🧼 Koreksi warna otomatis (hilangkan kebiruan / kekuningan berlebihan)  
- 🪪 Auto rename file berdasarkan NIS siswa  
- 🧾 Generate batch log (riwayat proses file)

---

## 🛠️ Tech Stack
- Python 3.10+
- :contentReference[oaicite:1]{index=1} (background remover)
- :contentReference[oaicite:2]{index=2} (image processing)
- :contentReference[oaicite:3]{index=3} (array & pixel ops)
- :contentReference[oaicite:4]{index=4} (auto color correction — opsional)

---

## 📦 Cara Instalasi & Setup

### 1. Clone Repository
```bash
git clone https://github.com/username/auto-pasfoto-generator.git
cd auto-pasfoto-generator
```

### 2. Buat Virtual Environment
Virtual environment menjaga library project agar tidak bentrok dengan sistem utama.

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
rembg
Pillow
numpy
opencv-python
```

### 4. Jalankan Project
Pastikan semua foto input diletakkan di folder input/ dan NIS file CSV (jika ada) sesuai urutan.

```bash
python auto_pasfoto.py
```

## 🧭 Struktur Folder

```bash
📂 auto-pasfoto-generator/
├── 📁 input/             # Tempat foto asli
├── 📁 output/            # Hasil edit siap cetak
├── 📄 auto_pasfoto.py    # Script utama
├── 📄 requirements.txt   # Daftar library
├── 📄 README.md          # Dokumentasi project
└── 📄 log.csv            # Log hasil rename batch (auto generate)
```

## 🪄 Cara Penggunaan
- Masukkan semua foto ke dalam folder input/.
- Jika ingin auto rename, buat file CSV berisi NIS sesuai urutan foto.
- Jalankan perintah:
```bash
python auto_pasfoto.py
```
- Foto hasil edit akan muncul di folder output/ dengan background merah dan ukuran pas foto 3x4.
- Log proses akan tersimpan otomatis di log.csv.

## 🧠 Tips & Catatan
Gunakan foto resolusi tinggi agar hasil pas foto tajam.
Jika ada foto yang terlalu gelap/terang, sistem akan mencoba auto adjust.
Warna background bisa diganti di bagian kode background_color = (R, G, B).

## 📜 Lisensi
Project ini bersifat open-source. Bebas digunakan dan dimodifikasi untuk keperluan sekolah, studio foto, maupun project pribadi.
