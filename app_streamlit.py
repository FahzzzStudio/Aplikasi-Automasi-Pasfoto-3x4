import streamlit as st
import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io

# Konfigurasi dasar
BACKGROUND_COLORS = {
    "Merah": (205, 18, 15),
    "Biru": (0, 102, 204)
}
WIDTH_PX = int(3 * 300 / 2.54)   # 354 px
HEIGHT_PX = int(4 * 300 / 2.54)  # 472 px
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def process_photo(image):
    # Convert ke OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, "❌ Tidak ada wajah terdeteksi."

    (x, y, w, h) = faces[0]
    cx, cy = x + w // 2, y + h // 2

    # Crop dengan rasio 3x4
    ratio = HEIGHT_PX / WIDTH_PX
    crop_w = int(w * 2.2)
    crop_h = int(crop_w * ratio)
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - int(crop_h * 0.45))
    x2 = min(img_cv.shape[1], x1 + crop_w)
    y2 = min(img_cv.shape[0], y1 + crop_h)
    crop_img = img_cv[y1:y2, x1:x2]

    # Hapus background dengan rembg
    _, encoded_img = cv2.imencode('.png', crop_img)
    img_no_bg = remove(encoded_img.tobytes())
    img_pil = Image.open(io.BytesIO(img_no_bg)).convert("RGBA")

    # Resize ke ukuran 3x4 cm
    img_resized = img_pil.resize((WIDTH_PX, HEIGHT_PX), Image.LANCZOS)
    return img_resized, None

st.set_page_config(page_title="Aplikasi PasFoto 3x4", layout="centered")
st.title("🎓📸 Aplikasi Automasi PasFoto 3x4 cm Merah & Biru")
st.write("Unggah foto kamu, pilih warna background, dan unduh hasilnya langsung.")

uploaded_file = st.file_uploader("Unggah foto anda (format .jpg/.png):", type=["jpg", "jpeg", "png"])
bg_choice = st.radio("Pilih warna latar belakang:", ["Merah", "Biru"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Foto Asli", use_container_width=True)

    with st.spinner("⏳ Memproses foto..."):
        processed_img, error = process_photo(image)

        if error:
            st.error(error)
        else:
            # Buat background sesuai pilihan
            bg_color = BACKGROUND_COLORS[bg_choice]
            bg = Image.new("RGBA", (WIDTH_PX, HEIGHT_PX), bg_color + (255,))
            final_img = Image.alpha_composite(bg, processed_img).convert("RGB")

            st.image(final_img, caption="✅ Hasil PasFoto 3x4 cm", use_container_width=False)

            # Tombol download
            img_bytes = io.BytesIO()
            final_img.save(img_bytes, format="JPEG", quality=100)
            st.download_button(
                label="⬇️ Unduh PasFoto 3x4 cm",
                data=img_bytes.getvalue(),
                file_name="pasfoto_3x4cm.jpg",
                mime="image/jpeg"
            )

st.caption("© 2025 Fahzzz Studio | Powered by Streamlit & OpenCV")
