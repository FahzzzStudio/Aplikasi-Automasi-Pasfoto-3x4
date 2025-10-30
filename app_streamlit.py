import streamlit as st
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io

# Fungsi deteksi wajah menggunakan Haar Cascade
def detect_face(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return (x, y, w, h)

#Fungsi crop & center wajah
def crop_and_center_face(image_np):
    face = detect_face(image_np)
    if face is None:
        return None
    
    x, y, w, h = face
    cx, cy = x + w // 2, y + h // 2

    crop_size = max(w, h) * 3
    half_crop = crop_size // 2

    start_x = max(cx - half_crop, 0)
    start_y = max(cy - int(half_crop *  0.9), 0)
    end_x = min(cx + half_crop, image_np.shape[1])
    end_y = min(cy + int(half_crop * 1.1), image_np.shape[0])
    
    cropped_image = image_np[start_y:end_y, start_x:end_x]

    # Resize ke ukuran 3x4 cm / 354x472 px
    final_image = cv2.resize(cropped_image, (354, 472))
    return final_image

#  Fungsi ubah background menjadi merah
def change_background(image_np, color='red'):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.bitwise_not(mask)

    if color == 'red':
        bg_color = (0, 0, 255)  # Merah dalam BGR
    elif color == 'blue':
        bg_color = (255, 0, 0)  # Biru dalam BGR
    else:
        bg_color = (255, 255, 255)  # Putih dalam BGR

    bg = np.full(image_np.shape, bg_color, dtype=np.uint8)
    combined = np.where(mask[:, :, None].astype(bool), image_np, bg)
    return combined

# Streamlit UI
st.set_page_config(page_title="Aplikasi Automasi PasFoto 3x4 cm ", layout="centered")
st.title("🎓📸 Aplikasi Automasi PasFoto 3x4 cm Merah & Biru")
st.write("Unggah foto anda, dan aplikasi ini akan mengubahnya menjadi PasFoto 3x4 cm dengan latar belakang merah.")
# st.file_uploader("Unggah foto anda :", type=["jpg", "jpeg", "png"], key="upload")

uploaded_file = st.file_uploader("Unggah foto anda (format .jpg/.png):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption='Foto Asli', use_container_width=True)

    bg_choice = st.radio("Pilih warna latar belakang pasfoto:", ["Merah", "Biru"])
    color = "red" if bg_choice == "Merah" else "blue"

    with  st.spinner("Memproses foto..."):
        processed = crop_and_center_face(image_cv)
        if processed is None:
            st.error("❌ Tidak ada wajah terdeteksi dalam foto. Silakan coba dengan foto lain.")
        else:
            final_result = change_background(processed, color)
            final_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
            st.image(final_rgb, caption="Hasil PasFoto 3x4 cm", use_container_width=False)

            result_pil = Image.fromarray(final_rgb)
            result_pil.save("hasil_pasfoto_3x4cm.jpg")

            with open("hasil_pasfoto_3x4cm.jpg", "rb") as file:
                btn = st.download_button(
                    label="⬇️ Unduh PasFoto 3x4 cm",
                    data=file,
                    file_name="pasfoto_3x4cm.jpg",
                    mime="image/jpeg"
                )

st.markdown("---")
st.caption("© 2025 Fahzzz Studio | Powered by Streamlit & OpenCV")
