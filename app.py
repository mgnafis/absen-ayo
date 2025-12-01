import streamlit as st
import face_recognition
import cv2
import numpy as np
import pickle
import av
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- KONFIGURASI FILE DATABASE ---
DB_FILE = 'faces_db.pkl'

# --- FUNGSI UTILITAS ---

def load_database():
    """Memuat database wajah dari file pickle."""
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, 'rb') as f:
        return pickle.load(f)

def save_database(db):
    """Menyimpan database wajah ke file pickle."""
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)

# --- CLASS PROSESOR VIDEO ---
# Ini digunakan oleh streamlit-webrtc untuk memproses frame video secara realtime
class FaceRecognitionProcessor(VideoTransformerBase):
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load database saat inisialisasi
        db = load_database()
        if db:
            self.known_face_names = list(db.keys())
            self.known_face_encodings = list(db.values())

    def transform(self, frame):
        # Konversi frame dari format AV ke format OpenCV (numpy array)
        img = frame.to_ndarray(format="bgr24")

        # Perkecil ukuran frame agar proses deteksi lebih cepat (opsional 0.25x)
        # small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        # rgb_small_frame = small_frame[:, :, ::-1] # BGR ke RGB
        
        # Menggunakan frame asli (jika komputer kuat) atau resize
        rgb_frame = img[:, :, ::-1]

        # 1. Deteksi lokasi wajah
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # 2. Encode wajah
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 3. Loop setiap wajah yang ditemukan
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Gunakan jarak terkecil untuk akurasi terbaik
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            # Gambar kotak dan nama di wajah
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return img

# --- APLIKASI UTAMA STREAMLIT ---

def main():
    st.set_page_config(page_title="Sistem Face Recognition", layout="wide")
    st.title("Sistem Face Recognition Real-time")

    menu = ["Registrasi Wajah", "Deteksi Realtime"]
    choice = st.sidebar.selectbox("Menu", menu)

    # --- MENU 1: REGISTRASI ---
    if choice == "Registrasi Wajah":
        st.subheader("Pendaftaran Wajah Baru")
        
        name_input = st.text_input("Masukkan Nama Orang:")
        uploaded_file = st.file_uploader("Upload Foto Wajah (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

        if st.button("Simpan Data Wajah"):
            if name_input and uploaded_file is not None:
                # Baca file gambar
                image = face_recognition.load_image_file(uploaded_file)
                
                # Cari encoding wajah
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    face_encoding = encodings[0] # Ambil wajah pertama yang terdeteksi
                    
                    # Load DB lama, update, dan simpan
                    db = load_database()
                    db[name_input] = face_encoding
                    save_database(db)
                    
                    st.success(f"Berhasil! Wajah '{name_input}' telah terdaftar.")
                    st.info("Silakan pindah ke menu Deteksi Realtime untuk mencoba.")
                else:
                    st.error("Wajah tidak terdeteksi dalam foto. Harap gunakan foto yang jelas.")
            else:
                st.warning("Harap masukkan nama dan upload foto.")

    # --- MENU 2: DETEKSI REALTIME ---
    elif choice == "Deteksi Realtime":
        st.subheader("Kamera Deteksi")
        
        # Cek apakah database kosong
        db = load_database()
        if not db:
            st.warning("Database wajah kosong. Silakan daftarkan wajah terlebih dahulu di menu Registrasi.")
        else:
            st.write("Mode aktif: Menunggu webcam...")
            
            # Menjalankan WebRTC Streamer
            # rtc_configuration diperlukan jika dideploy ke cloud (menggunakan STUN server Google)
            webrtc_streamer(
                key="face-recognition",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                video_processor_factory=FaceRecognitionProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

if __name__ == "__main__":
    main()