
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

st.set_page_config(page_title="تحليل مطابقة المهارات", layout="centered")
st.title("🎯 تقييم مطابقة المهارة من الفيديو")

mp_pose = mp.solutions.pose

def extract_landmarks_from_video(video_path):
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]
            landmarks_list.append(landmarks)
    cap.release()
    return landmarks_list

def calculate_similarity(ref_landmarks, live_landmarks):
    min_len = min(len(ref_landmarks), len(live_landmarks))
    similarities = []
    for i in range(min_len):
        ref_vec = np.array(ref_landmarks[i]).flatten().reshape(1, -1)
        live_vec = np.array(live_landmarks[i]).flatten().reshape(1, -1)
        sim = cosine_similarity(ref_vec, live_vec)[0][0]
        similarities.append(sim)
    return np.mean(similarities)

reference_video = st.file_uploader("📥 حمّل فيديو المهارة المرجعي", type=["mp4", "mov"])

if reference_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ref_tmp:
        ref_tmp.write(reference_video.read())
        ref_path = ref_tmp.name

    st.video(ref_path)
    st.info("📈 يتم استخراج نقاط المفاصل من الفيديو المرجعي...")
    ref_landmarks = extract_landmarks_from_video(ref_path)
    st.success(f"✅ تم استخراج {len(ref_landmarks)} إطارًا بنجاح.")

    st.header("🎥 تصوير مباشر بالكاميرا")
    run_camera = st.button("🔴 ابدأ التسجيل من الكاميرا")

    if run_camera:
        st.warning("📸 يتم فتح الكاميرا الآن. اضغط على 'Q' من النافذة لإيقاف التسجيل.")
        live_path = os.path.join(tempfile.gettempdir(), "live_capture.mp4")
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(live_path, fourcc, 10.0, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            cv2.imshow('اضغط Q لإنهاء التسجيل', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        st.success("✅ تم التسجيل من الكاميرا.")
        st.video(live_path)
        st.info("⚙️ يتم تحليل الفيديو ومقارنته...")

        live_landmarks = extract_landmarks_from_video(live_path)
        score = calculate_similarity(ref_landmarks, live_landmarks)
        st.success(f"🎯 درجة التطابق: {score * 10:.2f} من 10")
