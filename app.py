import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

st.set_page_config(page_title="تحليل المهارات", layout="centered")
st.title("🤖 مطابقة حركة الفيديو")

def extract_landmarks(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    results = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
            results.append(landmarks)
    cap.release()
    return results

def compare_landmarks(ref, live):
    n = min(len(ref), len(live))
    sims = []
    for i in range(n):
        a = np.array(ref[i]).flatten().reshape(1, -1)
        b = np.array(live[i]).flatten().reshape(1, -1)
        sim = cosine_similarity(a, b)[0][0]
        sims.append(sim)
    return np.mean(sims)

ref_video = st.file_uploader("📥 حمّل فيديو المهارة المرجعية", type=["mp4"])
live_video = st.file_uploader("📷 حمّل فيديو الأداء للمقارنة", type=["mp4"])

if ref_video and live_video:
    st.info("⏳ جاري المعالجة، الرجاء الانتظار...")

    # حفظ الفيديوهات مؤقتاً
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f1:
        f1.write(ref_video.read())
        ref_path = f1.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f2:
        f2.write(live_video.read())
        live_path = f2.name

    ref_lm = extract_landmarks(ref_path)
    live_lm = extract_landmarks(live_path)

    score = compare_landmarks(ref_lm, live_lm)
    st.success(f"✅ درجة التطابق: {score * 10:.2f} من 10")
