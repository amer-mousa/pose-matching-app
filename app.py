import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

mp_pose = mp.solutions.pose

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    lms = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lms.append([(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark])
    cap.release()
    return lms

st.title("🧠 تقييم المهارات الحركية")

ref_video = st.file_uploader("📥 حمّل فيديو المهارة المرجعي", type=["mp4"])
live_video = st.file_uploader("📷 حمّل فيديو الأداء للمقارنة", type=["mp4"])

if ref_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(ref_video.read())
        ref_path = tmp.name
    ref_lm = extract_landmarks(ref_path)
    st.success(f"✅ استخرجنا {len(ref_lm)} إطار من المرجع.")

if ref_video and live_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp2:
        tmp2.write(live_video.read())
        live_path = tmp2.name
    live_lm = extract_landmarks(live_path)
    st.success(f"✅ استخرجنا {len(live_lm)} إطار من الأداء.")

    # حساب التشابه
    n = min(len(ref_lm), len(live_lm))
    if n == 0:
        st.error("❌ لا توجد إطارات كافية للمقارنة.")
    else:
        sims = []
        for i in range(n):
            a = np.array(ref_lm[i]).flatten().reshape(1, -1)
            b = np.array(live_lm[i]).flatten().reshape(1, -1)
            sims.append(cosine_similarity(a, b)[0][0])
        score = np.mean(sims)*10
        st.success(f"🎯 درجة المطابقة: {score:.2f} من 10")
