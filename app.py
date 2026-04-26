import streamlit as st
from PIL import Image
import numpy as np
import cv2
import datetime

from predict import predict

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b,#111827);
}

/* Header */
.main-header{
    text-align:center;
    padding:18px;
    border-radius:18px;
    background: linear-gradient(90deg,#1e3a8a,#2563eb,#0ea5e9);
    color:white;
    margin-bottom:20px;
    box-shadow:0 8px 20px rgba(0,0,0,0.18);
}

.main-header h1{
    margin:0;
    font-size:38px;
    font-weight:800;
}

.main-header p{
    margin-top:8px;
    font-size:15px;
    opacity:0.95;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:#111827;
}

section[data-testid="stSidebar"] * {
    color:white;
}

/* Metric cards */
.metric-card{
    background: rgba(255,255,255,0.08);
    padding:22px;
    border-radius:18px;
    text-align:center;
    color:white;
    box-shadow:0 8px 24px rgba(0,0,0,0.20);
}

/* Report box */
.report-box{
    background: rgba(255,255,255,0.08);
    padding:24px;
    border-radius:18px;
    color:white;
    box-shadow:0 8px 24px rgba(0,0,0,0.20);
}

/* Buttons */
.stDownloadButton button{
    background:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    padding:10px 18px;
    font-weight:600;
}

.stDownloadButton button:hover{
    background:#1d4ed8;
}

/* Images */
img{
    border-radius:16px;
}

h2,h3{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🧠 Brain Tumor Detection Dashboard</h1>
    <p>AI Powered MRI Analysis and Patient Reporting System</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("👤 Patient Details")

patient_name = st.sidebar.text_input("Patient Name")
patient_id = st.sidebar.text_input("Patient ID")
patient_age = st.sidebar.text_input("Age")

uploaded_file = st.sidebar.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if uploaded_file:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Prediction
    mask, confidence, tumor_area, overlay = predict(image_np)

    diagnosis = "Tumor Detected" if tumor_area > 1 else "No Tumor Detected"

    # Convert images
    mask_pil = Image.fromarray(mask)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🩺 Diagnosis</h4>
            <h2>{diagnosis}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Confidence</h4>
            <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📍 Tumor Area</h4>
            <h2>{tumor_area:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##")

    # -------------------------------------------------
    # IMAGE OUTPUT
    # -------------------------------------------------
    st.subheader("🖼️ MRI Scan Output")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image,
            caption="Original MRI",
            use_container_width=True
        )

    with col2:
        st.image(
            mask_pil,
            caption="Predicted Mask",
            use_container_width=True
        )

    with col3:
        st.image(
            overlay_pil,
            caption="AI Overlay",
            use_container_width=True
        )

    st.markdown("##")

    # -------------------------------------------------
    # REPORT
    # -------------------------------------------------
    st.subheader("🧾 Patient Report")

    st.markdown(f"""
    <div class="report-box">

    <h3>Patient Information</h3>

    <b>Name:</b> {patient_name}<br>
    <b>Patient ID:</b> {patient_id}<br>
    <b>Age:</b> {patient_age}<br>
    <b>Date:</b> {datetime.datetime.now().strftime('%d-%m-%Y')}<br><br>

    <h3>Diagnosis Summary</h3>

    <b>Diagnosis:</b> {diagnosis}<br>
    <b>Confidence Score:</b> {confidence:.2f}%<br>
    <b>Tumor Area Percentage:</b> {tumor_area:.2f}%<br><br>

    <h3>Recommendation</h3>

    Please consult neurologist / radiologist for further evaluation.

    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------
    # DOWNLOAD
    # -------------------------------------------------
    report = f"""
BRAIN TUMOR DETECTION REPORT

Patient Name: {patient_name}
Patient ID: {patient_id}
Age: {patient_age}
Date: {datetime.datetime.now().strftime('%d-%m-%Y')}

Diagnosis: {diagnosis}
Confidence Score: {confidence:.2f}%
Tumor Area Percentage: {tumor_area:.2f}%

Recommendation:
Please consult neurologist / radiologist.
"""

    st.markdown("##")

    st.download_button(
        label="📄 Download Patient Report",
        data=report,
        file_name="Patient_Report.txt",
        mime="text/plain"
    )

else:
    st.markdown("""
    <div class="report-box">
        <h3>📤 Upload MRI Image</h3>
        <p>Please upload MRI scan from sidebar to begin AI analysis.</p>
    </div>
    """, unsafe_allow_html=True)