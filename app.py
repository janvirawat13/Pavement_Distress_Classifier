import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import cv2
import pandas as pd
from geopy.geocoders import Nominatim
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import sqlite3
import io
import plotly.express as px

# -------------------------------------------------
# INITIALIZE ALL SESSION STATE KEYS
# -------------------------------------------------
if "page" not in st.session_state: st.session_state.page = "Home"
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_name" not in st.session_state: st.session_state.user_name = ""
if "location" not in st.session_state: st.session_state.location = ""
if "lat" not in st.session_state: st.session_state.lat = 0.0
if "lon" not in st.session_state: st.session_state.lon = 0.0

# -------------------------------------------------
# DATABASE & PERSISTENCE
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect('road_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS audits 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT, 
                  location TEXT, 
                  latitude REAL, 
                  longitude REAL, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_audit(name, location, lat, lon):
    conn = sqlite3.connect('road_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO audits (name, location, latitude, longitude) VALUES (?, ?, ?, ?)", 
              (name, location, lat, lon))
    conn.commit()
    conn.close()

init_db()

# -------------------------------------------------
# PAGE CONFIG & CSS
# -------------------------------------------------
st.set_page_config(
    page_title="RoadEye AI | Infrastructure Monitoring",
    page_icon="🛣️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url("https://images.unsplash.com/photo-1503376780353-7e6692767b70");
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}

.home-title {
    font-size: 80px !important;
    font-weight: 800;
    color: #ffffff;
    text-align: center;
    letter-spacing: -2px;
}

.home-subtitle {
    font-size: 22px;
    color: #00d4ff;
    text-align: center;
    margin-bottom: 40px;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 30px;
    padding: 40px;
}

.dashboard-card {
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 30px;
    height: 250px;
    text-align: center;
    transition: 0.3s;
}

.dashboard-card:hover {
    background: rgba(255,255,255,0.2);
    transform: translateY(-10px);
}

.status-text {
    font-size: 35px !important;
    font-weight: 700;
    color: #00ff88;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return YOLO(r"D:\image_classifiction\runs\detect\train2\weights\best.pt")
    except:
        return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------------------------
# PDF REPORT
# -------------------------------------------------
def generate_pdf(data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Road Damage Audit Report", styles['Title']), Spacer(1, 20)]

    for item in data:
        text = f"<b>User:</b> {item['name']}<br/><b>City:</b> {item['location']}<br/><b>Coordinates:</b> {item['latitude']}, {item['longitude']}<hr/>"
        elements.append(Paragraph(text, styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
if st.session_state.logged_in:
    with st.sidebar:
        st.title("Control Panel")

        if st.button("🏠 Home"): st.session_state.page="Home"
        if st.button("📊 Dashboard"): st.session_state.page="Dashboard"
        if st.button("🖼️ Image Detection"): st.session_state.page="Image Detection"
        if st.button("🎥 Video Detection"): st.session_state.page="Video Detection"
        if st.button("📡 Real-Time"): st.session_state.page="Realtime"
        if st.button("📈 Analytics"): st.session_state.page="Analysis"

        if st.button("Logout"):
            st.session_state.logged_in=False
            st.session_state.page="Home"
            st.rerun()

# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
if st.session_state.page == "Home":

    st.markdown('<p class="home-title">RoadEye AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="home-subtitle">Precision Computer Vision for Safer Infrastructure</p>', unsafe_allow_html=True)

    _,col,_ = st.columns([1,1.2,1])

    with col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        u_name = st.text_input("Operator Name")
        u_loc = st.text_input("Target City")

        if st.button("LAUNCH SYSTEM", use_container_width=True):

            if u_name and u_loc:

                geolocator = Nominatim(user_agent="road_ai")
                loc = geolocator.geocode(u_loc)

                if loc:
                    st.session_state.user_name = u_name
                    st.session_state.location = u_loc
                    st.session_state.lat = loc.latitude
                    st.session_state.lon = loc.longitude
                    st.session_state.logged_in = True
                    st.session_state.page = "Dashboard"
                    st.rerun()

                else:
                    st.error("City not recognized.")

            else:
                st.warning("Please enter credentials.")

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
elif st.session_state.page == "Dashboard":

    st.title(f"Welcome back, {st.session_state.user_name}")

    def draw_card(title, desc, icon, target):
        st.markdown(f"<div class='dashboard-card'><h1>{icon}</h1><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)
        if st.button(f"Start {title}", key=title):
            st.session_state.page = target
            st.rerun()

    c1,c2,c3,c4 = st.columns(4)

    with c1: draw_card("Images","Analyze road photos","📸","Image Detection")
    with c2: draw_card("Videos","Process footage","🎞️","Video Detection")
    with c3: draw_card("Live","Webcam/IP monitoring","📡","Realtime")
    with c4: draw_card("Data","Statistics & maps","📊","Analysis")

# -------------------------------------------------
# IMAGE DETECTION
# -------------------------------------------------
elif st.session_state.page == "Image Detection":

    st.title("Static Image Analysis")

    file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

    if file:

        image = Image.open(file)
        st.image(image)

        img = np.array(image)

        if st.button("Detect Damage"):

            results = model(img)
            annotated = results[0].plot()

            st.image(annotated)

            boxes = results[0].boxes
            count = len(boxes) if boxes is not None else 0

            st.success(f"Total Damages Detected: {count}")

            save_audit(
                st.session_state.user_name,
                st.session_state.location,
                st.session_state.lat,
                st.session_state.lon
            )

# -------------------------------------------------
# VIDEO DETECTION
# -------------------------------------------------
elif st.session_state.page == "Video Detection":

    st.title("Video Stream Processing")

    vid_file = st.file_uploader("Upload MP4", type=["mp4"])

    if vid_file:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())

        cap = cv2.VideoCapture(tfile.name)

        st_frame = st.image([])

        stop = st.button("Stop Processing")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret: break
            res = model(frame)
            st_frame.image(res[0].plot(), channels="BGR")
        
            boxes = res[0].boxes
            count = len(boxes) if boxes is not None else 0
        
            if count > 0:
                save_audit(
                    st.session_state.user_name,
                    st.session_state.location,
                    st.session_state.lat,
                    st.session_state.lon
                )

# -------------------------------------------------
# REALTIME
# -------------------------------------------------
elif st.session_state.page == "Realtime":

    st.markdown('<p class="status-text">LIVE SYSTEM ACTIVE</p>', unsafe_allow_html=True)

    src = st.radio("Input Type", ["Webcam","IP Stream"])
    run = st.checkbox("Toggle Camera")

    st_window = st.image([])

    if run:

        path = 0 if src=="Webcam" else st.text_input("Enter Stream URL")

        cap = cv2.VideoCapture(path)

        while run:
            ret, frame = cap.read()
            if not ret: break
        
            res = model(frame)
            st_window.image(res[0].plot(), channels="BGR")
        
            boxes = res[0].boxes
            count = len(boxes) if boxes is not None else 0
        
            if count > 0:
                save_audit(
                    st.session_state.user_name,
                    st.session_state.location,
                    st.session_state.lat,
                    st.session_state.lon
                )

# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------
elif st.session_state.page == "Analysis":

    st.markdown('<p class="status-text">📊 REGIONAL INTELLIGENCE</p>', unsafe_allow_html=True)

    conn = sqlite3.connect('road_data.db')
    df = pd.read_sql_query("SELECT * FROM audits", conn)
    conn.close()

    if df.empty:
        st.info("No audit data yet.")

    else:

        st.dataframe(df)

        map_df = df.rename(columns={'latitude':'lat','longitude':'lon'})
        st.map(map_df)

        fig = px.density_mapbox(
            df,
            lat='latitude',
            lon='longitude',
            radius=9,
            zoom=3,
            mapbox_style="carto-darkmatter"
        )

        st.plotly_chart(fig)

        report_data = df.to_dict('records')
        pdf = generate_pdf(report_data)

        st.download_button(
            "📥 EXPORT OFFICIAL REPORT",
            pdf,
            "Road_Audit_Report.pdf",
            "application/pdf"
        )