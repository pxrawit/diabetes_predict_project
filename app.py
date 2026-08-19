import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== ตั้งค่าหน้าเว็บ ====================
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== โหลด CSS จากไฟล์แยก ====================
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ style.css กรุณาตรวจสอบว่าวางไฟล์ไว้ในโฟลเดอร์เดียวกัน")

load_css("style.css")

# ==================== โหลดและเตรียมข้อมูล ====================
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')

@st.cache_resource
def build_model():
    df = load_data()
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
        'model': model, 'scaler': scaler,
        'le_gender': le_gender, 'le_smoking': le_smoking,
        'feature_names': X.columns.tolist()
    }

with st.spinner("🔄 กำลังเตรียมระบบ..."):
    metrics = build_model()
    df_raw = load_data()

# ==================== TABS NAVIGATION ====================
tabs = st.tabs(["🏠 หน้าหลัก", "📊 วิเคราะห์ข้อมูล", "🤖 ประสิทธิภาพโมเดล", "🎮 ทายผลความเสี่ยง", "👨‍💻 ผู้พัฒนา"])

# ==================== TAB 1: หน้าหลัก ====================
with tabs[0]:
    st.markdown('<div class="main-header">🩺 ระบบพยากรณ์โรคเบาหวานด้วย AI</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📦 ข้อมูล</h3><h2>90,000+</h2><p>แถวข้อมูลทางการแพทย์</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠 โมเดล</h3><h2>Random Forest</h2><p>Ensemble Learning</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🎯 ความแม่นยำ</h3><h2>95%+</h2><p>Accuracy Score</p></div>', unsafe_allow_html=True)

    st.markdown('''
    <div class="metric-card" style="margin-top: 2rem; text-align: left;">
        <h3 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 1rem;">🎯 วัตถุประสงค์</h3>
        <p style="color: #cbd5e1; line-height: 1.8; font-size: 1rem;">
        💡 <strong style="color: #67e8f9;">พัฒนาโมเดล Machine Learning</strong> เพื่อคัดกรองความเสี่ยงโรคเบาหวานล่วงหน้า<br>
        🔬 <strong style="color: #67e8f9;">วิเคราะห์ปัจจัยสำคัญ</strong> ที่ส่งผลต่อการเกิดโรค<br>
        🌐 <strong style="color: #67e8f9;">สร้าง Web Application</strong> ที่ใช้งานง่ายและสวยงาม
        </p>
    </div>
    ''', unsafe_allow_html=True)

# ==================== TAB 2: วิเคราะห์ข้อมูล ====================
with tabs[1]:
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        counts = df_raw['diabetes'].value_counts()
        ax.pie(counts, labels=['ไม่เป็น (0)', 'เป็น (1)'], autopct='%1.1f%%', colors=['#10b981', '#f43f5e'], startangle=90, textprops={'color': 'white', 'fontweight': 'bold'})
        ax.set_title('Distribution of Diabetes', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
        fig.patch.set_alpha(0.0)
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=df_raw.sample(2000), x='age', y='blood_glucose_level', hue='diabetes', palette=['#10b981', '#f43f5e'], alpha=0.7, ax=ax)
        ax.set_title('Age vs Blood Glucose', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.legend(labels=['Non-Diabetic', 'Diabetic'], facecolor='rgba(255,255,255,0.1)', edgecolor='none', labelcolor='white')
        fig.patch.set_alpha(0.0)
        st.pyplot(fig)

# ==================== TAB 3: ประสิทธิภาพโมเดล ====================
with tabs[2]:
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><h2>{metrics["accuracy"]:.1%}</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3>Precision</h3><h2>{metrics["precision"]:.1%}</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><h3>Recall</h3><h2>{metrics["recall"]:.1%}</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><h3>F1-Score</h3><h2>{metrics["f1"]:.2f}</h2></div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 14, "color": "white", "fontweight": "bold"})
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)

# ==================== TAB 4: ทายผลความเสี่ยง ====================
with tabs[3]:
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยง</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0;">👤 ข้อมูลส่วนบุคคล</h3></div>', unsafe_allow_html=True)
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("ความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
    
    with col2:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0;">🩸 ข้อมูลทางการแพทย์</h3></div>', unsafe_allow_html=True)
        smoking = st.selectbox("สูบบุหรี่", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("น้ำตาลในเลือด (mg/dL)", 50, 400, 100)

    if st.button("🔮 ทำนายผล"):
        gender_enc = metrics['le_gender'].transform([gender])[0]
        smoking_enc = metrics['le_smoking'].transform([smoking])[0]
        
        input_data = np.array([[gender_enc, age, hypertension, heart_disease, smoking_enc, bmi, hba1c, glucose]])
        input_scaled = metrics['scaler'].transform(input_data)
        
        prediction = metrics['model'].predict(input_scaled)[0]
        proba = metrics['model'].predict_proba(input_scaled)[0]
        risk = proba[1] * 100
        
        st.markdown("---")
        if prediction == 1:
            st.markdown(f'''
                <div class="result-card-high">
                    <h2 style="color: #fda4af; margin: 0;">⚠️ มีความเสี่ยง</h2>
                    <p style="font-size: 1.4rem; margin: 1rem 0; font-weight: 600; color: #ffffff;">
                        โอกาสเป็นโรคเบาหวาน: <strong>{risk:.1f}%</strong>
                    </p>
                    <p style="font-size: 1rem; color: #fda4af;">💡 ควรปรึกษาแพทย์และควบคุมอาหาร</p>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="result-card-low">
                    <h2 style="color: #6ee7b7; margin: 0;">✅ ความเสี่ยงต่ำ</h2>
                    <p style="font-size: 1.4rem; margin: 1rem 0; font-weight: 600; color: #ffffff;">
                        โอกาสเป็นโรคเบาหวาน: <strong>{risk:.1f}%</strong>
                    </p>
                    <p style="font-size: 1rem; color: #6ee7b7;">💡 สุขภาพดี! ตรวจสุขภาพเป็นประจำ</p>
                </div>
            ''', unsafe_allow_html=True)
        
        st.progress(float(risk / 100))

# ==================== TAB 5: ผู้พัฒนา ====================
with tabs[4]:
    st.markdown('<div class="main-header">👨‍💻 เกี่ยวกับผู้พัฒนา</div>', unsafe_allow_html=True)
    
    # ระบบโหลดรูปแบบปลอดภัย (ถ้าไม่มีไฟล์รูป จะไม่ Error)
    img_html = ""
    if os.path.exists("profile.jpg"):
        with open("profile.jpg", "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
            img_html = f'<img src="data:image/jpeg;base64,{img_data}" class="dev-avatar" alt="Profile">'
    else:
        # รูปสำรองหากยังไม่ได้ใส่รูป
        img_html = '<img src="https://ui-avatars.com/api/?name=Phuwadit+Cham&background=06b6d4&color=fff&size=200&font-size=0Name=Phuwadit+Cham&background=06b6d4&color=fff&size=200&font-size=0.4" class="dev-avatar" alt="Profile">'

    st.markdown(f'''
        <div class="developer-card">
            {img_html}
            <div class="dev-name">นาย ภูวฤทธิ์ แช่มมั่นคง</div>
            <div class="dev-info">
                <p>👤 <strong>ชื่อ-สกุล:</strong> นาย ภูวฤทธิ์ แช่มมั่นคง</p>
                <p>🆔 <strong>รหัสนักศึกษา:</strong> 664245031</p>
                <p>🏫 <strong>หมู่เรียน:</strong> 66/44</p>
                <p>📅 <strong>ปีการศึกษา:</strong> 2026</p>
                <p>💻 <strong>โปรเจกต์:</strong> Machine Learning Project</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)