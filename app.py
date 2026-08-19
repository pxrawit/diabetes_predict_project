import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# โหลด CSS
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Prompt', sans-serif !important; }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        animation: gradientBG 15s ease infinite;
        min-height: 100vh;
        padding: 2rem;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(16px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card h3 {
        color: #667eea !important;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        color: #1e293b !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #64748b !important;
        font-size: 0.85rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.2);
        padding: 6px;
        border-radius: 50px;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 1.5rem;
        border-radius: 50px;
        background: transparent;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff !important;
        color: #667eea !important;
    }
    
    .developer-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(16px);
        padding: 3rem 2rem;
        border-radius: 24px;
        max-width: 600px;
        margin: 0 auto;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    }
    
    .dev-avatar {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #667eea;
        box-shadow: 0 0 25px rgba(102, 126, 234, 0.4);
        margin-bottom: 1.5rem;
    }
    
    .dev-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b !important;
        margin-bottom: 1.5rem;
    }
    
    .dev-info {
        background: rgba(241, 245, 249, 0.9);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: left;
    }
    
    .dev-info p {
        margin: 0.8rem 0;
        font-size: 1.05rem;
        color: #334155 !important;
    }
    
    .dev-info strong {
        color: #667eea !important;
    }
    
    .result-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.95) 0%, rgba(220, 38, 38, 0.95) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        color: #ffffff !important;
    }
    
    .result-card-low {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.95) 0%, rgba(22, 163, 74, 0.95) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        color: #ffffff !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# โหลดข้อมูล
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')

# สร้างโมเดล
@st.cache_resource
def build_model():
    df = load_data()
    
    # Preprocessing
    le_gender = LabelEncoder()
    le_smoking = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return {
        'model': model,
        'scaler': scaler,
        'le_gender': le_gender,
        'le_smoking': le_smoking,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': X.columns.tolist(),
        'df': df
    }

# โหลดข้อมูลและโมเดล
with st.spinner("🔄 กำลังเตรียมระบบ..."):
    metrics = build_model()
    df_raw = load_data()

# Tabs Navigation
tabs = st.tabs([
    "🏠 หน้าหลัก",
    "📊 วิเคราะห์ข้อมูล",
    "🤖 ประสิทธิภาพโมเดล",
    "🎮 ทายผลความเสี่ยง",
    "👨‍💻 ผู้พัฒนา"
])

# ==================== TAB 1: หน้าหลัก ====================
with tabs[0]:
    st.markdown('<div class="main-header"> ระบบพยากรณ์โรคเบาหวานด้วย AI</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📦 ข้อมูล</h3><h2>90,000+</h2><p>แถวข้อมูลทางการแพทย์</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠 โมเดล</h3><h2>Random Forest</h2><p>Ensemble Learning</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🎯 ความแม่นยำ</h3><h2>95%+</h2><p>Accuracy Score</p></div>', unsafe_allow_html=True)

    st.markdown('''
    <div class="metric-card" style="margin-top: 2rem; text-align: left;">
        <h3 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 1rem;"> วัตถุประสงค์</h3>
        <p style="color: #1e293b; line-height: 1.8; font-size: 1rem;">
        💡 <strong>พัฒนาโมเดล Machine Learning</strong> เพื่อคัดกรองความเสี่ยงโรคเบาหวานล่วงหน้า<br>
        🔬 <strong>วิเคราะห์ปัจจัยสำคัญ</strong> ที่ส่งผลต่อการเกิดโรค<br>
        🌐 <strong>สร้าง Web Application</strong> ที่ใช้งานง่ายและสวยงาม
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
        ax.pie(counts, labels=['ไม่เป็น (0)', 'เป็น (1)'], autopct='%1.1f%%', 
               colors=['#10b981', '#f43f5e'], startangle=90, 
               textprops={'color': 'white', 'fontweight': 'bold'})
        ax.set_title('Distribution of Diabetes', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
        fig.patch.set_alpha(0.0)
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=df_raw.sample(2000), x='age', y='blood_glucose_level', 
                       hue='diabetes', palette=['#10b981', '#f43f5e'], alpha=0.7, ax=ax)
        ax.set_title('Age vs Blood Glucose', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.legend(labels=['Non-Diabetic', 'Diabetic'], facecolor=(1, 1, 1, 0.1), edgecolor='none', labelcolor='white')
        fig.patch.set_alpha(0.0)
        st.pyplot(fig)
    
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_raw.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, annot_kws={'color': 'white', 'size': 9})
    ax.set_title('Feature Correlation', color='#e2e8f0', fontweight='bold', pad=15)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('#1e293b')
    plt.xticks(color='#94a3b8')
    plt.yticks(color='#94a3b8')
    st.pyplot(fig)

# ==================== TAB 3: ประสิทธิภาพโมเดล ====================
with tabs[2]:
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><h2>{metrics["y_pred_proba"].shape[0]}%</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Precision</h3><h2>{precision_score(metrics["y_test"], metrics["y_pred"]):.1%}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Recall</h3><h2>{recall_score(metrics["y_test"], metrics["y_pred"]):.1%}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>F1-Score</h3><h2>{f1_score(metrics["y_test"], metrics["y_pred"]):.2f}</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 14, "color": "white", "fontweight": "bold"})
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)
    
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">📈 ROC Curve</div>', unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#06b6d4', lw=3, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='#64748b', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate', color='#94a3b8')
    ax.set_ylabel('True Positive Rate', color='#94a3b8')
    ax.set_title('ROC Curve', color='#e2e8f0', fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('#1e293b')
    st.pyplot(fig)

# ==================== TAB 4: ทายผลความเสี่ยง ====================
with tabs[3]:
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยง</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0; color:#667eea;">👤 ข้อมูลส่วนบุคคล</h3></div>', unsafe_allow_html=True)
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("ความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
    
    with col2:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0; color:#667eea;"> ข้อมูลทางการแพทย์</h3></div>', unsafe_allow_html=True)
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
                    <p style="font-size: 1rem; color: #fda4af;"> ควรปรึกษาแพทย์และควบคุมอาหาร</p>
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
    
    # โหลดรูป
    img_html = ""
    if os.path.exists("profile.jpg"):
        with open("profile.jpg", "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
            img_html = f'<img src="data:image/jpeg;base64,{img_data}" class="dev-avatar" alt="Profile">'
    else:
        img_html = '<img src="https://ui-avatars.com/api/?name=Phuwadit+Cham&background=667eea&color=fff&size=200&font-size=0.4" class="dev-avatar" alt="Profile">'

    st.markdown(f'''
        <div class="developer-card">
            {img_html}
            <div class="dev-name">นาย ภูวฤทธิ์ แช่มมั่นคง</div>
            <div class="dev-info">
                <p>👤 <strong>ชื่อ-สกุล:</strong> นาย ภูวฤทธิ์ แช่มมั่นคง</p>
                <p> <strong>รหัสนักศึกษา:</strong> 664245031</p>
                <p>🏫 <strong>หมู่เรียน:</strong> 66/44</p>
                <p>📅 <strong>ปีการศึกษา:</strong> 2026</p>
                <p>💻 <strong>โปรเจกต์:</strong> Machine Learning Project</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)