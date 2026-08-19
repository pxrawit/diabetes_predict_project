import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS แบบเต็มรูปแบบ - Modern Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    /* ตั้งค่าฟอนต์ */
    * {
        font-family: 'Prompt', sans-serif !important;
    }
    
    /* ซ่อนเมนู Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* พื้นหลังหลัก - Gradient Animation */
    .main {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Navigation Menu แบบ Modern */
    .nav-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .nav-button {
        display: inline-block;
        padding: 0.8rem 1.5rem;
        margin: 0 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        text-decoration: none;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        cursor: pointer;
        border: none;
    }
    
    .nav-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.5);
    }
    
    /* หัวข้อหลัก */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        border-radius: 20px;
        background-color: rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        animation: fadeInDown 1s ease;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* การ์ดเมตริกแบบ Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        animation: fadeInUp 1s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card h3 {
        color: #667eea;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        color: #2d3748;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #718096;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* กล่องข้อมูลผู้พัฒนา */
    .developer-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        animation: slideInLeft 1s ease;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .dev-avatar {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 5px solid white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .dev-avatar:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    .dev-name {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .dev-info {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.8rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .dev-info p {
        margin: 0.6rem 0;
        font-size: 0.9rem;
        text-align: left;
        padding-left: 0.5rem;
    }
    
    /* ปุ่มทำนาย */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        border: none;
        width: 100%;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        50% {
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        }
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
        animation: none;
    }
    
    /* การ์ดผลลัพธ์ */
    .result-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.9) 0%, rgba(220, 38, 38, 0.9) 100%);
        border-left: 8px solid #dc2626;
        padding: 2.5rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
        color: white;
        backdrop-filter: blur(10px);
        animation: shake 0.5s ease;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    .result-card-low {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.9) 0%, rgba(22, 163, 74, 0.9) 100%);
        border-left: 8px solid #16a34a;
        padding: 2.5rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3);
        color: white;
        backdrop-filter: blur(10px);
        animation: bounce 0.5s ease;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    /* Progress bar สวยๆ */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        height: 12px;
    }
    
    /* Input fields */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 0.8rem;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.5);
        font-weight: 600;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.5);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# โหลดและเตรียมข้อมูล
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    return df

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
    
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
    )
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

# โหลดข้อมูล
with st.spinner("🔄 กำลังเตรียมระบบ..."):
    metrics = build_model()
    df_raw = load_data()

# Navigation Menu แบบ Modern
st.markdown("""
<div class="nav-container">
    <div style="text-align: center;">
        <a href="#" class="nav-button">🏠 หน้าหลัก</a>
        <a href="#" class="nav-button">📊 วิเคราะห์ข้อมูล</a>
        <a href="#" class="nav-button">🤖 ประสิทธิภาพโมเดล</a>
        <a href="#" class="nav-button"> ทายผลความเสี่ยง</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="developer-box">', unsafe_allow_html=True)
profile_image = "https://ui-avatars.com/api/?name=Phuwadit+Cham&background=ffffff&color=667eea&size=200&font-size=0.4"
st.sidebar.markdown(f'<img src="{profile_image}" class="dev-avatar" alt="Profile">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="dev-name">นาย ภูวฤทธิ์ แช่มมั่นคง</div>', unsafe_allow_html=True)
st.sidebar.markdown('''
    <div class="dev-info">
        <p>👤 <strong>ชื่อ-สกุล:</strong> นาย ภูวฤทธิ์ แช่มมั่นคง</p>
        <p> <strong>รหัสนักศึกษา:</strong> 664245031</p>
        <p>🏫 <strong>หมู่เรียน:</strong> 66/44</p>
    </div>
''', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# เมนูแบบ Tabs
page = st.sidebar.radio(
    " เลือกเมนู",
    ["🏠 หน้าหลัก", "📊 วิเคราะห์ข้อมูล", "🤖 ประสิทธิภาพโมเดล", "🎮 ทายผลความเสี่ยง"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #64748B; font-size: 0.8rem;'>📅 ปีการศึกษา 2026<br> Machine Learning Project</div>", unsafe_allow_html=True)

# หน้าหลัก
if "หน้าหลัก" in page:
    st.markdown('<div class="main-header">🩺 ระบบพยากรณ์โรคเบาหวานด้วย AI</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📦 ข้อมูล</h3><h2>100,000+</h2><p>แถวข้อมูลทางการแพทย์</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠 โมเดล</h3><h2>Random Forest</h2><p>Ensemble Learning</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🎯 ความแม่นยำ</h3><h2>95%+</h2><p>Accuracy Score</p></div>', unsafe_allow_html=True)

    st.markdown('<div style="background: rgba(255,255,255,0.9); padding: 2rem; border-radius: 20px; margin-top: 2rem; backdrop-filter: blur(10px);"><h2 style="color: #667eea; margin-bottom: 1rem;">🎯 วัตถุประสงค์</h2><p style="font-size: 1.1rem; line-height: 1.8;">💡 <strong>พัฒนาโมเดล Machine Learning</strong> เพื่อคัดกรองความเสี่ยงโรคเบาหวานล่วงหน้า<br>🔬 <strong>วิเคราะห์ปัจจัยสำคัญ</strong> ที่ส่งผลต่อการเกิดโรค<br>🌐 <strong>สร้าง Web Application</strong> ที่ใช้งานง่ายและสวยงาม</p></div>', unsafe_allow_html=True)

# หน้าวิเคราะห์ข้อมูล
elif "วิเคราะห์ข้อมูล" in page:
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        counts = df_raw['diabetes'].value_counts()
        ax.pie(counts, labels=['ไม่เป็น (0)', 'เป็น (1)'], autopct='%1.1f%%', 
                colors=['#22c55e', '#ef4444'], startangle=90)
        ax.set_title('Distribution of Diabetes', fontweight='bold', fontsize=14, color='#2d3748')
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=df_raw.sample(2000), x='age', y='blood_glucose_level', 
                       hue='diabetes', palette=['#22c55e', '#ef4444'], alpha=0.6)
        ax.set_title('Age vs Blood Glucose', fontweight='bold', fontsize=14, color='#2d3748')
        st.pyplot(fig)

# หน้าประสิทธิภาพโมเดล
elif "ประสิทธิภาพโมเดล" in page:
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1']:.2%}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14, color='#2d3748')
    st.pyplot(fig)

# หน้าทำนายผล
elif "ทายผลความเสี่ยง" in page:
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยง</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("ความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
    
    with col2:
        smoking = st.selectbox("สูบบุหรี่", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("น้ำตาลในเลือด (mg/dL)", 50, 400, 100)

    if st.button(" ทำนายผล"):
        gender_enc = metrics['le_gender'].transform([gender])[0]
        smoking_enc = metrics['le_smoking'].transform([smoking])[0]
        
        input_data = np.array([[gender_enc, age, hypertension, heart_disease, 
                               smoking_enc, bmi, hba1c, glucose]])
        input_scaled = metrics['scaler'].transform(input_data)
        
        prediction = metrics['model'].predict(input_scaled)[0]
        proba = metrics['model'].predict_proba(input_scaled)[0]
        risk = proba[1] * 100
        
        st.markdown("---")
        if prediction == 1:
            st.markdown(f'''
                <div class="result-card-high">
                    <h2 style="color: white; margin: 0;">⚠️ มีความเสี่ยง</h2>
                    <p style="font-size: 1.3rem; margin: 1rem 0;">
                        โอกาสเป็นโรคเบาหวาน: <strong>{risk:.1f}%</strong>
                    </p>
                    <p style="font-size: 1rem;">💡 ควรปรึกษาแพทย์และควบคุมอาหาร</p>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="result-card-low">
                    <h2 style="color: white; margin: 0;">✅ ความเสี่ยงต่ำ</h2>
                    <p style="font-size: 1.3rem; margin: 1rem 0;">
                        โอกาสเป็นโรคเบาหวาน: <strong>{risk:.1f}%</strong>
                    </p>
                    <p style="font-size: 1rem;">💡 สุขภาพดี! ตรวจสุขภาพเป็นประจำ</p>
                </div>
            ''', unsafe_allow_html=True)
        
        st.progress(float(risk / 100))