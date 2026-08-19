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

# ตั้งค่าหน้าเว็บ (initial_sidebar_state="expanded" จะทำให้มีปุ่ม 3 ขีดที่มุมซ้ายบน)
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None
    }
)
# ==================== Custom CSS แบบเต็มรูปแบบ ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Prompt', sans-serif !important; }
    
    /* ซ่อนเมนู Streamlit ด้านขวาบน */
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;} 
    header {visibility: hidden;}
    
    /* พื้นหลังหลัก */
    .main { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); min-height: 100vh; }
    
    /* ปรับแต่ง Sidebar ให้สวยงาม */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    /* ปรับแต่งเมนู Radio ใน Sidebar ให้เหมือนปุ่มกด */
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent !important;
        gap: 8px;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
        font-weight: 500;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(56, 189, 248, 0.1) !important;
        border-color: rgba(56, 189, 248, 0.3);
        color: #38bdf8 !important;
    }
    [data-testid="stSidebar"] .stRadio input:checked + div {
        background: linear-gradient(90deg, #0ea5e9 0%, #06b6d4 100%) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
        border: 1px solid rgba(255,255,255,0.2);
    }
    [data-testid="stSidebar"] .stRadio input:checked + div span {
        color: white !important;
    }
    
    /* กล่องข้อมูลผู้พัฒนาใน Sidebar */
    .developer-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .dev-avatar {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
        margin-bottom: 0.8rem;
    }
    .dev-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.8rem;
    }
    .dev-info {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: left;
    }
    .dev-info p {
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .dev-info strong {
        color: #38bdf8;
    }
    
    /* หัวข้อหลัก */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #0ea5e9;
    }
    
    /* การ์ดเมตริก */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        border: 1px solid #e2e8f0;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(14, 165, 233, 0.15);
        border-color: #38bdf8;
    }
    .metric-card h3 {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h2 {
        color: #0f172a;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-card p {
        color: #0ea5e9;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* ปุ่ม */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.5);
    }
    
    /* การ์ดผลลัพธ์ */
    .result-card-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
    }
    .result-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #22c55e;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
    }
    
    /* Sub Header */
    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #334155;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 1rem;
        border-left: 4px solid #0ea5e9;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9 0%, #06b6d4 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== โหลดและเตรียมข้อมูล ====================
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

# ==================== Sidebar Navigation ====================
# ส่วนข้อมูลผู้พัฒนาใน Sidebar
st.sidebar.markdown('<div class="developer-box">', unsafe_allow_html=True)
profile_image = "https://ui-avatars.com/api/?name=Phuwadit+Cham&background=0ea5e9&color=fff&size=200&font-size=0.4"
st.sidebar.markdown(f'<img src="{profile_image}" class="dev-avatar" alt="Profile">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="dev-name">นาย ภูวฤทธิ์ แช่มมั่นคง</div>', unsafe_allow_html=True)
st.sidebar.markdown('''
    <div class="dev-info">
        <p>👤 <strong>ชื่อ-สกุล:</strong> นาย ภูวฤทธิ์ แช่มมั่นคง</p>
        <p>🆔 <strong>รหัสนักศึกษา:</strong> 664245031</p>
        <p>🏫 <strong>หมู่เรียน:</strong> 66/44</p>
    </div>
''', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# เมนูเลือกหน้า (จะมีปุ่ม 3 ขีดที่มุมซ้ายบนของหน้าจอสำหรับเปิด-ปิด Sidebar)
page = st.sidebar.radio(
    "🧭 เมนูหลัก",
    ["🏠 หน้าหลัก", "📊 วิเคราะห์ข้อมูล", "🤖 ประสิทธิภาพโมเดล", "🎮 ทายผลความเสี่ยง"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>📅 ปีการศึกษา 2026<br>🏥 Machine Learning Project</div>", unsafe_allow_html=True)

# ==================== หน้าหลัก ====================
if page == "🏠 หน้าหลัก":
    st.markdown('<div class="main-header">🩺 ระบบพยากรณ์โรคเบาหวานด้วย AI</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📦 ข้อมูล</h3><h2>100,000+</h2><p>แถวข้อมูลทางการแพทย์</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠 โมเดล</h3><h2>Random Forest</h2><p>Ensemble Learning</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🎯 ความแม่นยำ</h3><h2>95%+</h2><p>Accuracy Score</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">🎯 วัตถุประสงค์</div>', unsafe_allow_html=True)
    st.info("""
    💡 **พัฒนาโมเดล Machine Learning** เพื่อคัดกรองความเสี่ยงโรคเบาหวานล่วงหน้า  
    🔬 **วิเคราะห์ปัจจัยสำคัญ** ที่ส่งผลต่อการเกิดโรค  
    🌐 **สร้าง Web Application** ที่ใช้งานง่ายและสวยงาม
    """)

# ==================== หน้าวิเคราะห์ข้อมูล ====================
elif page == "📊 วิเคราะห์ข้อมูล":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        counts = df_raw['diabetes'].value_counts()
        ax.pie(counts, labels=['ไม่เป็น (0)', 'เป็น (1)'], autopct='%1.1f%%', 
                colors=['#22c55e', '#ef4444'], startangle=90)
        ax.set_title('Distribution of Diabetes', fontweight='bold', fontsize=14)
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=df_raw.sample(2000), x='age', y='blood_glucose_level', 
                       hue='diabetes', palette=['#22c55e', '#ef4444'], alpha=0.6)
        ax.set_title('Age vs Blood Glucose', fontweight='bold', fontsize=14)
        st.pyplot(fig)

# ==================== หน้าประสิทธิภาพโมเดล ====================
elif page == "🤖 ประสิทธิภาพโมเดล":
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1']:.2%}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    st.pyplot(fig)

# ==================== หน้าทำนายผล ====================
elif page == "🎮 ทายผลความเสี่ยง":
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยง</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 👤 ข้อมูลส่วนบุคคล")
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("ความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
    
    with col2:
        st.markdown("##### 🩸 ข้อมูลทางการแพทย์")
        smoking = st.selectbox("สูบบุหรี่", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("น้ำตาลในเลือด (mg/dL)", 50, 400, 100)

    if st.button("🔮 ทำนายผล"):
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
                    <h2 style="color: #b91c1c; margin: 0;">⚠️ มีความเสี่ยง</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                        โอกาสเป็นโรคเบาหวาน: <strong>{risk:.1f}%</strong>
                    </p>
                </div>
            ''', unsafe_allow_html=True)
            st.warning("💡 ควรปรึกษาแพทย์และควบคุมอาหาร")
        else:
            st.markdown(f'''
                <div class="result-card-low">
                    <h2 style="color: #047857; margin: 0;">✅ ความเสี่ยงต่ำ</h2>
                    <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                        โอกาสเป็นโรคเบาหวาน: <strong>{risk:.1f}%</strong>
                    </p>
                </div>
            ''', unsafe_allow_html=True)
            st.success("💡 สุขภาพดี! ตรวจสุขภาพเป็นประจำ")
        
        st.progress(float(risk / 100))