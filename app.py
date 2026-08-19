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
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Prompt:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Prompt', sans-serif !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .main { background: #f8fafc; }
    .app-header {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        padding: 2rem; border-radius: 0 0 30px 30px;
        box-shadow: 0 10px 40px rgba(13, 148, 136, 0.2);
        margin-bottom: 2rem; text-align: center;
    }
    .app-header h1 { color: white; font-size: 2.5rem; font-weight: 700; margin: 0; }
    .app-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem; }
    .info-card {
        background: white; padding: 2rem; border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem; border-left: 5px solid #14b8a6;
    }
    .stat-card {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
        padding: 1.5rem; border-radius: 15px;
        text-align: center; border: 2px solid #99f6e4;
    }
    .stat-card h3 { color: #0f766e; font-size: 2rem; margin: 0; font-weight: 700; }
    .stat-card p { color: #14b8a6; margin: 0.5rem 0 0 0; font-size: 0.9rem; font-weight: 600; }
    .profile-card {
        background: white; padding: 2rem; border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center; margin-bottom: 2rem;
    }
    .profile-avatar {
        width: 120px; height: 120px; border-radius: 50%;
        border: 5px solid #14b8a6; margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.3);
    }
    .profile-name { font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; }
    .profile-info { background: #f0fdfa; padding: 1rem; border-radius: 10px; margin-top: 1rem; }
    .stButton > button {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        color: white !important; font-weight: 700; padding: 1rem 2rem;
        border-radius: 12px; border: none;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.3);
    }
    .result-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981; padding: 2rem;
        border-radius: 15px; color: #065f46;
    }
    .result-warning {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444; padding: 2rem;
        border-radius: 15px; color: #991b1b;
    }
    .section-title {
        font-size: 1.5rem; font-weight: 700; color: #0f172a;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 3px solid #14b8a6;
    }
</style>
""", unsafe_allow_html=True)

# โหลดข้อมูล
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
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
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

# Header
st.markdown("""
<div class="app-header">
    <h1>🏥 Diabetes Prediction System</h1>
    <p>ระบบพยากรณ์โรคเบาหวานด้วย Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.radio("เลือกเมนู", [" ภาพรวม", "📈 วิเคราะห์ข้อมูล", "🤖 โมเดล", " ทำนายผล"], horizontal=True, label_visibility="collapsed")

# โหลดข้อมูลและโมเดล
with st.spinner("⏳ กำลังโหลดระบบ..."):
    metrics = build_model()
    df_raw = load_data()

# ==================== TAB 1: ภาพรวม ====================
if page == "📊 ภาพรวม":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        <div class="profile-card">
            <img src="https://ui-avatars.com/api/?name=Phuwadit+Cham&background=14b8a6&color=fff&size=200" 
                 class="profile-avatar" alt="Profile">
            <div class="profile-name">นาย ภูวฤทธิ์ แช่มมั่นคง</div>
            <div class="profile-info">
                <p><strong>รหัสนักศึกษา:</strong> 664245031</p>
                <p><strong>หมู่เรียน:</strong> 66/44</p>
                <p><strong>ปีการศึกษา:</strong> 2026</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-title">เกี่ยวกับโปรเจกต์</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h3> วัตถุประสงค์</h3>
            <ul>
                <li>พัฒนาโมเดล Machine Learning สำหรับพยากรณ์โรคเบาหวาน</li>
                <li>วิเคราะห์ปัจจัยเสี่ยงที่สำคัญต่อการเกิดโรค</li>
                <li>สร้างระบบที่ช่วยในการคัดกรองเบื้องต้น</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">สถิติ Dataset</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-card"><h3>{len(df_raw):,}</h3><p>จำนวนข้อมูล</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><h3>{df_raw["diabetes"].sum()}</h3><p>ผู้ป่วยเบาหวาน</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><h3>{len(df_raw) - df_raw["diabetes"].sum():,}</h3><p>ไม่ป่วย</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card"><h3>8</h3><p>Features</p></div>', unsafe_allow_html=True)

# ==================== TAB 2: วิเคราะห์ข้อมูล ====================
elif page == "📈 วิเคราะห์ข้อมูล":
    st.markdown('<div class="section-title">การวิเคราะห์ข้อมูลเชิงสำรวจ</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-card"><h3>🥧 Distribution</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        counts = df_raw['diabetes'].value_counts()
        ax.pie(counts, labels=['ไม่เป็น', 'เป็น'], autopct='%1.1f%%', 
                colors=['#10b981', '#ef4444'], startangle=90)
        ax.set_title('สัดส่วนผู้ป่วยเบาหวาน', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="info-card"><h3>📊 Age Distribution</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.histplot(data=df_raw, x='age', hue='diabetes', multiple='stack', palette=['#10b981', '#ef4444'])
        ax.set_title('การกระจายตัวของอายุ', fontweight='bold')
        st.pyplot(fig)
    
    st.markdown('<div class="info-card"><h3> Correlation Matrix</h3></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df_raw.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='teal', fmt='.2f', ax=ax)
    ax.set_title('Correlation Heatmap', fontweight='bold', pad=15)
    st.pyplot(fig)

# ==================== TAB 3: โมเดล ====================
elif page == "🤖 โมเดล":
    st.markdown('<div class="section-title">Random Forest Classifier</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>📚 เกี่ยวกับอัลกอริทึม</h3>
        <p>Random Forest เป็น Ensemble Learning ที่สร้าง Decision Trees หลายต้น 
        และนำผลมาโหวตเพื่อทำนายผล มีข้อดีคือป้องกัน Overfitting และทำงานได้รวดเร็ว</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><h3>{metrics["accuracy"]:.1%}</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><h3>{metrics["precision"]:.1%}</h3><p>Precision</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><h3>{metrics["recall"]:.1%}</h3><p>Recall</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><h3>{metrics["f1"]:.2f}</h3><p>F1-Score</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-card"><h3> Confusion Matrix</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='teal', ax=ax)
        ax.set_title('Confusion Matrix', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="info-card"><h3>📈 ROC Curve</h3></div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='#0d9488', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.set_title('ROC Curve', fontweight='bold')
        st.pyplot(fig)

# ==================== TAB 4: ทำนายผล ====================
elif page == "🎯 ทำนายผล":
    st.markdown('<div class="section-title">แบบฟอร์มประเมินความเสี่ยง</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card"><h3>👤 ข้อมูลส่วนตัว</h3></div>', unsafe_allow_html=True)
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("ความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
    
    with col2:
        st.markdown('<div class="info-card"><h3>🩺 ข้อมูลสุขภาพ</h3></div>', unsafe_allow_html=True)
        smoking = st.selectbox("สูบบุหรี่", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("น้ำตาลในเลือด (mg/dL)", 50, 400, 100)
    
    if st.button("🔮 ประเมินความเสี่ยง", use_container_width=True):
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
            st.markdown(f"""
            <div class="result-warning">
                <h2>⚠️ มีความเสี่ยงเป็นโรคเบาหวาน</h2>
                <p style="font-size: 1.3rem;"><strong>โอกาส: {risk:.1f}%</strong></p>
                <p> ควรปรึกษาแพทย์และตรวจสุขภาพอย่างละเอียด</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-success">
                <h2>✅ ความเสี่ยงต่ำ</h2>
                <p style="font-size: 1.3rem;"><strong>โอกาส: {risk:.1f}%</strong></p>
                <p>💡 สุขภาพดี! ควรตรวจสุขภาพเป็นประจำ</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.progress(float(risk / 100))