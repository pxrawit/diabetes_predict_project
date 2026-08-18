import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# ==================== ตั้งค่าหน้าเว็บ ====================
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS สำหรับความสวยงาม ====================
st.markdown("""
<style>
    /* ซ่อนเมนู Streamlit ด้านขวาบน */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* หัวข้อหลัก */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    /* หัวข้อรอง */
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* การ์ดเมตริก */
    .metric-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #3B82F6;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* ปุ่มทำนาย */
    .stButton>button {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1D4ED8 0%, #1E3A8A 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* กล่องข้อมูลผู้พัฒนา */
    .developer-box {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==================== โหลดและเตรียมข้อมูล (Cache เพื่อความเร็ว) ====================
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    return df

@st.cache_resource
def build_model():
    df = load_data()
    
    # Encode Categorical
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
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions for metrics
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'model': model,
        'scaler': scaler,
        'le_gender': le_gender,
        'le_smoking': le_smoking,
        'feature_names': X.columns.tolist()
    }
    return metrics

# โหลดข้อมูลและโมเดล
with st.spinner("🔄 กำลังเตรียมข้อมูลและโมเดล AI..."):
    metrics = build_model()
    df_raw = load_data()

# ==================== Sidebar Navigation & Developer Profile ====================
st.sidebar.markdown("""
<style>
    /* กล่องข้อมูลผู้พัฒนา */
    .developer-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 15px;
        padding: 20px 15px;
        border: 2px solid #3B82F6;
        text-align: center;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        margin-bottom: 10px;
    }
    
    /* กรอบรูปโปรไฟล์ - วงกลม */
    .profile-image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 15px;
    }
    
    .profile-image {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #3B82F6;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* ชื่อผู้พัฒนา */
    .developer-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    
    /* ข้อมูลผู้พัฒนา */
    .developer-info {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        margin-top: 8px;
        border-left: 4px solid #3B82F6;
        text-align: left;
    }
    
    .developer-info p {
        margin: 5px 0;
        font-size: 0.9rem;
        color: #374151;
    }
    
    .developer-info strong {
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🩺 Diabetes Prediction AI")
st.sidebar.markdown("---")

# ส่วนข้อมูลผู้พัฒนา
st.sidebar.markdown('<div class="developer-box">', unsafe_allow_html=True)

# รูปโปรไฟล์ (วงกลม พร้อมกรอบ)
# วิธีที่ 1: ใช้รูปจาก GitHub (แนะนำ) - วางไฟล์ profile.jpg ในโฟลเดอร์โปรเจกต์
profile_image = "profile.jpg"

# วิธีที่ 2: ถ้ายังไม่มีรูป ให้ใช้รูปตัวอย่างนี้ก่อน (comment บรรทัดบน แล้ว uncomment บรรทัดล่าง)
# profile_image = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"

st.sidebar.markdown(f'''
    <div class="profile-image-container">
        <img src="{profile_image}" class="profile-image" alt="Profile">
    </div>
''', unsafe_allow_html=True)

# ชื่อผู้พัฒนา
st.sidebar.markdown('<div class="developer-name">นาย ภูวฤทธิ์ แซ่มั่นคง</div>', unsafe_allow_html=True)

# ข้อมูลผู้พัฒนา
st.sidebar.markdown('''
    <div class="developer-info">
        <p>👤 <strong>ชื่อ-สกุล:</strong> นาย ภูวฤทธิ์ แช่มมั่นคง</p>
        <p>🆔 <strong>รหัสนักศึกษา:</strong> 664245031</p>
        <p>🏫 <strong>หมู่เรียน:</strong> 66/44</p>
    </div>
''', unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# เมนูนำทาง
page = st.sidebar.radio(
    "เลือกเมนู",
    ["🏠 หน้าหลัก", "📊 วิเคราะห์ข้อมูล", "🤖 ประสิทธิภาพโมเดล", "🎮 ทายผลความเสี่ยง"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("###### 📅 ปีการศึกษา 2026")
st.sidebar.markdown("###### 🏥 Machine Learning Project")

# ==================== PAGE 1: หน้าหลัก ====================
if page == "🏠 หน้าหลัก":
    st.markdown('<div class="main-header">🩺 ระบบพยากรณ์โรคเบาหวานด้วย AI</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📦 ข้อมูล</h3><h2>100,000+</h2><p>แถวข้อมูลทางการแพทย์</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🧠 โมเดล</h3><h2>Random Forest</h2><p>Ensemble Learning</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🎯 ความแม่นยำ</h3><h2>95%+</h2><p>Accuracy Score</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-header">🎯 วัตถุประสงค์ของโปรเจกต์</div>', unsafe_allow_html=True)
    st.markdown("""
    - พัฒนาโมเดล Machine Learning เพื่อคัดกรองความเสี่ยงโรคเบาหวานล่วงหน้า
    - วิเคราะห์ปัจจัยสำคัญ (Features) ที่ส่งผลต่อการเกิดโรค เช่น ระดับน้ำตาล, BMI, อายุ
    - สร้าง Web Application ที่ใช้งานง่ายสำหรับบุคคลทั่วไปในการประเมินสุขภาพเบื้องต้น
    """)

    st.markdown('<div class="sub-header">📋 รายละเอียดข้อมูล (Dataset)</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(5), width="stretch", hide_index=True)

# ==================== PAGE 2: วิเคราะห์ข้อมูล ====================
elif page == "📊 วิเคราะห์ข้อมูล":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">🥧 สัดส่วนการเป็นโรคเบาหวาน</div>', unsafe_allow_html=True)
        counts = df_raw['diabetes'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(counts, labels=['ไม่เป็น (0)', 'เป็น (1)'], autopct='%1.1f%%', 
                colors=['#10B981', '#EF4444'], startangle=90, textprops={'color': "black", 'fontweight': 'bold'})
        ax1.set_title('Distribution of Diabetes', fontweight='bold')
        st.pyplot(fig1)
        
    with col2:
        st.markdown('<div class="sub-header">📈 อายุ vs ระดับน้ำตาลในเลือด</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        sns.scatterplot(data=df_raw.sample(2000), x='age', y='blood_glucose_level', hue='diabetes', palette=['#10B981', '#EF4444'], alpha=0.6, ax=ax2)
        ax2.set_title('Age vs Blood Glucose Level', fontweight='bold')
        st.pyplot(fig2)

    st.markdown('<div class="sub-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    numeric_df = df_raw.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3, linewidths=0.5)
    st.pyplot(fig3)

# ==================== PAGE 3: ประสิทธิภาพโมเดล ====================
elif page == "🤖 ประสิทธิภาพโมเดล":
    st.markdown('<div class="main-header">🤖 Random Forest Model Evaluation</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">📚 ทำไมต้อง Random Forest?</div>', unsafe_allow_html=True)
    st.info("""
    **Random Forest** เป็นอัลกอริทึมแบบ Ensemble ที่สร้าง Decision Tree หลายร้อยต้นและนำผลมาโหวตกัน 
    **ข้อดี:** ป้องกัน Overfitting ได้ดี, จัดการกับข้อมูลที่ไม่สมดุล (Imbalanced Data) ได้ยอดเยี่ยม, 
    และสามารถบอกความสำคัญของแต่ละปัจจัย (Feature Importance) ได้อย่างชัดเจน
    """)

    st.markdown('<div class="sub-header">📊 ผลการประเมิน (Evaluation Metrics)</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall", f"{metrics['recall']:.2%}")
    m4.metric("F1-Score", f"{metrics['f1']:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                    xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
        ax_cm.set_xlabel('Predicted', fontweight='bold')
        ax_cm.set_ylabel('Actual', fontweight='bold')
        st.pyplot(fig_cm)

    with col2:
        st.markdown('<div class="sub-header">📈 ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color='#2563EB', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

# ==================== PAGE 4: ทายผลความเสี่ยง (Prediction) ====================
elif page == "🎮 ทายผลความเสี่ยง":
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยงโรคเบาหวาน</div>', unsafe_allow_html=True)
    st.markdown("กรุณากรอกข้อมูลสุขภาพของคุณด้านล่าง เพื่อประเมินความเสี่ยงเบื้องต้นด้วยโมเดล AI")
    st.markdown("---")

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        st.markdown("##### 👤 ข้อมูลส่วนบุคคล")
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("โรคความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี (0)" if x == 0 else "มี (1)", horizontal=True)
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี (0)" if x == 0 else "มี (1)", horizontal=True)

    with col_input2:
        st.markdown("##### 🩸 ข้อมูลทางการแพทย์")
        smoking_history = st.selectbox("ประวัติการสูบบุหรี่", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("ค่า BMI (ดัชนีมวลกาย)", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("ระดับน้ำตาลเฉลี่ย HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        blood_glucose = st.number_input("ระดับน้ำตาลในเลือด (mg/dL)", 50, 400, 100)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ปุ่มทำนาย
    if st.button("🔮 ประเมินความเสี่ยงทันที"):
        with st.spinner("🔄 กำลังวิเคราะห์ข้อมูล..."):
            # 1. Encode
            gender_enc = metrics['le_gender'].transform([gender])[0]
            smoking_enc = metrics['le_smoking'].transform([smoking_history])[0]
            
            # 2. สร้าง Input Array
            input_data = np.array([[
                gender_enc, age, hypertension, heart_disease,
                smoking_enc, bmi, hba1c, blood_glucose
            ]])
            
            # 3. Scale
            input_scaled = metrics['scaler'].transform(input_data)
            
            # 4. Predict
            prediction = metrics['model'].predict(input_scaled)[0]
            proba = metrics['model'].predict_proba(input_scaled)[0]
            risk_percentage = proba[1] * 100

            st.markdown("---")
            st.markdown('<div class="sub-header">📊 ผลการประเมิน</div>', unsafe_allow_html=True)
            
            # แสดงผลลัพธ์แบบ Visual
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                if prediction == 1:
                    st.error(f"### ⚠️ มีความเสี่ยง")
                    st.markdown(f"**โอกาสเป็นโรคเบาหวาน:** {risk_percentage:.1f}%")
                else:
                    st.success(f"### ✅ ความเสี่ยงต่ำ")
                    st.markdown(f"**โอกาสเป็นโรคเบาหวาน:** {risk_percentage:.1f}%")
            
            with res_col2:
                # Progress bar แสดงความน่าจะเป็น
                st.markdown("##### ระดับความเสี่ยง")
                st.progress(float(risk_percentage / 100))
                
                if prediction == 1:
                    st.warning("💡 **คำแนะนำ:** ค่า HbA1c หรือระดับน้ำตาลของคุณอยู่ในเกณฑ์ที่ควรเฝ้าระวัง ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม และควบคุมอาหาร")
                else:
                    st.info("💡 **คำแนะนำ:** สุขภาพของคุณอยู่ในเกณฑ์ดี! ควรตรวจสุขภาพประจำปีอย่างสม่ำเสมอ และรักษาพฤติกรรมการกินที่ดีต่อไป")

            # Feature Importance แบบง่ายๆ
            st.markdown('<div class="sub-header">🔍 ปัจจัยที่มีผลต่อการตัดสินใจของโมเดล</div>', unsafe_allow_html=True)
            importance = pd.DataFrame({
                'Feature': metrics['feature_names'],
                'Importance': metrics['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
            sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis', ax=ax_imp)
            ax_imp.set_title('Feature Importance (Random Forest)', fontweight='bold')
            ax_imp.set_xlabel('Score')
            ax_imp.set_ylabel('')
            st.pyplot(fig_imp)