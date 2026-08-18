import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #2ca02c;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-weight: bold;
    border-bottom: 2px solid #2ca02c;
    padding-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ==================== โหลดและเตรียมข้อมูล ====================
@st.cache_data
def load_and_prepare_data():
    """โหลดข้อมูลและ preprocessing (cache ไว้)"""
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    # Encode categorical
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
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_gender, le_smoking

# ==================== เทรนโมเดล ====================
@st.cache_resource
def train_model(X_train, y_train):
    """เทรน Random Forest (cache ไว้)"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

# โหลดข้อมูลและเทรนโมเดล
with st.spinner("⏳ กำลังโหลดข้อมูลและเทรนโมเดล..."):
    df, X_train, X_test, y_train, y_test, scaler, le_gender, le_smoking = load_and_prepare_data()
    model = train_model(X_train, y_train)

feature_names = ['gender', 'age', 'hypertension', 'heart_disease',
                 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Sidebar Navigation
st.sidebar.title(" Navigation")
page = st.sidebar.radio(
    "เลือกหน้า",
    ["🏠 Home", "📊 Dataset Info", "🤖 Model Info", "🎮 Prediction"]
)

# ==================== HOME PAGE ====================
if page == " Home":
    st.markdown('<h1 class="main-header">🏥 Diabetes Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📋 Project:** Diabetes Prediction")
    with col2:
        st.success("**🤖 Model:** Random Forest")
    with col3:
        st.warning("**📅 Year:** 2026")
    
    st.markdown('<h2 class="sub-header"> Project Objectives</h2>', unsafe_allow_html=True)
    objectives = [
        "พัฒนาโมเดล Machine Learning สำหรับพยากรณ์โรคเบาหวาน",
        "ใช้ Random Forest ซึ่งเป็น Ensemble Learning ที่มีความแม่นยำสูง",
        "วิเคราะห์ปัจจัยเสี่ยงที่สำคัญต่อการเกิดโรคเบาหวาน",
        "สร้าง Web Application สำหรับทำนายความเสี่ยงแบบ Real-time"
    ]
    for i, obj in enumerate(objectives, 1):
        st.markdown(f"**{i}.** {obj}")
    
    st.markdown('<h2 class="sub-header">📊 Dataset Summary</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns)-1)
    with col3:
        st.metric("Diabetes Cases", int(df['diabetes'].sum()))
    with col4:
        st.metric("Non-Diabetes", int(len(df) - df['diabetes'].sum()))
    
    st.markdown('<h2 class="sub-header">🔬 Features</h2>', unsafe_allow_html=True)
    features_desc = {
        "gender": "เพศ (Female/Male/Other)",
        "age": "อายุ (ปี)",
        "hypertension": "ความดันโลหิตสูง (0/1)",
        "heart_disease": "โรคหัวใจ (0/1)",
        "smoking_history": "ประวัติสูบบุหรี่",
        "bmi": "ดัชนีมวลกาย",
        "HbA1c_level": "ระดับน้ำตาลเฉลี่ย 3 เดือน (%)",
        "blood_glucose_level": "ระดับน้ำตาลในเลือด (mg/dL)"
    }
    for feat, desc in features_desc.items():
        st.markdown(f"- **{feat}**: {desc}")

# ==================== DATASET INFO ====================
elif page == "📊 Dataset Info":
    st.markdown('<h1 class="main-header">📊 Dataset Information</h1>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">📄 Data Sample</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(10), width='stretch')
    
    st.markdown('<h2 class="sub-header">📈 Data Statistics</h2>', unsafe_allow_html=True)
    st.dataframe(df.describe(), width='stretch')
    
    st.markdown('<h2 class="sub-header">🔍 Missing Values</h2>', unsafe_allow_html=True)
    missing = df.isnull().sum()
    st.dataframe(missing, width='stretch')
    
    st.markdown('<h2 class="sub-header"> Target Distribution</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        counts = df['diabetes'].value_counts()
        ax.pie(counts.values, labels=['No Diabetes', 'Diabetes'],
               autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        ax.set_title('Diabetes Distribution')
        st.pyplot(fig)
    with col2:
        st.write(df['diabetes'].value_counts())

# ==================== MODEL INFO ====================
elif page == "🤖 Model Info":
    st.markdown('<h1 class="main-header">🤖 Random Forest Model</h1>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">📚 ทฤษฎี Random Forest</h2>', unsafe_allow_html=True)
    st.markdown("""
    **Random Forest** เป็น Ensemble Learning Method ที่สร้าง Decision Trees จำนวนมาก
    และนำผลลัพธ์มาโหวต (Voting) เพื่อทำนายผล
    
    ### หลักการทำงาน:
    1. **Bootstrap Sampling**: สุ่มเลือกข้อมูลด้วย Replacement
    2. **Feature Randomness**: สุ่มเลือก features ในการ split แต่ละ node
    3. **Tree Building**: สร้าง Decision Tree หลายต้น
    4. **Majority Voting**: นำผลลัพธ์มาโหวตเพื่อตัดสินใจ
    
    ### ข้อดี:
    - ✅ ลดปัญหา Overfitting
    - ✅ ทำงานได้รวดเร็ว
    - ✅ บอก Feature Importance ได้
    - ✅ รองรับข้อมูลทั้ง Numerical และ Categorical
    """)
    
    st.markdown('<h2 class="sub-header">⚙️ Model Parameters</h2>', unsafe_allow_html=True)
    params = {
        "n_estimators": "200 (จำนวน Decision Trees)",
        "max_depth": "15 (ความลึกสูงสุด)",
        "min_samples_split": "5",
        "min_samples_leaf": "2",
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": "42"
    }
    for param, desc in params.items():
        st.markdown(f"- **{param}**: {desc}")
    
    # คำนวณ metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.markdown('<h2 class="sub-header">📊 Model Performance</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{acc:.4f}")
    with col2:
        st.metric("Precision", f"{prec:.4f}")
    with col3:
        st.metric("Recall", f"{rec:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    
    st.markdown('<h2 class="sub-header"> Confusion Matrix</h2>', unsafe_allow_html=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    
    st.markdown('<h2 class="sub-header">📈 ROC Curve</h2>', unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    st.markdown('<h2 class="sub-header">📝 Classification Report</h2>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred,
                                   target_names=['No Diabetes', 'Diabetes'],
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format('{:.4f}'), width='stretch')

# ==================== PREDICTION PAGE ====================
elif page == " Prediction":
    st.markdown('<h1 class="main-header">🎮 Diabetes Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">📋 กรอกข้อมูลผู้ป่วย</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender (เพศ)", ["Female", "Male", "Other"])
        age = st.slider("Age (อายุ)", 0, 100, 30)
        hypertension = st.selectbox("Hypertension (ความดันโลหิตสูง)", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")
        heart_disease = st.selectbox("Heart Disease (โรคหัวใจ)", [0, 1],
                                     format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col2:
        smoking_history = st.selectbox("Smoking History",
            ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI (ดัชนีมวลกาย)", 10.0, 70.0, 25.0, 0.1)
        hba1c = st.number_input("HbA1c Level (%)", 3.0, 15.0, 5.7, 0.1)
        blood_glucose = st.number_input("Blood Glucose Level (mg/dL)", 50, 400, 100)
    
    if st.button("🔮 ทำนายผล", type="primary", width='stretch'):
        # Encode inputs
        gender_enc = le_gender.transform([gender])[0]
        smoking_enc = le_smoking.transform([smoking_history])[0]
        
        # Create input array
        input_data = np.array([[
            gender_enc, age, hypertension, heart_disease,
            smoking_enc, bmi, hba1c, blood_glucose
        ]])
        
        # Scale
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 ผลลัพธ์การพยากรณ์</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("No Diabetes", f"{proba[0]*100:.2f}%")
        with col2:
            st.metric("Diabetes", f"{proba[1]*100:.2f}%")
        
        if prediction == 1:
            st.error(f"⚠️ **มีความเสี่ยงเป็นโรคเบาหวาน** (ความมั่นใจ {proba[1]*100:.2f}%)")
            st.warning("💡 แนะนำ: ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม")
        else:
            st.success(f"✅ **ไม่มีความเสี่ยงเป็นโรคเบาหวาน** (ความมั่นใจ {proba[0]*100:.2f}%)")
            st.info("💡 แนะนำ: ควรตรวจสุขภาพเป็นประจำและรักษาสุขภาพ")
        
        # Feature Importance
        st.markdown('<h2 class="sub-header">🔍 Feature Importance</h2>', unsafe_allow_html=True)
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🏥 Diabetes Prediction System\nBuilt with Streamlit ❤️")