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

# ==================== ตั้งค่าหน้าเว็บ ====================
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ขั้นสูง (Modern Health-Tech Theme) ====================
st.markdown("""
<style>
    /* Import ฟอนต์ไทยสวยๆ */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* ตั้งค่าฟอนต์หลัก */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif !important;
    }
    
    /* ซ่อนเมนู Streamlit ด้านขวาบน */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* หัวข้อหลัก */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F172A;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 4px solid #3B82F6;
        letter-spacing: -0.5px;
    }
    
    /* หัวข้อรอง */
    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E40AF;
        margin-top: 2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    
    /* การ์ดเมตริกแบบ Modern Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    .metric-card h3 {
        color: #64748B;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h2 {
        color: #0F172A;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card p {
        color: #3B82F6;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* ปุ่มทำนาย */
    .stButton>button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E3A8A 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }

    /* กล่องข้อมูลผู้พัฒนา (Sidebar) */
    .developer-box {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 16px;
        padding: 20px 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .dev-avatar {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 4px solid #FFFFFF;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
    .dev-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 12px;
    }
    .dev-info {
        background: #FFFFFF;
        border-radius: 8px;
        padding: 10px;
        margin-top: 8px;
        border: 1px solid #E2E8F0;
    }
    .dev-info p {
        margin: 6px 0;
        font-size: 0.85rem;
        color: #475569;
        text-align: left;
        padding-left: 10px;
    }
    .dev-info strong {
        color: #0F172A;
    }

    /* ผลลัพธ์การทำนาย */
    .result-card-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border-left: 6px solid #EF4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
    .result-card-low {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border-left: 6px solid #22C55E;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
    
    /* ปรับแต่ง St dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
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
st.sidebar.markdown("### 🩺 Diabetes AI")
st.sidebar.markdown("---")

# ส่วนข้อมูลผู้พัฒนา
st.sidebar.markdown('<div class="developer-box">', unsafe_allow_html=True)

# รูปโปรไฟล์ (ใช้ URL รูป Avatar สวยๆ เป็น Default หากยังไม่มีไฟล์)
profile_image = "https://ui-avatars.com/api/?name=Phuwadit+Cham&background=3B82F6&color=fff&size=200&font-size=0.4"
# หมายเหตุ: หากมีไฟล์รูป ให้เปลี่ยนเป็น: profile_image = "profile.jpg"

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

# เมนูนำทาง
page = st.sidebar.radio(
    "เลือกเมนู",
    ["🏠 หน้าหลัก", "📊 วิเคราะห์ข้อมูล", "🤖 ประสิทธิภาพโมเดล", "🎮 ทายผลความเสี่ยง"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #64748B; font-size: 0.8rem;'>📅 ปีการศึกษา 2026<br>🏥 Machine Learning Project</div>", unsafe_allow_html=True)

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
    <div style="background: #F8FAFC; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; line-height: 1.8;">
    • พัฒนาโมเดล Machine Learning เพื่อคัดกรองความเสี่ยงโรคเบาหวานล่วงหน้า<br>
    • วิเคราะห์ปัจจัยสำคัญ (Features) ที่ส่งผลต่อการเกิดโรค เช่น ระดับน้ำตาล, BMI, อายุ<br>
    • สร้าง Web Application ที่ใช้งานง่ายและสวยงามสำหรับบุคคลทั่วไปในการประเมินสุขภาพเบื้องต้น
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sub-header">📋 ตัวอย่างข้อมูล (Dataset Preview)</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(5), width="stretch", hide_index=True)

# ==================== PAGE 2: วิเคราะห์ข้อมูล ====================
elif page == "📊 วิเคราะห์ข้อมูล":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">🥧 สัดส่วนการเป็นโรคเบาหวาน</div>', unsafe_allow_html=True)
        counts = df_raw['diabetes'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie(counts, labels=['ไม่เป็น (0)', 'เป็น (1)'], autopct='%1.1f%%', 
                colors=['#22C55E', '#EF4444'], startangle=90, textprops={'color': "white", 'fontweight': 'bold', 'fontsize': 12})
        ax1.set_title('Distribution of Diabetes', fontweight='bold', color='#0F172A')
        st.pyplot(fig1)
        
    with col2:
        st.markdown('<div class="sub-header">📈 อายุ vs ระดับน้ำตาลในเลือด</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        sns.scatterplot(data=df_raw.sample(2000), x='age', y='blood_glucose_level', hue='diabetes', palette=['#22C55E', '#EF4444'], alpha=0.6, ax=ax2)
        ax2.set_title('Age vs Blood Glucose Level', fontweight='bold', color='#0F172A')
        ax2.legend(title='Diabetes', labels=['No', 'Yes'])
        st.pyplot(fig2)

    st.markdown('<div class="sub-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    numeric_df = df_raw.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3, linewidths=0.5, annot_kws={"size": 10})
    ax3.set_title('Feature Correlation Matrix', fontweight='bold', color='#0F172A', pad=15)
    st.pyplot(fig3)

# ==================== PAGE 3: ประสิทธิภาพโมเดล ====================
elif page == "🤖 ประสิทธิภาพโมเดล":
    st.markdown('<div class="main-header">🤖 Random Forest Model Evaluation</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">📚 ทำไมต้อง Random Forest?</div>', unsafe_allow_html=True)
    st.info("""
    **Random Forest** เป็นอัลกอริทึมแบบ Ensemble ที่สร้าง Decision Tree หลายร้อยต้นและนำผลมาโหวตกัน 
    **ข้อดี:** ป้องกัน Overfitting ได้ดี, จัดการกับข้อมูลที่ไม่สมดุล (Imbalanced Data) ได้ยอดเยี่ยม, 
    และสามารถบอกความสำคัญของแต่ละปัจจัย (Feature Importance) ได้อย่างชัดเจน
    """, icon="💡")

    st.markdown('<div class="sub-header">📊 ผลการประเมิน (Evaluation Metrics)</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><h3>Accuracy</h3><h2>{metrics["accuracy"]:.1%}</h2></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><h3>Precision</h3><h2>{metrics["precision"]:.1%}</h2></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><h3>Recall</h3><h2>{metrics["recall"]:.1%}</h2></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><h3>F1-Score</h3><h2>{metrics["f1"]:.1%}</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                    xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], annot_kws={"size": 14})
        ax_cm.set_xlabel('Predicted', fontweight='bold', color='#0F172A')
        ax_cm.set_ylabel('Actual', fontweight='bold', color='#0F172A')
        ax_cm.set_title('Confusion Matrix', fontweight='bold', pad=15)
        st.pyplot(fig_cm)

    with col2:
        st.markdown('<div class="sub-header">📈 ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color='#2563EB', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='#94A3B8', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate', fontweight='bold', color='#0F172A')
        ax_roc.set_ylabel('True Positive Rate', fontweight='bold', color='#0F172A')
        ax_roc.set_title('Receiver Operating Characteristic', fontweight='bold', pad=15)
        ax_roc.legend(loc="lower right", frameon=True, shadow=True)
        ax_roc.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_roc)

# ==================== PAGE 4: ทายผลความเสี่ยง (Prediction) ====================
elif page == "🎮 ทายผลความเสี่ยง":
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยงโรคเบาหวาน</div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #64748B; margin-bottom: 2rem;'>กรุณากรอกข้อมูลสุขภาพของคุณด้านล่าง เพื่อประเมินความเสี่ยงเบื้องต้นด้วยโมเดล AI</div>", unsafe_allow_html=True)

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
        with st.spinner("🔄 กำลังวิเคราะห์ข้อมูลด้วยโมเดล AI..."):
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
            st.markdown('<div class="sub-header">📊 ผลการประเมินสุขภาพ</div>', unsafe_allow_html=True)
            
            # แสดงผลลัพธ์แบบ Visual ที่สวยงาม
            if prediction == 1:
                st.markdown(f'''
                    <div class="result-card-high">
                        <h2 style="color: #B91C1C; margin: 0;">⚠️ มีความเสี่ยงเป็นโรคเบาหวาน</h2>
                        <p style="color: #991B1B; font-size: 1.1rem; margin-top: 0.5rem;">
                            โมเดลประเมินว่าคุณมีโอกาสร้อยละ <strong>{risk_percentage:.1f}%</strong> ที่จะมีความเสี่ยง
                        </p>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.progress(float(risk_percentage / 100))
                st.warning("💡 **คำแนะนำจาก AI:** ค่า HbA1c หรือระดับน้ำตาลของคุณอยู่ในเกณฑ์ที่ควรเฝ้าระวัง ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม ควบคุมอาหารหวานและออกกำลังกายอย่างสม่ำเสมอ")
            else:
                st.markdown(f'''
                    <div class="result-card-low">
                        <h2 style="color: #15803D; margin: 0;">✅ ความเสี่ยงอยู่ในระดับต่ำ</h2>
                        <p style="color: #166534; font-size: 1.1rem; margin-top: 0.5rem;">
                            โมเดลประเมินว่าคุณมีโอกาสร้อยละ <strong>{risk_percentage:.1f}%</strong> ซึ่งอยู่ในเกณฑ์ปลอดภัย
                        </p>
                    </div>
                ''', unsafe_allow_html=True)
                
                st.progress(float(risk_percentage / 100))
                st.info("💡 **คำแนะนำจาก AI:** สุขภาพของคุณอยู่ในเกณฑ์ดี! ควรตรวจสุขภาพประจำปีอย่างสม่ำเสมอ และรักษาพฤติกรรมการกินที่ดีต่อไป")

            # Feature Importance
            st.markdown('<div class="sub-header">🔍 ปัจจัยที่มีผลต่อการตัดสินใจของโมเดล</div>', unsafe_allow_html=True)
            importance = pd.DataFrame({
                'Feature': metrics['feature_names'],
                'Importance': metrics['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
            sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis', ax=ax_imp)
            ax_imp.set_title('Feature Importance (Random Forest)', fontweight='bold', color='#0F172A', pad=10)
            ax_imp.set_xlabel('Importance Score', fontweight='bold')
            ax_imp.set_ylabel('')
            ax_imp.grid(axis='x', linestyle=':', alpha=0.6)
            st.pyplot(fig_imp)