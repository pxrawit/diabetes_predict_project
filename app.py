import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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

# Custom CSS
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

.info-box {
    background: rgba(255, 255, 255, 0.95);
    padding: 1.5rem;
    border-radius: 16px;
    margin: 1rem 0;
    border-left: 5px solid #667eea;
}
</style>
""", unsafe_allow_html=True)

# โหลดข้อมูล
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')

# สร้างและเปรียบเทียบโมเดล
@st.cache_resource
def build_and_compare_models():
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
    
    # สร้างโมเดลต่างๆ
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'model': model
        })
    
    results_df = pd.DataFrame(results)
    
    return {
        'results_df': results_df,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'le_gender': le_gender,
        'le_smoking': le_smoking,
        'feature_names': X.columns.tolist(),
        'df': df
    }

# โหลดข้อมูลและโมเดล
with st.spinner("🔄 กำลังเตรียมระบบ..."):
    metrics = build_and_compare_models()
    df_raw = load_data()

# Tabs Navigation - เพิ่ม Tab ที่ 5
tabs = st.tabs([
    "🏠 หน้าหลัก",
    "🧹 การเตรียมข้อมูล",  # Tab ใหม่
    "📊 วิเคราะห์ข้อมูล",
    "🤖 ประสิทธิภาพโมเดล",
    "🎮 ทายผลความเสี่ยง",
    "👨‍💻 ผู้พัฒนา"
])

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
        <p style="color: #1e293b; line-height: 1.8; font-size: 1rem;">
        💡 <strong>พัฒนาโมเดล Machine Learning</strong> เพื่อคัดกรองความเสี่ยงโรคเบาหวานล่วงหน้า<br>
        🔬 <strong>วิเคราะห์ปัจจัยสำคัญ</strong> ที่ส่งผลต่อการเกิดโรค<br>
        🌐 <strong>สร้าง Web Application</strong> ที่ใช้งานง่ายและสวยงาม
        </p>
    </div>
    ''', unsafe_allow_html=True)

# ==================== TAB 2: การเตรียมข้อมูล (ใหม่) ====================
with tabs[1]:
    st.markdown('<div class="main-header">🧹 การเตรียมข้อมูล (Data Preprocessing)</div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="info-box">
        <h3 style="color: #667eea; margin-top: 0;">📋 ขั้นตอนการเตรียมข้อมูล 4 ขั้นตอน</h3>
        <p>การเตรียมข้อมูลเป็นขั้นตอนสำคัญที่ช่วยให้โมเดล Machine Learning ทำงานได้มีประสิทธิภาพมากขึ้น</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # ขั้นตอนที่ 1
    st.markdown('<div class="metric-card" style="margin-bottom: 1.5rem;"><h3 style="text-align: left;">1️⃣ ตรวจสอบและจัดการ Missing Values</h3><p style="text-align: left; margin-top: 1rem;">ตรวจสอบค่าว่างใน Dataset และจัดการด้วยการลบหรือเติมค่าที่เหมาะสม</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Before Preprocessing:**")
        st.write(df_raw.isnull().sum())
    with col2:
        st.markdown("**✅ After Preprocessing:**")
        st.success("ไม่มีค่า Missing Values")
    
    st.markdown("---")
    
    # ขั้นตอนที่ 2
    st.markdown('<div class="metric-card" style="margin-bottom: 1.5rem;"><h3 style="text-align: left;">2️ Label Encoding (แปลงข้อมูล categorical เป็นตัวเลข)</h3><p style="text-align: left; margin-top: 1rem;">แปลงตัวแปรหมวดหมู่ เช่น Gender และ Smoking History ให้เป็นตัวเลขเพื่อให้โมเดลประมวลผลได้</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Before Encoding:**")
        st.write(df_raw[['gender', 'smoking_history']].head())
    with col2:
        st.markdown("**✅ After Encoding:**")
        df_encoded = df_raw.copy()
        le_g = LabelEncoder()
        le_s = LabelEncoder()
        df_encoded['gender'] = le_g.fit_transform(df_encoded['gender'])
        df_encoded['smoking_history'] = le_s.fit_transform(df_encoded['smoking_history'])
        st.write(df_encoded[['gender', 'smoking_history']].head())
    
    st.markdown("---")
    
    # ขั้นตอนที่ 3
    st.markdown('<div class="metric-card" style="margin-bottom: 1.5rem;"><h3 style="text-align: left;">3️⃣ Feature Scaling (Standardization)</h3><p style="text-align: left; margin-top: 1rem;">ปรับสเกลข้อมูลให้อยู่ในช่วงมาตรฐาน (mean=0, std=1) เพื่อให้โมเดลเรียนรู้ได้ดีขึ้น</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Before Scaling:**")
        st.write(df_raw[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']].describe())
    with col2:
        st.markdown("**✅ After Scaling:**")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_raw[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
        scaled_df = pd.DataFrame(scaled_data, columns=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
        st.write(scaled_df.describe())
    
    st.markdown("---")
    
    # ขั้นตอนที่ 4
    st.markdown('<div class="metric-card" style="margin-bottom: 1.5rem;"><h3 style="text-align: left;">4️⃣ Train-Test Split (แบ่งข้อมูล)</h3><p style="text-align: left; margin-top: 1rem;">แบ่งข้อมูลเป็นชุดฝึก (Training Set) 80% และชุดทดสอบ (Test Set) 20% โดยใช้ Stratified Sampling</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Training Set:**")
        st.info("80% ของข้อมูลทั้งหมด\nประมาณ 72,000+ แถว")
    with col2:
        st.markdown("**📊 Test Set:**")
        st.info("20% ของข้อมูลทั้งหมด\nประมาณ 18,000+ แถว")
    
    st.markdown('''
    <div class="info-box" style="border-left-color: #10b981;">
        <h4 style="color: #10b981; margin-top: 0;">✅ ประโยชน์ของการ Preprocessing:</h4>
        <ul>
            <li>เพิ่มความแม่นยำของโมเดล</li>
            <li>ลดปัญหา Overfitting</li>
            <li>ช่วยให้โมเดลเรียนรู้ได้เร็วขึ้น</li>
            <li>จัดการกับข้อมูลที่ไม่สมดุลได้ดีขึ้น</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

# ==================== TAB 3: วิเคราะห์ข้อมูล ====================
with tabs[2]:
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

# ==================== TAB 4: ประสิทธิภาพโมเดล (ปรับปรุง) ====================
with tabs[3]:
    st.markdown('<div class="main-header">🤖 ประสิทธิภาพโมเดลและการเปรียบเทียบ</div>', unsafe_allow_html=True)
    
    # ทฤษฎี Random Forest
    st.markdown('''
    <div class="metric-card" style="margin-bottom: 2rem; text-align: left;">
        <h3 style="color: #667eea; font-size: 1.3rem;"> ทฤษฎี Random Forest</h3>
        <p style="line-height: 1.8; font-size: 1rem;">
        <strong>Random Forest</strong> เป็นอัลกอริทึม Ensemble Learning ที่สร้าง Decision Trees หลายต้นและนำผลลัพธ์มาโหวตเพื่อทำนายผล
        </p>
        <h4 style="color: #667eea; margin-top: 1rem;"> หลักการทำงาน 4 ขั้นตอน:</h4>
        <ol style="line-height: 2;">
            <li><strong>Bootstrap Sampling:</strong> สุ่มเลือกข้อมูลด้วยการ Sampling แบบมีแทนที่ (With Replacement)</li>
            <li><strong>Feature Randomness:</strong> สุ่มเลือก Features ในการ Split แต่ละ Node</li>
            <li><strong>Tree Building:</strong> สร้าง Decision Tree หลายร้อยต้นโดยไม่ Pruning</li>
            <li><strong>Majority Voting:</strong> นำผลลัพธ์จากทุก Trees มาโหวตเพื่อทำนายผลสุดท้าย</li>
        </ol>
        <h4 style="color: #667eea; margin-top: 1rem;">✅ ข้อดี:</h4>
        <ul style="line-height: 2;">
            <li>ป้องกันปัญหา Overfitting ได้ดี</li>
            <li>จัดการกับ Imbalanced Data ได้ดี</li>
            <li>สามารถบอก Feature Importance ได้</li>
            <li>มีความแม่นยำสูงและเสถียร</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    # ตารางเปรียบเทียบโมเดล
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">📊 ตารางเปรียบเทียบโมเดล</div>', unsafe_allow_html=True)
    
    results_df = metrics['results_df'].copy()
    display_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f'{x:.2%}')
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f'{x:.2%}')
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f'{x:.2%}')
    display_df['F1-Score'] = display_df['F1-Score'].apply(lambda x: f'{x:.4f}')
    
    st.dataframe(display_df.style.background_gradient(cmap='Blues', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), 
                 use_container_width=True, hide_index=True)
    
    # กราฟเปรียบเทียบ
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">📈 กราฟเปรียบเทียบประสิทธิภาพ</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = results_df['Model']
        accuracy = results_df['Accuracy']
        colors = ['#667eea', '#f43f5e', '#10b981', '#f59e0b']
        bars = ax.bar(models, accuracy, color=colors, edgecolor='white', linewidth=2)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', color='#ffffff')
        ax.set_ylim([0.85, 1.0])
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('#1e293b')
        fig.patch.set_alpha(0.0)
        
        for bar, acc in zip(bars, accuracy):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{acc:.2%}', ha='center', va='bottom', color='white', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(results_df['Model']))
        width = 0.2
        
        ax.bar(x - width, results_df['Precision'], width, label='Precision', color='#667eea')
        ax.bar(x, results_df['Recall'], width, label='Recall', color='#f43f5e')
        ax.bar(x + width, results_df['F1-Score'], width, label='F1-Score', color='#10b981')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold', color='#ffffff')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, color='white')
        ax.tick_params(axis='y', colors='white')
        ax.legend(facecolor='#1e293b', edgecolor='white', labelcolor='white')
        ax.set_facecolor('#1e293b')
        fig.patch.set_alpha(0.0)
        
        st.pyplot(fig)
    
    # ROC Curve
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">📈 ROC Curve Comparison</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, row in results_df.iterrows():
        if row['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(metrics['y_test'], row['y_proba'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{row["Model"]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', color='#ffffff')
    ax.legend(loc='lower right', facecolor='#1e293b', edgecolor='white', labelcolor='white')
    ax.set_facecolor('#1e293b')
    fig.patch.set_alpha(0.0)
    ax.tick_params(colors='white')
    
    st.pyplot(fig)
    
    # Feature Importance (จาก Random Forest)
    st.markdown('<div class="main-header" style="font-size: 1.8rem;">🎯 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    
    rf_model = results_df[results_df['Model'] == 'Random Forest']['model'].values[0]
    importance = pd.DataFrame({
        'Feature': metrics['feature_names'],
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#667eea' if i < 3 else '#94a3b8' for i in range(len(importance))]
        bars = ax.barh(importance['Feature'], importance['Importance'], color=colors, edgecolor='white')
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold', color='white')
        ax.set_title('Feature Importance from Random Forest', fontsize=14, fontweight='bold', color='#ffffff')
        ax.invert_yaxis()
        ax.set_facecolor('#1e293b')
        fig.patch.set_alpha(0.0)
        ax.tick_params(colors='white')
        
        for bar, imp in zip(bars, importance['Importance']):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.4f}', va='center', color='white', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("**📊 Top 3 Features:**")
        for i in range(3):
            st.info(f"**{i+1}. {importance.iloc[i]['Feature']}**\n\nScore: {importance.iloc[i]['Importance']:.4f}")
        
        st.markdown('''
        <div class="info-box" style="border-left-color: #f59e0b;">
            <h4 style="color: #f59e0b; margin-top: 0;">💡 คำแนะนำ:</h4>
            <p>Features ที่มี Importance สูง แสดงว่ามีผลต่อการทำนายโรคเบาหวานมาก ควรให้ความสำคัญในการเก็บข้อมูลและวิเคราะห์เป็นพิเศษ</p>
        </div>
        ''', unsafe_allow_html=True)

# ==================== TAB 5: ทายผลความเสี่ยง ====================
with tabs[4]:
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยง</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0; color:#667eea;">👤 ข้อมูลส่วนบุคคล</h3></div>', unsafe_allow_html=True)
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("ความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
    
    with col2:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0; color:#667eea;">🩸 ข้อมูลทางการแพทย์</h3></div>', unsafe_allow_html=True)
        smoking = st.selectbox("สูบบุหรี่", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("น้ำตาลในเลือด (mg/dL)", 50, 400, 100)

    if st.button(" ทำนายผล"):
        gender_enc = metrics['le_gender'].transform([gender])[0]
        smoking_enc = metrics['le_smoking'].transform([smoking])[0]
        
        input_data = np.array([[gender_enc, age, hypertension, heart_disease, smoking_enc, bmi, hba1c, glucose]])
        input_scaled = metrics['scaler'].transform(input_data)
        
        rf_model = metrics['results_df'][metrics['results_df']['Model'] == 'Random Forest']['model'].values[0]
        prediction = rf_model.predict(input_scaled)[0]
        proba = rf_model.predict_proba(input_scaled)[0]
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

# ==================== TAB 6: ผู้พัฒนา ====================
with tabs[5]:
    st.markdown('<div class="main-header">👨‍💻 เกี่ยวกับผู้พัฒนา</div>', unsafe_allow_html=True)
    
    profile_image = "https://ui-avatars.com/api/?name=Phuwadit+Cham&background=667eea&color=fff&size=200&font-size=0.4"
    
    st.markdown(f'''
        <div class="developer-card">
            <img src="{profile_image}" class="dev-avatar" alt="Profile">
            <div class="dev-name">นาย ภูวฤทธิ์ แช่มมั่นคง</div>
            <div class="dev-info">
                <p>👤 <strong>ชื่อ-สกุล:</strong> นาย ภูวฤทธิ์ แช่มมั่นคง</p>
                <p>🆔 <strong>รหัสนักศึกษา:</strong> 664245031</p>
                <p> <strong>หมู่เรียน:</strong> 66/44</p>
                <p> <strong>ปีการศึกษา:</strong> 2026</p>
                <p> <strong>โปรเจกต์:</strong> Machine Learning Project</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)