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

st.set_page_config(page_title="Diabetes AI", page_icon="️", layout="wide", initial_sidebar_state="collapsed")

# ==================== DARK MODE CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Prompt:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', 'Prompt', sans-serif !important; }
    
    /* พื้นหลังหลัก - Dark Gradient */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%);
        min-height: 100vh;
    }
    
    /* ซ่อนเมนู Streamlit */
    #MainMenu, footer, header { visibility: hidden !important; }
    
    /* Hero Header แบบ Neon */
    .hero-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(56, 189, 248, 0.1), inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #22d3ee 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Navigation Pills */
    .nav-pills {
        display: flex;
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.15);
        margin-bottom: 2rem;
    }
    
    .nav-pill {
        flex: 1;
        padding: 1rem;
        text-align: center;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #64748b;
    }
    
    .nav-pill.active {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.4);
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 20px 40px rgba(14, 165, 233, 0.2);
    }
    
    .glass-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .glass-card:hover::after {
        left: 100%;
    }
    
    /* Stat Cards แบบ Neon */
    .neon-stat {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .neon-stat::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #22d3ee, #a78bfa);
    }
    
    .neon-stat h3 {
        color: #38bdf8;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
    }
    
    .neon-stat p {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.5rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Profile Card แบบ Premium */
    .profile-premium {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(167, 139, 250, 0.1) 100%);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        position: relative;
    }
    
    .profile-premium::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(135deg, #38bdf8, #a78bfa);
        border-radius: 24px;
        z-index: -1;
        opacity: 0.3;
        filter: blur(10px);
    }
    
    .profile-avatar-glow {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        border: 3px solid transparent;
        background: linear-gradient(135deg, #38bdf8, #a78bfa) border-box;
        -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        padding: 3px;
        margin: 0 auto 1rem;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.5);
    }
    
    /* Section Titles */
    .section-title-dark {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 2rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 4px solid #38bdf8;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
        color: white !important;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(14, 165, 233, 0.6);
    }
    
    /* Input Fields - Dark Mode */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(15, 23, 42, 0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 10px !important;
        padding: 0.8rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Result Cards */
    .result-danger-dark {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 20px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .result-danger-dark::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ef4444, #f97316);
    }
    
    .result-success-dark {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 20px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .result-success-dark::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #22c55e, #10b981);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #06b6d4, #a78bfa) !important;
        border-radius: 10px !important;
        height: 10px !important;
    }
    
    /* Streamlit default overrides */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95) !important;
        backdrop-filter: blur(20px);
    }
    
    .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load Data & Model ====================
@st.cache_data
def load_data():
    return pd.read_csv('diabetes_prediction_dataset.csv')

@st.cache_resource
def build_model():
    df = load_data()
    le_g = LabelEncoder()
    le_s = LabelEncoder()
    df['gender'] = le_g.fit_transform(df['gender'])
    df['smoking_history'] = le_s.fit_transform(df['smoking_history'])
    
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    
    return {
        'acc': accuracy_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred),
        'rec': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
        'model': model, 'scaler': scaler,
        'le_g': le_g, 'le_s': le_s,
        'features': X.columns.tolist()
    }

with st.spinner("⚡ Initializing AI System..."):
    m = build_model()
    df = load_data()

# ==================== Hero Header ====================
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">⚕️ DIABETES PREDICTION AI</h1>
    <p class="hero-subtitle">Advanced Machine Learning System for Early Diabetes Detection</p>
</div>
""", unsafe_allow_html=True)

# ==================== Navigation Pills ====================
tabs = ["🏠 Overview", "📊 Analytics", "🤖 Model", "🎯 Predict"]
selected = st.radio("Menu", tabs, horizontal=True, label_visibility="collapsed")

# ==================== PAGE 1: OVERVIEW ====================
if selected == "🏠 Overview":
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="profile-premium">
            <img src="https://ui-avatars.com/api/?name=Phuwadit+Cham&background=38bdf8&color=fff&size=200&bold=true" 
                 style="width:120px; height:120px; border-radius:50%; margin-bottom:1rem;">
            <h3 style="color:#e2e8f0; margin:0.5rem 0;">Phuwadit Chammonkhong</h3>
            <p style="color:#94a3b8; font-size:0.9rem;">Machine Learning Developer</p>
            <div style="background:rgba(15,23,42,0.6); border-radius:12px; padding:1rem; margin-top:1rem; text-align:left;">
                <p style="color:#38bdf8; margin:0.4rem 0; font-size:0.85rem;">🆔 <strong style="color:#e2e8f0;">664245031</strong></p>
                <p style="color:#38bdf8; margin:0.4rem 0; font-size:0.85rem;">📚 <strong style="color:#e2e8f0;">Class 66/44</strong></p>
                <p style="color:#38bdf8; margin:0.4rem 0; font-size:0.85rem;">📅 <strong style="color:#e2e8f0;">Academic Year 2026</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-title-dark">📈 Dataset Statistics</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="neon-stat"><h3>{len(df):,}</h3><p>Total Records</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="neon-stat"><h3>8</h3><p>Features</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="neon-stat"><h3>{df["diabetes"].sum()}</h3><p>Diabetes Cases</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="neon-stat"><h3>{m["acc"]:.1%}</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;"> Project Objectives</h3><ul style="color:#cbd5e1; line-height:1.8;"><li>Develop ML model for early diabetes detection</li><li>Analyze key risk factors (glucose, BMI, age, HbA1c)</li><li>Build user-friendly prediction interface</li><li>Achieve high accuracy with Random Forest</li></ul></div>', unsafe_allow_html=True)

# ==================== PAGE 2: ANALYTICS ====================
elif selected == "📊 Analytics":
    st.markdown('<div class="section-title-dark">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;"> Class Distribution</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        counts = df['diabetes'].value_counts()
        wedges, texts, autotexts = ax.pie(counts, labels=['Non-Diabetic', 'Diabetic'], 
                autopct='%1.1f%%', colors=['#22c55e', '#ef4444'], startangle=90,
                textprops={'color': 'white', 'fontweight': 'bold'})
        ax.set_title('Diabetes Distribution', color='#e2e8f0', fontweight='bold', fontsize=14)
        fig.patch.set_facecolor('#0f172a')
        st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;"> Age vs Glucose</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        sample = df.sample(2000)
        ax.scatter(sample[sample['diabetes']==0]['age'], sample[sample['diabetes']==0]['blood_glucose_level'], 
                   c='#22c55e', alpha=0.5, label='Non-Diabetic', s=30)
        ax.scatter(sample[sample['diabetes']==1]['age'], sample[sample['diabetes']==1]['blood_glucose_level'], 
                   c='#ef4444', alpha=0.7, label='Diabetic', s=30)
        ax.set_xlabel('Age', color='#94a3b8')
        ax.set_ylabel('Blood Glucose', color='#94a3b8')
        ax.set_title('Age vs Blood Glucose', color='#e2e8f0', fontweight='bold')
        ax.legend()
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        st.pyplot(fig)
    
    st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;">🔥 Correlation Heatmap</h3></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, 
                annot_kws={'color': 'white', 'size': 9})
    ax.set_title('Feature Correlation', color='#e2e8f0', fontweight='bold', pad=15)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    plt.xticks(color='#94a3b8')
    plt.yticks(color='#94a3b8')
    st.pyplot(fig)

# ==================== PAGE 3: MODEL ====================
elif selected == "🤖 Model":
    st.markdown('<div class="section-title-dark">🤖 Random Forest Performance</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;">📚 Algorithm Overview</h3><p style="color:#cbd5e1; line-height:1.8;">Random Forest is an ensemble method that builds multiple decision trees and merges them for more accurate predictions. It prevents overfitting through bootstrap sampling and feature randomness, making it ideal for medical prediction tasks.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title-dark">📊 Performance Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="neon-stat"><h3>{m["acc"]:.1%}</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="neon-stat"><h3>{m["prec"]:.1%}</h3><p>Precision</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="neon-stat"><h3>{m["rec"]:.1%}</h3><p>Recall</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="neon-stat"><h3>{m["f1"]:.2f}</h3><p>F1-Score</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;">🎯 Confusion Matrix</h3></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = confusion_matrix(m['y_test'], m['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    annot_kws={'color': 'white', 'size': 14, 'fontweight': 'bold'})
        ax.set_title('Confusion Matrix', color='#e2e8f0', fontweight='bold', pad=15)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        plt.xticks(color='#94a3b8')
        plt.yticks(color='#94a3b8')
        st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;">📈 ROC Curve</h3></div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(m['y_test'], m['y_proba'])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='#38bdf8', lw=3, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='#64748b', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate', color='#94a3b8')
        ax.set_ylabel('True Positive Rate', color='#94a3b8')
        ax.set_title('ROC Curve', color='#e2e8f0', fontweight='bold', pad=15)
        ax.legend(loc='lower right')
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        st.pyplot(fig)

# ==================== PAGE 4: PREDICT ====================
elif selected == "🎯 Predict":
    st.markdown('<div class="section-title-dark">🎯 Risk Assessment Form</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;">👤 Personal Info</h3></div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Age (years)", 0, 100, 30)
        hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
        heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    
    with col2:
        st.markdown('<div class="glass-card"><h3 style="color:#38bdf8; margin-top:0;">🩺 Medical Data</h3></div>', unsafe_allow_html=True)
        smoking = st.selectbox("Smoking History", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("Blood Glucose (mg/dL)", 50, 400, 100)
    
    if st.button("⚡ Analyze Risk"):
        g_enc = m['le_g'].transform([gender])[0]
        s_enc = m['le_s'].transform([smoking])[0]
        
        input_data = np.array([[g_enc, age, hypertension, heart_disease, s_enc, bmi, hba1c, glucose]])
        input_scaled = m['scaler'].transform(input_data)
        
        pred = m['model'].predict(input_scaled)[0]
        proba = m['model'].predict_proba(input_scaled)[0]
        risk = proba[1] * 100
        
        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div class="result-danger-dark">
                <h2 style="color:#ef4444; margin:0;">⚠️ High Risk Detected</h2>
                <p style="font-size:1.5rem; color:#fca5a5; margin:1rem 0;"><strong>Diabetes Probability: {risk:.1f}%</strong></p>
                <p style="color:#cbd5e1;">🏥 Consult a doctor immediately for further diagnosis</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-success-dark">
                <h2 style="color:#22c55e; margin:0;">✅ Low Risk</h2>
                <p style="font-size:1.5rem; color:#86efac; margin:1rem 0;"><strong>Diabetes Probability: {risk:.1f}%</strong></p>
                <p style="color:#cbd5e1;">💡 Maintain healthy lifestyle and regular checkups</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div style="color:#94a3b8; margin-top:1rem;">Risk Level</div>', unsafe_allow_html=True)
        st.progress(float(risk / 100))
        
        # Feature Importance
        st.markdown('<div class="section-title-dark">🔍 Feature Importance</div>', unsafe_allow_html=True)
        importance = pd.DataFrame({
            'Feature': m['features'],
            'Importance': m['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#38bdf8' if i < 3 else '#64748b' for i in range(len(importance))]
        sns.barplot(data=importance, x='Importance', y='Feature', palette=colors, ax=ax)
        ax.set_title('Top Predictors', color='#e2e8f0', fontweight='bold')
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        plt.xticks(color='#94a3b8')
        plt.yticks(color='#e2e8f0')
        st.pyplot(fig)

# Footer
st.markdown("""
<div style="text-align:center; padding:2rem; color:#64748b; border-top:1px solid rgba(56,189,248,0.1); margin-top:3rem;">
    <p>⚕️ Diabetes Prediction AI System | Built with Streamlit & Machine Learning</p>
    <p style="font-size:0.85rem;">© 2026 Phuwadit Chammonkhong</p>
</div>
""", unsafe_allow_html=True)