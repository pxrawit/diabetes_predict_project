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
from sklearn.neighbors import KNeighborsClassifier
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
def load_css(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("style.css")

# โหลดและเตรียมข้อมูล
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
    
    # สร้างหลายโมเดลเพื่อเปรียบเทียบ
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    best_model = None
    best_acc = 0
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        }
        
        if acc > best_acc:
            best_acc = acc
            best_model = name
    
    return {
        'results': results,
        'best_model': best_model,
        'best_acc': best_acc,
        'scaler': scaler,
        'le_gender': le_gender,
        'le_smoking': le_smoking,
        'feature_names': X.columns.tolist(),
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }

with st.spinner("🔄 กำลังเตรียมระบบ..."):
    metrics = build_model()
    df_raw = load_data()

# TABS NAVIGATION (เพิ่มให้ครบ 7 tabs)
tabs = st.tabs([
    "🏠 หน้าหลัก",
    " Data Preprocessing",
    " ทฤษฎีโมเดล ML",
    " วิเคราะห์ข้อมูล",
    " ประสิทธิภาพโมเดล",
    "🎮 ทายผลความเสี่ยง",
    "👨💻 ผู้พัฒนา"
])

# ==================== TAB 1: หน้าหลัก ====================
with tabs[0]:
    st.markdown('<div class="main-header">🩺 ระบบพยากรณ์โรคเบาหวานด้วย AI</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3> ข้อมูล</h3><h2>100,000+</h2><p>แถวข้อมูลทางการแพทย์</p></div>', unsafe_allow_html=True)
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

# ==================== TAB 2: Data Preprocessing ====================
with tabs[1]:
    st.markdown('<div class="main-header">🧹 Data Preprocessing</div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="metric-card" style="text-align: left; margin-bottom: 1.5rem;">
        <h3 style="color: #67e8f9; margin-top: 0;">📋 ขั้นตอนการเตรียมข้อมูล</h3>
        <p style="color: #cbd5e1; line-height: 1.8;">
        การเตรียมข้อมูล (Data Preprocessing) เป็นขั้นตอนสำคัญก่อนนำข้อมูลไปฝึกโมเดล Machine Learning โดยมีขั้นตอนดังนี้:
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="metric-card" style="text-align: left;">
            <h3 style="color: #67e8f9; margin-top: 0;">1️⃣ ตรวจสอบ Missing Values</h3>
            <p style="color: #cbd5e1;">ตรวจสอบและจัดการค่าว่างใน Dataset เพื่อให้ข้อมูลสมบูรณ์</p>
            
            <h3 style="color: #67e8f9;">2️⃣ Label Encoding</h3>
            <p style="color: #cbd5e1;">แปลงตัวแปรหมวดหมู่ (Categorical) ให้เป็นตัวเลข:</p>
            <ul style="color: #cbd5e1; padding-left: 1.5rem;">
                <li><strong>Gender:</strong> Female → 0, Male → 1, Other → 2</li>
                <li><strong>Smoking History:</strong> never → 0, former → 1, current → 2, ...</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="metric-card" style="text-align: left;">
            <h3 style="color: #67e8f9; margin-top: 0;">3️⃣ Feature Scaling (StandardScaler)</h3>
            <p style="color: #cbd5e1;">ปรับสเกลข้อมูลให้อยู่ในช่วงมาตรฐาน (mean=0, std=1) ช่วยให้โมเดลเรียนรู้ได้ดีขึ้น</p>
            <p style="color: #67e8f9; font-family: monospace; background: rgba(0,0,0,0.3); padding: 0.5rem; border-radius: 8px;">
            z = (x - μ) / σ
            </p>
            
            <h3 style="color: #67e8f9;">4️⃣ Train-Test Split</h3>
            <p style="color: #cbd5e1;">แบ่งข้อมูลเป็น:</p>
            <ul style="color: #cbd5e1; padding-left: 1.5rem;">
                <li><strong>Training Set:</strong> 80% (ฝึกโมเดล)</li>
                <li><strong>Test Set:</strong> 20% (ทดสอบโมเดล)</li>
            </ul>
            <p style="color: #67e8f9;">ใช้ stratify=y เพื่อรักษาสัดส่วน classes</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="metric-card" style="text-align: left; margin-top: 1.5rem;">
        <h3 style="color: #67e8f9; margin-top: 0;"> โครงสร้าง Dataset</h3>
        <p style="color: #cbd5e1;"><strong>จำนวนข้อมูล:</strong> 100,000+ records</p>
        <p style="color: #cbd5e1;"><strong>Features (8 ตัว):</strong> Gender, Age, Hypertension, Heart Disease, Smoking History, BMI, HbA1c Level, Blood Glucose Level</p>
        <p style="color: #cbd5e1;"><strong>Target Variable:</strong> Diabetes (0 = ไม่เป็น, 1 = เป็น)</p>
    </div>
    ''', unsafe_allow_html=True)

# ==================== TAB 3: ทฤษฎีโมเดล ML ====================
with tabs[2]:
    st.markdown('<div class="main-header">🧠 ทฤษฎีโมเดล Machine Learning</div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="metric-card" style="text-align: left; margin-bottom: 1.5rem;">
        <h3 style="color: #67e8f9; margin-top: 0;">🌳 Random Forest Classifier</h3>
        <p style="color: #cbd5e1; line-height: 1.8;">
        <strong>Random Forest</strong> เป็นอัลกอริทึม Ensemble Learning ที่สร้าง Decision Trees หลายๆ ต้น แล้วนำผลลัพธ์มาโหวตเพื่อทำนายผล
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="metric-card" style="text-align: left;">
            <h3 style="color: #67e8f9; margin-top: 0;">🔄 หลักการทำงาน 4 ขั้นตอน</h3>
            <ol style="color: #cbd5e1; padding-left: 1.5rem; line-height: 1.8;">
                <li><strong>Bootstrap Sampling:</strong> สุ่มเลือกข้อมูลด้วยการ sampling แบบมีแทนที่</li>
                <li><strong>Feature Randomness:</strong> สุ่มเลือก features ในการ split แต่ละ node</li>
                <li><strong>Tree Building:</strong> สร้าง Decision Tree หลายร้อยต้น</li>
                <li><strong>Majority Voting:</strong> นำผลลัพธ์จากทุก trees มาโหวตเพื่อทำนายผล</li>
            </ol>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="metric-card" style="text-align: left;">
            <h3 style="color: #67e8f9; margin-top: 0;">✅ เหตุผลที่เลือก Random Forest</h3>
            <ul style="color: #cbd5e1; padding-left: 1.5rem; line-height: 1.8;">
                <li>ป้องกันปัญหา <strong>Overfitting</strong> ได้ดี</li>
                <li>จัดการกับ <strong>Imbalanced Data</strong> ได้ดี (ใช้ class_weight='balanced')</li>
                <li>สามารถบอก <strong>Feature Importance</strong> ได้</li>
                <li>ให้ความ<strong>แม่นยำสูง</strong> (Accuracy > 95%)</li>
                <li>ทำงานได้เร็วด้วย <strong>Parallel Processing</strong> (n_jobs=-1)</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="metric-card" style="text-align: left; margin-top: 1.5rem;">
        <h3 style="color: #67e8f9; margin-top: 0;">️ Parameters ที่ใช้</h3>
        <p style="color: #67e8f9; font-family: monospace; background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; line-height: 1.8;">
        n_estimators = 200  # จำนวน Decision Trees<br>
        max_depth = 15      # ความลึกสูงสุดของ tree<br>
        class_weight = 'balanced'  # จัดการ imbalanced data<br>
        random_state = 42   # เพื่อความ reproducible<br>
        n_jobs = -1         # ใช้ทุก CPU cores
        </p>
    </div>
    ''', unsafe_allow_html=True)

# ==================== TAB 4: วิเคราะห์ข้อมูล ====================
with tabs[3]:
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

# ==================== TAB 5: ประสิทธิภาพโมเดล ====================
with tabs[4]:
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    
    # ตารางเปรียบเทียบหลายโมเดล
    st.markdown('''
    <div class="metric-card" style="text-align: left; margin-bottom: 2rem;">
        <h3 style="color: #67e8f9; margin-top: 0;">📊 ตารางเปรียบเทียบโมเดล</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    comparison_data = []
    for name, result in metrics['results'].items():
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.2%}",
            'Precision': f"{result['precision']:.2%}",
            'Recall': f"{result['recall']:.2%}",
            'F1-Score': f"{result['f1']:.2%}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, height=200)
    
    st.success(f"🏆 **โมเดลที่ดีที่สุด: {metrics['best_model']}** (Accuracy: {metrics['best_acc']:.2%})")
    
    # Confusion Matrix ของโมเดลที่ดีที่สุด
    best_result = metrics['results'][metrics['best_model']]
    
    st.markdown('''
    <div class="metric-card" style="text-align: left; margin-top: 2rem; margin-bottom: 1rem;">
        <h3 style="color: #67e8f9; margin-top: 0;">🎯 Confusion Matrix ({})</h3>
    </div>
    '''.format(metrics['best_model']), unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(metrics['y_test'], best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 14, "color": "white", "fontweight": "bold"})
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)
    
    # ROC Curve
    st.markdown('''
    <div class="metric-card" style="text-align: left; margin-top: 2rem; margin-bottom: 1rem;">
        <h3 style="color: #67e8f9; margin-top: 0;">📈 ROC Curve</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, result in metrics['results'].items():
        fpr, tpr, _ = roc_curve(metrics['y_test'], result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', color='white', fontsize=12)
    ax.set_ylabel('True Positive Rate', color='white', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontweight='bold', fontsize=14, color='#ffffff', pad=15)
    ax.legend(loc='lower right', facecolor=(1, 1, 1, 0.1), edgecolor='none', labelcolor='white')
    ax.tick_params(colors='white')
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)

# ==================== TAB 6: ทายผลความเสี่ยง ====================
with tabs[5]:
    st.markdown('<div class="main-header">🎮 ตรวจสอบความเสี่ยง</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0; color:#67e8f9;">👤 ข้อมูลส่วนบุคคล</h3></div>', unsafe_allow_html=True)
        gender = st.selectbox("เพศ", ["Female", "Male", "Other"])
        age = st.slider("อายุ (ปี)", 0, 100, 30)
        hypertension = st.radio("ความดันโลหิตสูง", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
        heart_disease = st.radio("โรคหัวใจ", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")
    
    with col2:
        st.markdown('<div class="metric-card" style="text-align:left; margin-bottom:1rem;"><h3 style="margin-top:0; color:#67e8f9;"> ข้อมูลทางการแพทย์</h3></div>', unsafe_allow_html=True)
        smoking = st.selectbox("สูบบุหรี่", ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0, 0.1)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.7, 0.1)
        glucose = st.number_input("น้ำตาลในเลือด (mg/dL)", 50, 400, 100)

    if st.button(" ทำนายผล"):
        gender_enc = metrics['le_gender'].transform([gender])[0]
        smoking_enc = metrics['le_smoking'].transform([smoking])[0]
        
        input_data = np.array([[gender_enc, age, hypertension, heart_disease, smoking_enc, bmi, hba1c, glucose]])
        input_scaled = metrics['scaler'].transform(input_data)
        
        best_model_obj = metrics['results'][metrics['best_model']]['model']
        prediction = best_model_obj.predict(input_scaled)[0]
        proba = best_model_obj.predict_proba(input_scaled)[0]
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

# ==================== TAB 7: ผู้พัฒนา ====================
with tabs[6]:
    st.markdown('<div class="main-header">👨‍💻 เกี่ยวกับผู้พัฒนา</div>', unsafe_allow_html=True)
    
    img_html = ""
    if os.path.exists("profile.jpg"):
        with open("profile.jpg", "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
            img_html = f'<img src="data:image/jpeg;base64,{img_data}" class="dev-avatar" alt="Profile">'
    else:
        img_html = '<img src="https://ui-avatars.com/api/?name=Phuwadit+Cham&background=06b6d4&color=fff&size=200&font-size=0.4" class="dev-avatar" alt="Profile">'

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