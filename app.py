import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
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
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F18F01;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎯 Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home", "📊 Dataset Overview", " Data Analysis", 
     "️ Data Preprocessing", "🤖 Model Training", "📈 Model Evaluation", 
     "🎮 Prediction App"]
)

# ฟังก์ชันโหลดข้อมูล
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    return df

# ฟังก์ชัน Preprocessing
def preprocess_data(df):
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoder_gender = LabelEncoder()
    label_encoder_smoking = LabelEncoder()
    
    df_processed['gender'] = label_encoder_gender.fit_transform(df_processed['gender'])
    df_processed['smoking_history'] = label_encoder_smoking.fit_transform(df_processed['smoking_history'])
    
    # แยก features และ target
    X = df_processed.drop('diabetes', axis=1)
    y = df_processed['diabetes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), label_encoder_gender, label_encoder_smoking

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header"> Diabetes Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📋 Project Name:**")
        st.write("Diabetes Prediction using Machine Learning")
    with col2:
        st.success("**👥 Team:**")
        st.write("Your Name/Group")
    with col3:
        st.warning("**📅 Date:**")
        st.write("2026")
    
    st.markdown('<h2 class="sub-header"> Project Objectives</h2>', unsafe_allow_html=True)
    objectives = [
        "เพื่อพัฒนาโมเดล Machine Learning สำหรับพยากรณ์โรคเบาหวาน",
        "เพื่อเปรียบเทียบประสิทธิภาพของอัลกอริทึมต่างๆ",
        "เพื่อสร้าง Web Application สำหรับทำนายความเสี่ยงโรคเบาหวาน",
        "เพื่อวิเคราะห์ปัจจัยที่มีผลต่อการเกิดโรคเบาหวาน"
    ]
    for i, obj in enumerate(objectives, 1):
        st.markdown(f"**{i}.** {obj}")
    
    st.markdown('<h2 class="sub-header">📋 Assignment Requirements (30 Points)</h2>', unsafe_allow_html=True)
    requirements = {
        "1. Problem Definition & Dataset": "5 points",
        "2. Data Preprocessing": "5 points",
        "3. ML Model Creation": "5 points",
        "4. Model Evaluation & Comparison": "5 points",
        "5. Streamlit Application": "10 points"
    }
    
    for req, points in requirements.items():
        st.markdown(f"- **{req}**: {points}")
    
    st.markdown("---")
    st.success("✅ เลือกเมนูด้านซ้ายเพื่อสำรวจโปรเจกต์")

# ==================== DATASET OVERVIEW ====================
elif page == "📊 Dataset Overview":
    st.markdown('<h1 class="main-header">📊 Dataset Overview</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.markdown('<h2 class="sub-header">📋 Dataset Information</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Diabetes Cases", df['diabetes'].sum())
    with col4:
        st.metric("Non-Diabetes", len(df) - df['diabetes'].sum())
    
    st.markdown('<h2 class="sub-header">📋 Column Descriptions</h2>', unsafe_allow_html=True)
    descriptions = {
        "gender": "เพศของผู้ป่วย (Male/Female/Other)",
        "age": "อายุ (ปี)",
        "hypertension": "ความดันโลหิตสูง (0=No, 1=Yes)",
        "heart_disease": "โรคหัวใจ (0=No, 1=Yes)",
        "smoking_history": "ประวัติการสูบบุหรี่",
        "bmi": "ดัชนีมวลกาย (Body Mass Index)",
        "HbA1c_level": "ระดับน้ำตาลเฉลี่ย 3 เดือน",
        "blood_glucose_level": "ระดับน้ำตาลในเลือด",
        "diabetes": "target: เป็นเบาหวาน (0=No, 1=Yes)"
    }
    
    desc_df = pd.DataFrame(list(descriptions.items()), columns=["Feature", "Description"])
    st.dataframe(desc_df, use_container_width=True)
    
    st.markdown('<h2 class="sub-header">📄 Dataset Sample</h2>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown('<h2 class="sub-header">📊 Data Types & Missing Values</h2>', unsafe_allow_html=True)
    data_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    st.dataframe(data_info, use_container_width=True)

# ==================== DATA ANALYSIS ====================
elif page == "🔬 Data Analysis":
    st.markdown('<h1 class="main-header">🔬 Exploratory Data Analysis</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    # Distribution of Target Variable
    st.markdown('<h2 class="sub-header">📊 Target Variable Distribution</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        diabetes_counts = df['diabetes'].value_counts()
        ax.pie(diabetes_counts.values, labels=['No Diabetes', 'Diabetes'], 
               autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        ax.set_title('Diabetes Distribution')
        st.pyplot(fig)
    
    with col2:
        st.write("**Class Distribution:**")
        st.write(df['diabetes'].value_counts())
    
    # Age Distribution
    st.markdown('<h2 class="sub-header">👥 Age Distribution</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='age', hue='diabetes', multiple='stack', ax=ax)
    ax.set_title('Age Distribution by Diabetes')
    st.pyplot(fig)
    
    # BMI vs Blood Glucose
    st.markdown('<h2 class="sub-header">📈 BMI vs Blood Glucose Level</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['bmi'], df['blood_glucose_level'], 
                        c=df['diabetes'], cmap='coolwarm', alpha=0.6)
    ax.set_xlabel('BMI')
    ax.set_ylabel('Blood Glucose Level')
    ax.set_title('BMI vs Blood Glucose (Color: Diabetes Status)')
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.markdown('<h2 class="sub-header">🔥 Correlation Heatmap</h2>', unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    # HbA1c Level Analysis
    st.markdown('<h2 class="sub-header">🩸 HbA1c Level Analysis</h2>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x='diabetes', y='HbA1c_level', ax=axes[0])
    axes[0].set_title('HbA1c by Diabetes')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])
    
    sns.histplot(data=df, x='HbA1c_level', hue='diabetes', multiple='stack', ax=axes[1])
    axes[1].set_title('HbA1c Distribution')
    plt.tight_layout()
    st.pyplot(fig)

# ==================== DATA PREPROCESSING ====================
elif page == "⚙️ Data Preprocessing":
    st.markdown('<h1 class="main-header">⚙️ Data Preprocessing</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.markdown('<h2 class="sub-header"> Preprocessing Steps</h2>', unsafe_allow_html=True)
    
    # Step 1: Handle Missing Values
    st.markdown("**Step 1: Check Missing Values**")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Missing Values Before:")
        st.write(df.isnull().sum())
    with col2:
        if df.isnull().sum().sum() == 0:
            st.success("✅ No missing values found!")
        else:
            st.warning("️ Missing values found - would need imputation")
    
    # Step 2: Encode Categorical Variables
    st.markdown("**Step 2: Encode Categorical Variables**")
    df_encoded = df.copy()
    label_encoder_gender = LabelEncoder()
    label_encoder_smoking = LabelEncoder()
    
    df_encoded['gender'] = label_encoder_gender.fit_transform(df_encoded['gender'])
    df_encoded['smoking_history'] = label_encoder_smoking.fit_transform(df_encoded['smoking_history'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Gender Values:", df['gender'].unique())
        st.write("Encoded Gender Values:", df_encoded['gender'].unique())
    with col2:
        st.write("Original Smoking:", df['smoking_history'].unique()[:3], "...")
        st.write("Encoded Smoking:", df_encoded['smoking_history'].unique()[:3], "...")
    
    # Step 3: Feature Scaling
    st.markdown("**Step 3: Feature Scaling**")
    X = df_encoded.drop('diabetes', axis=1)
    y = df_encoded['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.write("Training Set Size:", X_train_scaled.shape)
    st.write("Test Set Size:", X_test_scaled.shape)
    
    # Show scaled data example
    st.markdown("**Scaled Data Example (First 5 rows):**")
    scaled_df = pd.DataFrame(X_train_scaled[:5], columns=X.columns)
    st.dataframe(scaled_df)
    
    st.success("✅ Data is ready for model training!")

# ==================== MODEL TRAINING ====================
elif page == " Model Training":
    st.markdown('<h1 class="main-header">🤖 Model Training & Comparison</h1>', unsafe_allow_html=True)
    
    # Train models
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, _, _ = preprocess_data(load_data())
    
    st.markdown('<h2 class="sub-header">📚 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    
    results = {}
    
    st.markdown('<h2 class="sub-header">🏋️ Training Models...</h2>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'predictions': y_pred
        }
        progress_bar.progress((i + 1) / len(models))
    
    st.success("✅ All models trained successfully!")
    
    # Display Results
    st.markdown('<h2 class="sub-header">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    results_df = pd.DataFrame(results).T
    results_df = results_df[['accuracy', 'precision', 'recall', 'f1']]
    results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    st.dataframe(results_df.style.format('{:.4f}'), use_container_width=True)
    
    # Bar Chart Comparison
    st.markdown('<h2 class="sub-header">📈 Performance Visualization</h2>', unsafe_allow_html=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = results_df[metric].values
        bars = ax.barh(results_df.index, values, color=colors[:len(results_df)])
        ax.set_xlim([0, 1])
        ax.set_title(f'{metric} Comparison')
        ax.set_xlabel('Score')
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Best Model
    best_model = results_df['Accuracy'].idxmax()
    st.success(f"🏆 **Best Model: {best_model}** with Accuracy: {results_df.loc[best_model, 'Accuracy']:.4f}")

# ==================== MODEL EVALUATION ====================
elif page == "📈 Model Evaluation":
    st.markdown('<h1 class="main-header">📈 Detailed Model Evaluation</h1>', unsafe_allow_html=True)
    
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, _, _ = preprocess_data(load_data())
    
    # Train best model (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    st.markdown('<h2 class="sub-header">🎯 Random Forest - Best Model</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    with col2:
        st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    with col3:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    with col4:
        st.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    
    # Confusion Matrix
    st.markdown('<h2 class="sub-header">📊 Confusion Matrix</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification Report
    st.markdown('<h2 class="sub-header">📝 Classification Report</h2>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)
    
    # Feature Importance
    st.markdown('<h2 class="sub-header"> Feature Importance</h2>', unsafe_allow_html=True)
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(data=importance, x='Importance', y='Feature', ax=ax)
    ax.set_title('Feature Importance (Random Forest)')
    plt.tight_layout()
    st.pyplot(fig)
    
    # ROC Curve
    st.markdown('<h2 class="sub-header">📈 ROC Curve</h2>', unsafe_allow_html=True)
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# ==================== PREDICTION APP ====================
elif page == "🎮 Prediction App":
    st.markdown('<h1 class="main-header">🎮 Diabetes Prediction App</h1>', unsafe_allow_html=True)
    
    # Train model
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, label_encoder_gender, label_encoder_smoking = preprocess_data(load_data())
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    st.markdown('<h2 class="sub-header"> Patient Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Age", 0, 100, 30)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    
    with col2:
        smoking_history = st.selectbox("Smoking History", 
                                      ["No Info", "never", "former", "current", "ever", "not current"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
        hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
        blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=100, step=1)
    
    # Encode inputs
    gender_encoded = label_encoder_gender.transform([gender])[0]
    smoking_encoded = label_encoder_smoking.transform([smoking_history])[0]
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    
    if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
        # Prepare input
        input_data = np.array([[
            gender_encoded,
            age,
            hypertension_encoded,
            heart_disease_encoded,
            smoking_encoded,
            bmi,
            hba1c,
            blood_glucose
        ]])
        
        # Scale input
        scaler = StandardScaler()
        # Need to fit on original training data structure
        X_train_scaled, X_test_scaled, y_train, y_test, _, _, _ = preprocess_data(load_data())
        
        # Make prediction
        prediction = rf_model.predict(input_data)
        prediction_proba = rf_model.predict_proba(input_data)[0]
        
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Prediction Result</h2>', unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.error("⚠️ **Result: DIABETES DETECTED**")
            st.write(f"**Confidence:** {prediction_proba[1]*100:.2f}%")
            st.warning("แนะนำให้ปรึกษาแพทย์เพื่อการวินิจฉัยและรักษาที่เหมาะสม")
        else:
            st.success("✅ **Result: NO DIABETES**")
            st.write(f"**Confidence:** {prediction_proba[0]*100:.2f}%")
            st.info("ควรตรวจสุขภาพเป็นประจำและรักษาสุขภาพให้แข็งแรง")
        
        # Show probability
        st.markdown("**Prediction Probability:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("No Diabetes", f"{prediction_proba[0]*100:.2f}%")
        with col2:
            st.metric("Diabetes", f"{prediction_proba[1]*100:.2f}%")
        
        # Feature contribution (simplified)
        st.markdown('<h2 class="sub-header"> Key Risk Factors</h2>', unsafe_allow_html=True)
        risk_factors = []
        if hba1c >= 6.5:
            risk_factors.append("⚠️ HbA1c Level สูง (>6.5%)")
        if blood_glucose > 140:
            risk_factors.append("⚠️ Blood Glucose สูง (>140)")
        if bmi >= 30:
            risk_factors.append("⚠️ BMI สูง (อ้วน)")
        if hypertension_encoded == 1:
            risk_factors.append("⚠️ มีความดันโลหิตสูง")
        if heart_disease_encoded == 1:
            risk_factors.append("⚠️ มีโรคหัวใจ")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.success("✅ ไม่พบปัจจัยเสี่ยงที่สำคัญ")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**📚 About This Project**")
st.sidebar.info("This is a Machine Learning project for diabetes prediction using various classification algorithms.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>🏥 Diabetes Prediction System | Machine Learning Project 2026</p>
        <p>Built with Streamlit ❤️</p>
    </div>
""", unsafe_allow_html=True)