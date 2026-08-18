import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏥 DIABETES PREDICTION - RANDOM FOREST MODEL")
print("="*60)

# ==================== 1. LOAD DATA ====================
print("\n[1/7] Loading dataset...")
df = pd.read_csv('diabetes_prediction_dataset.csv')
print(f"✅ Loaded {len(df)} samples, {len(df.columns)} columns")

# ==================== 2. PREPROCESSING ====================
print("\n[2/7] Preprocessing data...")

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Encode categorical variables
le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

# Save encoders
joblib.dump(le_gender, 'label_encoder_gender.pkl')
joblib.dump(le_smoking, 'label_encoder_smoking.pkl')

# Separate features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")
print(f"✅ Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"✅ Class distribution - Test:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ==================== 3. TRAIN MODEL ====================
print("\n[3/7] Training Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=200,          # จำนวน Decision Trees
    max_depth=15,              # ความลึกสูงสุด
    min_samples_split=5,       # samples ขั้นต่ำในการ split
    min_samples_leaf=2,        # samples ขั้นต่ำใน leaf
    max_features='sqrt',       # สุ่ม features ที่ใช้
    class_weight='balanced',   # จัดการ imbalanced data
    random_state=42,
    n_jobs=-1                  # ใช้ CPU ทุก core
)

rf_model.fit(X_train_scaled, y_train)
print("✅ Model trained successfully!")

# ==================== 4. EVALUATE ====================
print("\n[4/7] Evaluating model...")

y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n🔄 5-Fold Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==================== 5. SAVE MODEL ====================
print("\n[5/7] Saving model...")
joblib.dump(rf_model, 'random_forest_model.pkl')
print("✅ Saved: random_forest_model.pkl")

# ==================== 6. VISUALIZATION ====================
print("\n[6/7] Creating visualizations...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()

# Feature Importance
importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()

print("✅ Saved: confusion_matrix.png, roc_curve.png, feature_importance.png")

# ==================== 7. CLASSIFICATION REPORT ====================
print("\n[7/7] Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred,
                           target_names=['No Diabetes (0)', 'Diabetes (1)']))

print("\n" + "="*60)
print("✅ TRAINING COMPLETED!")
print("="*60)
print("\n📁 Output Files:")
print("   • random_forest_model.pkl")
print("   • scaler.pkl")
print("   • label_encoder_gender.pkl")
print("   • label_encoder_smoking.pkl")
print("   • feature_names.pkl")
print("   • confusion_matrix.png")
print("   • roc_curve.png")
print("   • feature_importance.png")