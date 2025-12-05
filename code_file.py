# Main Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Deep Learning (AI Model)
import tensorflow as tf
from tensorflow import keras

FILE_PATH = 'heart.csv'
TARGET_COLUMN = 'target'
RANDOM_SEED = 42

try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found. Please ensure it is uploaded.")
    exit()

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in categorical_cols:
    df[col] = df[col].astype('object')

print("\nTarget Variable Distribution:\n", df[TARGET_COLUMN].value_counts())
df_encoded = pd.get_dummies(df, drop_first=True)
print(f"\nDataFrame shape after One-Hot Encoding: {df_encoded.shape}")

X = df_encoded.drop(TARGET_COLUMN, axis=1) 
y = df_encoded[TARGET_COLUMN]             
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nData split and scaled successfully.")
def get_metrics(y_true, y_pred, y_proba):
    """Calculates standard classification metrics."""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
results = {}

# --- A. Logistic Regression ---
log_reg = LogisticRegression(random_state=RANDOM_SEED, solver='liblinear')
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
y_proba_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = get_metrics(y_test, y_pred_log_reg, y_proba_log_reg)

# --- B. Decision Tree ---
dt_classifier = DecisionTreeClassifier(random_state=RANDOM_SEED)
dt_classifier.fit(X_train_scaled, y_train)
y_pred_dt = dt_classifier.predict(X_test_scaled)
y_proba_dt = dt_classifier.predict_proba(X_test_scaled)[:, 1]
results['Decision Tree'] = get_metrics(y_test, y_pred_dt, y_proba_dt)

# --- C. Random Forest ---
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
rf_classifier.fit(X_train_scaled, y_train)
y_pred_rf = rf_classifier.predict(X_test_scaled)
y_proba_rf = rf_classifier.predict_proba(X_test_scaled)[:, 1]
results['Random Forest'] = get_metrics(y_test, y_pred_rf, y_proba_rf)

# --- D. Neural Network (AI Model) ---
input_dim = X_train_scaled.shape[1]
model = keras.Sequential([
    keras.layers.Dense(units=32, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0) 

y_proba_nn = model.predict(X_test_scaled).flatten()
y_pred_nn = (y_proba_nn > 0.5).astype("int32")
results['Neural Network'] = get_metrics(y_test, y_pred_nn, y_proba_nn)

df_comparison = pd.DataFrame(results).T 
df_comparison = df_comparison.sort_values(by='ROC-AUC', ascending=False)

print("\n" + "="*50)
print("             FINAL MODEL COMPARISON (TEST SET)")
print("="*50)
print(df_comparison.to_string(float_format="{:.4f}".format))
print("="*50)

df_comparison.to_csv('model_comparison_results.csv')
print("\nComparison results saved to 'model_comparison_results.csv'")

importances = rf_classifier.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]
sorted_importances = importances[sorted_indices]
sorted_feature_names = feature_names[sorted_indices]

plt.figure(figsize=(12, 6))
plt.title("Random Forest Feature Importance")
plt.bar(range(X_train_scaled.shape[1]), sorted_importances, align='center')
plt.xticks(range(X_train_scaled.shape[1]), sorted_feature_names, rotation=90)
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png')
plt.close()
print("\nFeature importance plot saved as 'random_forest_feature_importance.png'")

with open('best_heart_model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)
print("\nRandom Forest model (rf_classifier) saved as 'best_heart_model.pkl'")

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler object saved as 'scaler.pkl'")

try:
    tf.keras.models.save_model(model, 'best_nn_model.keras')
    print("Neural Network model saved as 'best_nn_model.keras'")
except Exception as e:
    print(f"Failed to save Neural Network model due to: {e}")

