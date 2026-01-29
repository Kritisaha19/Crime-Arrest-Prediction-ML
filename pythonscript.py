import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
df = pd.read_csv("/Users/kritisaha/Downloads/Crime_Data_from_2020_to_Present (1).csv")
df['Arrest_Outcome'] = np.where(
    df['Status Desc'].isin(['Adult Arrest', 'Juvenile Arrest']),
    1,
    0)
selected_columns = [
    'AREA NAME',
    'Crm Cd Desc',
    'Vict Sex',
    'Vict Age',
    'Weapon Desc',
    'Premis Desc',
    'TIME OCC',
    'Arrest_Outcome'
]
df = df[selected_columns].copy() 
df['Vict Age'] = df['Vict Age'].fillna(df['Vict Age'].median())
df = df.fillna('Unknown')
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
X = df.drop('Arrest_Outcome', axis=1)
y = df['Arrest_Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Arrest Outcome Distribution")
plt.xlabel("Arrest Outcome (0 = No Arrest, 1 = Arrest)")
plt.ylabel("Count")
plt.show()
class_counts = y.value_counts(normalize=True) * 100
plt.figure(figsize=(6,4))
class_counts.plot(kind='bar')
plt.title("Class Distribution (%)")
plt.ylabel("Percentage")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
feature_importance = pd.Series(model.coef_[0],index=X.columns).sort_values()
plt.figure(figsize=(8,5))
feature_importance.plot(kind='barh')
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Coefficient Value")
plt.show()
plt.figure(figsize=(7,4))
sns.histplot(y_prob, bins=30, kde=True)
plt.title("Distribution of Predicted Arrest Probabilities")
plt.xlabel("Predicted Probability of Arrest")
plt.ylabel("Frequency")
plt.show()
thresholds = np.arange(0.1, 1.0, 0.1)
precision_vals = []
recall_vals = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    precision_vals.append(precision_score(y_test, preds))
    recall_vals.append(recall_score(y_test, preds))
plt.figure(figsize=(7,4))
plt.plot(thresholds, precision_vals, label="Precision")
plt.plot(thresholds, recall_vals, label="Recall")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision–Recall Tradeoff by Threshold")
plt.legend()
plt.show()
time_df = pd.DataFrame({
    'Hour': X_test[:, X.columns.get_loc('TIME OCC')],
    'Arrest_Probability': y_prob
})
plt.figure(figsize=(7,4))
sns.lineplot(data=time_df, x='Hour', y='Arrest_Probability')
plt.title("Arrest Probability by Time of Occurrence")
plt.xlabel("Time (HHMM)")
plt.ylabel("Average Arrest Probability")
plt.show()
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)
plt.figure(figsize=(7,4))
sns.barplot(
    data=coef_df.head(10),
    x='Coefficient',
    y='Feature'
)
plt.title("Top 10 Features Influencing Arrest Outcome")
plt.show()
# Recreate DataFrame from scaled X_test
error_df = pd.DataFrame(
    X_test,
    columns=X.columns
)
# Add actual and predicted values
error_df['Actual'] = y_test.values
error_df['Predicted'] = y_pred
# Identify errors
error_df['Error'] = error_df['Actual'] != error_df['Predicted']
crime_prob = pd.DataFrame({
    'CrimeType': df['Crm Cd Desc'],
    'ArrestProb': model.predict_proba(X)[:, 1]
})
crime_grouped = crime_prob.groupby('CrimeType').mean().sort_values(
    by='ArrestProb', ascending=False
).head(10)
plt.figure(figsize=(7,4))
crime_grouped.plot(kind='barh')
plt.title("Top Crime Types by Average Arrest Probability")
plt.xlabel("Average Arrest Probability")
plt.show()
confidence = np.abs(y_prob - 0.5)
plt.figure(figsize=(7,4))
sns.boxplot(x=(y_test == y_pred), y=confidence)
plt.xticks([0,1], ['Incorrect', 'Correct'])
plt.title("Model Confidence vs Prediction Correctness")
plt.ylabel("Confidence Level")
plt.show()
