import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv(r"D:\Neha Final Year Project\BankChurners.csv")
df.head()
df['Churn'] = df['Attrition_Flag'].map({
    'Existing Customer': 0,
    'Attrited Customer': 1
})

df = df.drop(columns=[
    'Attrition_Flag',
    'CLIENTNUM',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
])
df['Avg_Transaction_Value'] = df['Total_Trans_Amt'] / (df['Total_Trans_Ct'] + 1)
df['Engagement_Score'] = (
    df['Total_Relationship_Count'] +
    df['Contacts_Count_12_mon'] -
    df['Months_Inactive_12_mon']
)
X = df.drop('Churn', axis=1)
y = df['Churn']

categorical_cols = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    
    "SVM": SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    ),
    
    "Naive Bayes": GaussianNB(),
    
    "KNN": KNeighborsClassifier(n_neighbors=7),
    
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ),
    
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
}
baseline_results = {}

for name, model in models.items():
    
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    baseline_results[name] = {
        "Accuracy Before SMOTE": accuracy_score(y_test, preds),
        "Recall Before SMOTE": recall_score(y_test, preds)
    }

pd.DataFrame(baseline_results).T
smote_results = {}

for name, model in models.items():
    
    pipe = ImbPipeline([
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    smote_results[name] = {
        "Accuracy After SMOTE": accuracy_score(y_test, preds),
        "Recall After SMOTE": recall_score(y_test, preds)
    }

pd.DataFrame(smote_results).T
final_pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    ))
])

final_pipeline.fit(X_train, y_train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    final_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='recall'
)

print("CV Recall Scores:", cv_scores)
print("Mean CV Recall:", cv_scores.mean())
probs = final_pipeline.predict_proba(X_test)[:, 1]

threshold = 0.40
y_pred = (probs >= threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, probs)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1:", f1)
print("AUC:", auc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
metrics = {
    "accuracy": accuracy,
    "recall": recall,
    "precision": precision,
    "f1_score": f1,
    "auc": auc
}
import joblib

# Save trained pipeline
joblib.dump(final_pipeline, "model.pkl")
joblib.dump(metrics, "metrics.pkl")
# Get feature names from preprocessor
feature_names = final_pipeline.named_steps['preprocess'].get_feature_names_out()

# Get trained model
model = final_pipeline.named_steps['model']

# Get importance values
importances = model.feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Show top 15 features
feature_importance_df.head(15)
plt.figure(figsize=(8,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df.head(10)
)
plt.title("Top 10 Important Features")
plt.show()
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
print("Model saved successfully!")