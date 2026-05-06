import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Explainable Churn Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* =========================
GLOBAL FONT
========================= */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
}
/* =========================
MAIN TITLES
========================= */
h1 { font-size: 50px !important; font-weight:700; }
h2 { font-size: 40px !important; font-weight:700; }
h3 { font-size: 30px !important; font-weight:700; }
h4 { font-size: 24px !important; font-weight:600; }
/* =========================
METRIC CARDS
========================= */
/* metric container */
[data-testid="stMetric"]{
    background-color:#ffffff;
    padding:22px;
    border-radius:12px;
    border:1px solid #e2e8f0;
}
/* metric label (Accuracy, Recall etc) */
[data-testid="stMetricLabel"]{
    font-size:22px !important;
    font-weight:600 !important;
}
/* metric value (0.97 etc) */
[data-testid="stMetricValue"]{
    font-size:34px !important;
    font-weight:700 !important;
}
/* =========================
SIDEBAR TEXT
========================= */
[data-testid="stSidebar"] *{
    font-size:18px !important;
}
/* =========================
BUTTONS========================= */
div.stButton > button{
    font-size:20px !important;
    font-weight:600;
    padding:10px;
}
/* =========================
TABLES
========================= */
table{
    font-size:18px !important;
}
/* =========================
PROGRESS BAR
========================= */
.stProgress > div > div > div{
    height:18px;
}
/* =========================
SUCCESS / INFO BOX
========================= */
.stAlert{
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)
# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
model_pipeline = load_model()

@st.cache_resource
def load_metrics():
    return joblib.load("metrics.pkl")
metrics = load_metrics()
# ==========================================
# HELPER FUNCTIONS
# ==========================================
def segment_risk(p):
    if p >= 0.70:
        return "High Risk", "🔴"
    elif p >= 0.30:
        return "Medium Risk", "🟠"
    else:
        return "Low Risk", "🟢"
def recommended_action(risk_group, var):
    if risk_group == "High Risk" and var > 5000:
        return "Immediate retention call + premium incentive"
    elif risk_group == "High Risk":
        return "Retention email campaign"
    elif risk_group == "Medium Risk" and var > 5000:
        return "Targeted engagement campaign"
    elif risk_group == "Medium Risk":
        return "Monitor engagement"
    else:
        return "No immediate action"
# ==========================================
# HEADER
# ==========================================
st.title("💳 Early Bank Customer Churn Risk Prediction")
st.markdown("### Decision Support System using Machine Learning")
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Overview",
    "🔍 Risk Prediction",
    "🧠 Explainability",
    "📈 Decision Support"
])
# ==========================================
# MODEL PERFORMANCE SECTION
# ==========================================
with tab1:
    st.subheader("📊 Model Performance Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ACCURACY", f"{metrics['accuracy']:.2f}")
    with col2:
        st.metric("RECALL", f"{metrics['recall']:.2f}")
    with col3:
        st.metric("PRECISION", f"{metrics['precision']:.2f}")
    with col4:
        st.metric("AUC SCORE", f"{metrics['auc']:.2f}")
    # ==========================================
    # SYSTEM ARCHITECTURE
    # ==========================================
    st.markdown("---")
    st.subheader("⚙ System Architecture")
    st.write("""
        1. Data Preprocessing using ColumnTransformer  
        2. Class Imbalance Handling using SMOTE  
        3. Machine Learning Model: XGBoost  
        4. Explainability using Feature Importance  
        5. Risk Segmentation using Churn Probability  
        6. Financial Risk Estimation using Value at Risk (VaR)  
        7. Customer Prioritisation using VaR  
        8. Decision Support using Retention Strategy Simulation
        """)
        # ==========================================
        # METHODOLOGY FOOTER
        # ==========================================
    st.markdown("---")
    with st.expander("ℹ️ Project Methodology"):
        st.write("""
            • Data preprocessing performed using ColumnTransformer with OneHotEncoding for categorical    variables.  
            • Class imbalance addressed using SMOTE oversampling technique during model training.  
            • Multiple machine learning algorithms evaluated and XGBoost selected as the final model.  
            • Model performance evaluated using Accuracy, Recall, Precision and AUC score.  
            • Feature importance used to provide explainability for churn predictions.  
            • Churn probability used for risk segmentation of customers.  
            • Value at Risk (VaR) used to estimate potential financial loss from customer churn.  
            • What-if simulation implemented to analyze the impact of retention strategies.
        """) 
# ==========================================
# SIDEBAR INPUT
# ==========================================
st.sidebar.header("📋 Customer Profile")
st.sidebar.markdown("Adjust parameters to simulate risk.")
user_inputs = {}
with st.sidebar:
    st.subheader("Demographics")
    user_inputs["Customer_Age"] = st.slider("Age",18,80,44)
    user_inputs["Gender"] = st.selectbox("Gender",["M","F"])
    user_inputs["Dependent_count"] = st.number_input("Dependents",0,5,2)
    user_inputs["Education_Level"] = st.selectbox(
        "Education",
        ["Graduate","High School","Unknown","Uneducated","College","Post-Graduate","Doctorate"]
    )
    user_inputs["Marital_Status"] = st.selectbox(
        "Marital Status",
        ["Married","Single","Unknown","Divorced"]
    )
    user_inputs["Income_Category"] = st.selectbox(
        "Income",
        ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Unknown"]
    )
    st.markdown("---")
    st.subheader("Account Details")
    user_inputs["Card_Category"] = st.selectbox(
        "Card Type",
        ["Blue","Silver","Gold","Platinum"]
    )
    user_inputs["Months_on_book"] = st.slider("Months with Bank",13,60,36)
    user_inputs["Total_Relationship_Count"] = st.slider("Products",1,6,3)
    user_inputs["Months_Inactive_12_mon"] = st.slider("Inactive Months",0,12,1)
    user_inputs["Contacts_Count_12_mon"] = st.slider("Contacts with Bank",0,6,2)
    st.markdown("---")
    st.subheader("Financial")
    user_inputs["Credit_Limit"] = st.number_input("Credit Limit",1000,35000,5000)
    user_inputs["Total_Revolving_Bal"] = st.number_input("Revolving Balance",0,3000,1200)
    user_inputs["Avg_Open_To_Buy"] = user_inputs["Credit_Limit"] - user_inputs["Total_Revolving_Bal"]
    user_inputs["Total_Amt_Chng_Q4_Q1"] = st.slider("Amount Change Q4/Q1",0.0,3.5,0.7)
    user_inputs["Total_Trans_Amt"] = st.number_input("Total Transaction Amount",500,20000,4000)
    user_inputs["Total_Trans_Ct"] = st.slider("Total Transactions",10,150,60)
    user_inputs["Total_Ct_Chng_Q4_Q1"] = st.slider("Count Change Q4/Q1",0.0,3.5,0.7)
    user_inputs["Avg_Utilization_Ratio"] = st.slider("Utilization Ratio",0.0,1.0,0.2)
    st.markdown("---")
    analyze_button = st.sidebar.button("🔍 Analyze Customer Risk", type="primary")
# ==========================================
# FEATURE ENGINEERING
# ==========================================
user_inputs["Avg_Transaction_Value"] = user_inputs["Total_Trans_Amt"]/(user_inputs["Total_Trans_Ct"]+1)
user_inputs["Engagement_Score"] = (
    user_inputs["Total_Relationship_Count"] +
    user_inputs["Contacts_Count_12_mon"] -
    user_inputs["Months_Inactive_12_mon"]
)
input_df = pd.DataFrame([user_inputs])
# ==========================================
# PREDICTION
# ==========================================
prob = None
risk_group = None
var = None
action = None
with tab2:
    st.subheader("🔍 Customer Risk Prediction")
    if analyze_button:
        prob = model_pipeline.predict_proba(input_df)[:,1][0]
        risk_group,icon = segment_risk(prob)
        var = prob * user_inputs["Credit_Limit"]
        action = recommended_action(risk_group,var)
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.metric("CHURN PROBABILITY",f"{prob:.2%}")
        with col2:
            st.metric("RISK LEVEL",f"{icon} {risk_group}")
        with col3:
            st.metric("VALUE AT RISK",f"₹{var:,.2f}")
        with col4:
            if var > 8000:
                priority_label = "High Value Customer"
            elif var > 3000:
                priority_label = "Medium Value Customer"
            else:
                priority_label = "Low Value Customer"
            st.metric("Customer Value Segment", priority_label)
        st.progress(float(prob))
        st.markdown("---")
# ==========================================
# EXPLAINABILITY
# ==========================================
    with tab3:
        st.subheader("🧠 Feature Importance")
        xgb_model = model_pipeline.named_steps["model"]
        preprocessor = model_pipeline.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        feat_importance = pd.Series(
            xgb_model.feature_importances_,
            index=feature_names
        ).nlargest(10)
        feat_importance.index = feat_importance.index.str.replace("remainder__","")
        feat_importance.index = feat_importance.index.str.replace("cat__","")
        feat_importance = feat_importance.sort_values()
        fig,ax = plt.subplots()
        sns.barplot(
            x=feat_importance.values,
            y=feat_importance.index,
            palette="viridis",
            ax=ax
        )
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        st.pyplot(fig)
        st.write("""
**Insights**
- Transaction count and inactivity strongly influence churn.
- Customers with fewer products show higher churn risk.
- Engagement improvements can reduce churn probability.
""")
# ==========================================
# DECISION SUPPORT
# ==========================================
    with tab4:
        st.subheader("Recommended Action")
        if action is not None:
            st.success(action)
        else:
            st.info("Run risk prediction first.")
        st.markdown("---")
        st.subheader("🔮 Retention Strategy Simulation")
        if prob is not None:
            baseline_pred = prob
        else:
            st.info("Run risk prediction to see decision support.")
            st.stop()
        st.write(f"Baseline Churn Risk: {baseline_pred:.2%}")

        # Scenario 1
        engagement_df = input_df.copy()
        engagement_df["Total_Relationship_Count"] += 1
        engagement_df["Months_Inactive_12_mon"] = max(
           0,
           engagement_df["Months_Inactive_12_mon"].values[0]-1
        )
        engagement_df["Total_Trans_Ct"] += 10
        # recompute engineered features
        engagement_df["Avg_Open_To_Buy"] = engagement_df["Credit_Limit"] - engagement_df["Total_Revolving_Bal"]
        engagement_df["Avg_Transaction_Value"] = (
            engagement_df["Total_Trans_Amt"] /
            (engagement_df["Total_Trans_Ct"] + 1)
        )
        engagement_df["Engagement_Score"] = (
        engagement_df["Total_Relationship_Count"] +
        engagement_df["Contacts_Count_12_mon"] -
        engagement_df["Months_Inactive_12_mon"]
        )
        engagement_risk = model_pipeline.predict_proba(engagement_df)[:,1][0]
        # Scenario 2
        financial_df = input_df.copy()
        financial_df["Credit_Limit"] += 2000
        financial_df["Avg_Open_To_Buy"] = financial_df["Credit_Limit"] - financial_df["Total_Revolving_Bal"]
        financial_df["Avg_Transaction_Value"] = financial_df["Total_Trans_Amt"]/(financial_df["Total_Trans_Ct"]+1)
        financial_df["Engagement_Score"] = (
            financial_df["Total_Relationship_Count"] +
            financial_df["Contacts_Count_12_mon"] -
            financial_df["Months_Inactive_12_mon"]
        )
        financial_risk = model_pipeline.predict_proba(financial_df)[:,1][0]
        # Scenario 3
        activity_df = input_df.copy()
        activity_df["Total_Trans_Ct"] += 20
        activity_df["Total_Trans_Amt"] += 2000
        activity_df["Avg_Transaction_Value"]=activity_df["Total_Trans_Amt"]/(activity_df["Total_Trans_Ct"]+1)
        activity_df["Engagement_Score"] = (
            activity_df["Total_Relationship_Count"] +
            activity_df["Contacts_Count_12_mon"] -
            activity_df["Months_Inactive_12_mon"]
        )
        activity_risk = model_pipeline.predict_proba(activity_df)[:,1][0]
        results = pd.DataFrame({
            "Scenario":["Baseline","Engagement Program","Financial Incentive","Transaction Boost"],
            "Churn Risk":[
                f"{baseline_pred:.2%}",
                f"{engagement_risk:.2%}",
                f"{financial_risk:.2%}",
                f"{activity_risk:.2%}"
            ]
        })
        st.table(results)
        chart_df = pd.DataFrame({
            "Scenario":["Baseline","Engagement Program","Financial Incentive","Transaction Boost"],
            "Risk":[baseline_pred,engagement_risk,financial_risk,activity_risk]
        })
        fig,ax = plt.subplots(figsize=(8,5))
        sns.barplot(
            data=chart_df,
            x="Scenario",
            y="Risk",
            palette="viridis",
            ax=ax
        )
        for i,v in enumerate(chart_df["Risk"]):
            ax.text(i,v+0.005,f"{v:.2%}",ha='center')
        ax.set_title("Retention Strategy Impact on Churn Risk")
        st.pyplot(fig, use_container_width=False)
        strategy_risks={
            "Engagement Program":engagement_risk,
            "Financial Incentive":financial_risk,
            "Transaction Boost":activity_risk
        }
        best_strategy=min(strategy_risks,key=strategy_risks.get)
        best_risk=strategy_risks[best_strategy]
        reduction=(baseline_pred-best_risk)*100
        st.success(f"🏆 Best Strategy: {best_strategy} (Risk Reduction: {reduction:.2f}%)") 