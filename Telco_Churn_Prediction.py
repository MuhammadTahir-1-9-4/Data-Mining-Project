import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

st.set_page_config(page_title="Telco Churn App", layout="wide")

# load  dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-customer-churn(cleaned).csv")
    return df

df = load_data()
df.drop('Unnamed: 0', axis=1, inplace=True)

# Encode target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])

# One hot encoding
one_hot_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
df_encoded = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=np.int32)

# Features and target
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# Resample using SMOTEENN
sm = SMOTEENN(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Sidebar
st.sidebar.title("⚙️ Model Configuration")
n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100)
max_depth = st.sidebar.slider("max_depth", 1, 50, 9)
min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 10, 1)

# Train the model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

st.sidebar.markdown("---")
st.sidebar.metric("Train Accuracy", f"{train_acc:.2%}")
st.sidebar.metric("Test Accuracy", f"{test_acc:.2%}")
st.sidebar.caption("🎯 Random Forest Classifier")

# Tabs:  Prediction | Dashboard
tab2, tab1 = st.tabs(["🤖 Prediction","📈 Dashboard" ])

    # --------------- Dashboard Tab ---------------- #
with tab1:
    st.title("📊 Telco Customer Churn Dashboard")
    
    # Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original vs Resampled Data Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Original data
        original_counts = df['Churn'].value_counts()
        resampled_counts = pd.Series(y_resampled).value_counts()
        
        width = 0.35
        x = np.arange(2)
        bars1 = ax.bar(x - width/2, original_counts, width, label='Original', color='#4C72B0')
        bars2 = ax.bar(x + width/2, resampled_counts, width, label='Resampled', color='#DD8452')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['No Churn', 'Churn'])
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Churn Rate by Tenure Group")
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                                   labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
        tenure_churn = df.groupby('tenure_group')['Churn'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=tenure_churn, x='tenure_group', y='Churn', palette='viridis', ax=ax)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel('Churn Rate')
        ax.set_xlabel('Tenure (months)')
        st.pyplot(fig)
    
    # Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Churn by Contract Type")
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        contract_churn.plot(kind='bar', stacked=True, color=['#8BC34A', '#F44336'], ax=ax)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel('Percentage')
        ax.legend(['No Churn', 'Churn'], bbox_to_anchor=(1, 1))
        st.pyplot(fig)
        
    with col4:
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df, x='MonthlyCharges', hue='Churn', bins=30, kde=True, 
                     palette={0: '#8BC34A', 1: '#F44336'}, ax=ax)
        ax.set_xlabel('Monthly Charges ($)')
        st.pyplot(fig)
    
    # Row 3 - Keep your existing visualizations
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(churn_counts, labels=['No', 'Yes'], autopct='%1.1f%%', colors=['#8BC34A', '#F44336'], startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    with col6:
        st.subheader("Internet Service Type")
        st.bar_chart(df['InternetService'].value_counts())
    
    # Row 4 - Additional Visualizations
    st.subheader("Additional Insights")
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("**Churn by Internet Service Type**")
        internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        internet_churn.plot(kind='bar', stacked=True, color=['#8BC34A', '#F44336'], ax=ax)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel('Percentage')
        ax.legend(['No Churn', 'Churn'], bbox_to_anchor=(1, 1))
        st.pyplot(fig)
        
    with col8:
        st.markdown("**Payment Method Distribution**")
        payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        payment_churn.plot(kind='barh', stacked=True, color=['#8BC34A', '#F44336'], ax=ax)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_xlabel('Percentage')
        ax.legend(['No Churn', 'Churn'], bbox_to_anchor=(1, 1))
        st.pyplot(fig)
    
    # Row 5 - Keep your existing scatter plot
    st.markdown("### Monthly Charges vs Tenure")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='tenure', y='MonthlyCharges', hue='Churn', palette='coolwarm', ax=ax2)
    st.pyplot(fig2)
# --------------- Prediction Tab ---------------- #
with tab2:
    st.title("🤖 Predict Customer Churn")
    st.write("Fill in the customer details below:")

    # Input fields
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (Months)', 0, 72, 12)
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
    total_charges = st.number_input('Total Charges', min_value=0.0)
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])

    # Prepare input dictionary
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender_Male': 1 if gender == 'Male' else 0,
        'SeniorCitizen_Yes': 1 if senior_citizen == 'Yes' else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'PhoneService_Yes': 1,
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaperlessBilling_Yes': 1,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'tenure_group_13 - 24': 0,
        'tenure_group_25 - 36': 0,
        'tenure_group_37 - 48': 0,
        'tenure_group_49 - 60': 0,
        'tenure_group_61 - 72': 0,
        'tenure_group_> 72': 0
    }

    # Set correct tenure group
    if tenure <= 12:
        input_data['tenure_group_1 - 12'] = 1
    elif 13 <= tenure <= 24:
        input_data['tenure_group_13 - 24'] = 1
    elif 25 <= tenure <= 36:
        input_data['tenure_group_25 - 36'] = 1
    elif 37 <= tenure <= 48:
        input_data['tenure_group_37 - 48'] = 1
    elif 49 <= tenure <= 60:
        input_data['tenure_group_49 - 60'] = 1
    elif 61 <= tenure <= 72:
        input_data['tenure_group_61 - 72'] = 1
    else:
        input_data['tenure_group_> 72'] = 1

    # Prediction
    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        missing_cols = set(X_train.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X_train.columns]

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to churn.")
        else:
            st.success(f"✅ Customer is not likely to churn.")

        # Feature Importance
        st.markdown("---")
        st.subheader("Top Influencing Factors")
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        st.bar_chart(feature_importance.set_index('Feature'))


import os
os.system('pip freeze > requirements.txt')



