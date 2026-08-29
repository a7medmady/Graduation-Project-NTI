import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid rgba(128,128,128,0.25);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.8;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 12px;
}

.metric-card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.25);
    text-align: center;
    margin-bottom: 15px;
}

.metric-card h2 {
    margin: 5px 0;
    font-size: 30px;
}

.metric-card p {
    margin: 0;
    opacity: 0.7;
}

.result-box {
    padding: 28px;
    border-radius: 20px;
    border: 2px solid rgba(128,128,128,0.25);
    margin-top: 25px;
    text-align: center;
}

.result-box h2 {
    font-size: 32px;
}

.small-note {
    text-align: center;
    opacity: 0.65;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# Load Model and Data
# =========================
@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")


@st.cache_data
def load_data():
    return pd.read_excel("ecommerce_customer_churn_cleaned.xlsx")


model = load_model()
df = load_data()


# =========================
# Header
# =========================
st.markdown("""
<div class="hero">

<h1>🛒 E-Commerce Customer Churn Predictor</h1>

<p>
Machine Learning application for predicting customer churn
using a Random Forest Classification Model.
</p>

</div>
""", unsafe_allow_html=True)


# =========================
# Model Performance Cards
# =========================
st.markdown('<div class="section-title">📊 Model Performance</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <p>Model</p>
        <h2>🌲 Random Forest</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <p>Accuracy</p>
        <h2>91.44%</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <p>Precision</p>
        <h2>92.27%</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <p>ROC-AUC</p>
        <h2>92.42%</h2>
    </div>
    """, unsafe_allow_html=True)


st.divider()


# =========================
# Customer Information
# =========================
st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter the customer's information below and click "
    "Predict Customer Churn."
)


with st.form("customer_form"):

    # -------------------------
    # Personal Information
    # -------------------------
    st.markdown("### 👤 Personal Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Age = st.number_input(
            "Age",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            value=int(df["Age"].median())
        )

    with col2:
        Gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].dropna().unique().tolist())
        )

    with col3:
        Country = st.selectbox(
            "Country",
            sorted(df["Country"].dropna().unique().tolist())
        )

    with col4:
        City = st.selectbox(
            "City",
            sorted(df["City"].dropna().unique().tolist())
        )


    st.divider()


    # -------------------------
    # Engagement
    # -------------------------
    st.markdown("### 📱 Customer Engagement")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Membership_Years = st.number_input(
            "Membership Years",
            min_value=float(df["Membership_Years"].min()),
            max_value=float(df["Membership_Years"].max()),
            value=float(df["Membership_Years"].median())
        )

    with col2:
        Login_Frequency = st.number_input(
            "Login Frequency",
            min_value=float(df["Login_Frequency"].min()),
            max_value=float(df["Login_Frequency"].max()),
            value=float(df["Login_Frequency"].median())
        )

    with col3:
        Session_Duration_Avg = st.number_input(
            "Avg Session Duration",
            min_value=float(df["Session_Duration_Avg"].min()),
            max_value=float(df["Session_Duration_Avg"].max()),
            value=float(df["Session_Duration_Avg"].median())
        )

    with col4:
        Pages_Per_Session = st.number_input(
            "Pages Per Session",
            min_value=float(df["Pages_Per_Session"].min()),
            max_value=float(df["Pages_Per_Session"].max()),
            value=float(df["Pages_Per_Session"].median())
        )


    st.divider()


    # -------------------------
    # Shopping Behavior
    # -------------------------
    st.markdown("### 🛍️ Shopping Behavior")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Cart_Abandonment_Rate = st.number_input(
            "Cart Abandonment Rate",
            min_value=float(df["Cart_Abandonment_Rate"].min()),
            max_value=float(df["Cart_Abandonment_Rate"].max()),
            value=float(df["Cart_Abandonment_Rate"].median())
        )

    with col2:
        Wishlist_Items = st.number_input(
            "Wishlist Items",
            min_value=float(df["Wishlist_Items"].min()),
            max_value=float(df["Wishlist_Items"].max()),
            value=float(df["Wishlist_Items"].median())
        )

    with col3:
        Total_Purchases = st.number_input(
            "Total Purchases",
            min_value=float(df["Total_Purchases"].min()),
            max_value=float(df["Total_Purchases"].max()),
            value=float(df["Total_Purchases"].median())
        )

    with col4:
        Average_Order_Value = st.number_input(
            "Average Order Value",
            min_value=float(df["Average_Order_Value"].min()),
            max_value=float(df["Average_Order_Value"].max()),
            value=float(df["Average_Order_Value"].median())
        )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Days_Since_Last_Purchase = st.number_input(
            "Days Since Last Purchase",
            min_value=float(df["Days_Since_Last_Purchase"].min()),
            max_value=float(df["Days_Since_Last_Purchase"].max()),
            value=float(df["Days_Since_Last_Purchase"].median())
        )

    with col2:
        Discount_Usage_Rate = st.number_input(
            "Discount Usage Rate",
            min_value=float(df["Discount_Usage_Rate"].min()),
            max_value=float(df["Discount_Usage_Rate"].max()),
            value=float(df["Discount_Usage_Rate"].median())
        )

    with col3:
        Returns_Rate = st.number_input(
            "Returns Rate",
            min_value=float(df["Returns_Rate"].min()),
            max_value=float(df["Returns_Rate"].max()),
            value=float(df["Returns_Rate"].median())
        )

    with col4:
        Email_Open_Rate = st.number_input(
            "Email Open Rate",
            min_value=float(df["Email_Open_Rate"].min()),
            max_value=float(df["Email_Open_Rate"].max()),
            value=float(df["Email_Open_Rate"].median())
        )


    st.divider()


    # -------------------------
    # Customer Value & Support
    # -------------------------
    st.markdown("### 💰 Customer Value & Support")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Customer_Service_Calls = st.number_input(
            "Customer Service Calls",
            min_value=float(df["Customer_Service_Calls"].min()),
            max_value=float(df["Customer_Service_Calls"].max()),
            value=float(df["Customer_Service_Calls"].median())
        )

    with col2:
        Product_Reviews_Written = st.number_input(
            "Product Reviews Written",
            min_value=float(df["Product_Reviews_Written"].min()),
            max_value=float(df["Product_Reviews_Written"].max()),
            value=float(df["Product_Reviews_Written"].median())
        )

    with col3:
        Social_Media_Engagement_Score = st.number_input(
            "Social Media Engagement Score",
            min_value=float(df["Social_Media_Engagement_Score"].min()),
            max_value=float(df["Social_Media_Engagement_Score"].max()),
            value=float(df["Social_Media_Engagement_Score"].median())
        )

    with col4:
        Mobile_App_Usage = st.number_input(
            "Mobile App Usage",
            min_value=float(df["Mobile_App_Usage"].min()),
            max_value=float(df["Mobile_App_Usage"].max()),
            value=float(df["Mobile_App_Usage"].median())
        )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Payment_Method_Diversity = st.number_input(
            "Payment Method Diversity",
            min_value=float(df["Payment_Method_Diversity"].min()),
            max_value=float(df["Payment_Method_Diversity"].max()),
            value=float(df["Payment_Method_Diversity"].median())
        )

    with col2:
        Lifetime_Value = st.number_input(
            "Lifetime Value",
            min_value=float(df["Lifetime_Value"].min()),
            max_value=float(df["Lifetime_Value"].max()),
            value=float(df["Lifetime_Value"].median())
        )

    with col3:
        Credit_Balance = st.number_input(
            "Credit Balance",
            min_value=float(df["Credit_Balance"].min()),
            max_value=float(df["Credit_Balance"].max()),
            value=float(df["Credit_Balance"].median())
        )

    with col4:
        Signup_Quarter = st.selectbox(
            "Signup Quarter",
            sorted(df["Signup_Quarter"].dropna().unique().tolist())
        )


    st.divider()


    # =========================
    # Prediction Button
    # =========================
    submitted = st.form_submit_button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )


# =========================
# Prediction
# =========================
if submitted:

    input_data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "Country": Country,
        "City": City,
        "Membership_Years": Membership_Years,
        "Login_Frequency": Login_Frequency,
        "Session_Duration_Avg": Session_Duration_Avg,
        "Pages_Per_Session": Pages_Per_Session,
        "Cart_Abandonment_Rate": Cart_Abandonment_Rate,
        "Wishlist_Items": Wishlist_Items,
        "Total_Purchases": Total_Purchases,
        "Average_Order_Value": Average_Order_Value,
        "Days_Since_Last_Purchase": Days_Since_Last_Purchase,
        "Discount_Usage_Rate": Discount_Usage_Rate,
        "Returns_Rate": Returns_Rate,
        "Email_Open_Rate": Email_Open_Rate,
        "Customer_Service_Calls": Customer_Service_Calls,
        "Product_Reviews_Written": Product_Reviews_Written,
        "Social_Media_Engagement_Score": Social_Media_Engagement_Score,
        "Mobile_App_Usage": Mobile_App_Usage,
        "Payment_Method_Diversity": Payment_Method_Diversity,
        "Lifetime_Value": Lifetime_Value,
        "Credit_Balance": Credit_Balance,
        "Signup_Quarter": Signup_Quarter
    }])


    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]


    st.divider()

    # =========================
    # Result
    # =========================
    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            result_text = "🔴 Likely to Churn"
        else:
            result_text = "🟢 Unlikely to Churn"

        st.markdown(
            f"""
            <div class="result-box">
                <h2>{result_text}</h2>
                <p>Machine Learning Prediction</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col2:

        st.markdown(
            f"""
            <div class="result-box">
                <h2>{probability:.1%}</h2>
                <p>Churn Probability</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =========================
    # Probability Progress Bar
    # =========================
    st.write("### 📈 Churn Risk Level")

    st.progress(float(probability))


    # =========================
    # Recommendation
    # =========================
    if prediction == 1:

        st.error(
            "⚠️ High Churn Risk — This customer may need a retention strategy."
        )

        st.info(
            "💡 Recommended Action: Consider personalized offers, "
            "customer support follow-up, and targeted engagement campaigns."
        )

    else:

        st.success(
            "✅ Low Churn Risk — This customer is currently unlikely to churn."
        )

        st.info(
            "💡 Recommended Action: Maintain engagement and continue "
            "providing a positive customer experience."
        )


# =========================
# Footer
# =========================
st.markdown("""
<div class="small-note">
    E-Commerce Customer Churn Prediction | Machine Learning + Streamlit
</div>
""", unsafe_allow_html=True)