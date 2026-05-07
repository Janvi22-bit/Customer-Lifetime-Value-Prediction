import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64

# ================= BACKGROUND FUNCTION =================
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    /* 1. Reset App Position & Background */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    /* 2. Transparent Header & Toolbar (Keeps Deploy/Menu visible) */
    header[data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0) !important;
        background-color: transparent !important;
    }}
    
    header[data-testid="stHeader"] button, 
    header[data-testid="stHeader"] svg,
    [data-testid="stToolbar"] button {{
        color: white !important;
        fill: white !important;
    }}

    /* 3. Fix Three-Dot Menu */
    div[data-testid="stMainMenu"] ul {{
        background-color: white !important;
    }}
    
    div[data-testid="stMainMenu"] span, 
    div[data-testid="stMainMenu"] p, 
    div[data-testid="stMainMenu"] li {{
        color: black !important;
    }}

    /* 4. BUTTON FIX: Predict CLV Button Visibility */
    .stButton button {{
        background-color: #ff4b4b !important; /* Your red color */
        color: white !important;              /* Force text to be white */
        border-radius: 8px !important;
        border: none !important;
        opacity: 1 !important;                 /* Ensure it's not transparent */
    }}

    /* Ensure it stays visible on hover/active */
    .stButton button:hover, .stButton button:active, .stButton button:focus {{
        background-color: #ff3333 !important;
        color: white !important;
    }}

    /* General text coloring */
    h1, h2, h3, h4, h5, h6, p, label {{
        color: white !important;
    }}

    /* Input fields styling */
    input, textarea {{
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }}

    /* Fix: Number input values (R, F, M) → BLACK text */
    div[data-baseweb="input"] input {{
        color: black !important;
        background-color: rgba(255,255,255,0.9) !important;
    }}

    /* Container transparency */
    .block-container {{
        background-color: rgba(0,0,0,0.6);
        padding: 3rem 2rem;
        border-radius: 10px;
    }}

    /* 5. Metric Fix */
    [data-testid="stMetricValue"], 
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] div {{
        color: white !important;
        -webkit-text-fill-color: white !important;
    }}

    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {{
        color: white !important;
    }}
    
    </style>
    """, unsafe_allow_html=True)


# ================= PAGE CONFIG =================
st.set_page_config(page_title="CLV Dashboard", layout="wide")

# ================= APPLY BACKGROUND =================
set_bg("Bg.png")  # use your image path

# ================= LOAD MODELS =================
reg_model = joblib.load('clv_regression_model.pkl')
clf_model = joblib.load('clv_classification_model.pkl')

# ================= TITLE =================
st.title("👨🏻‍💼 Customer Lifetime Value Dashboard")
st.markdown("### 📊 Predict and analyze customer value for smarter decisions")

# ================= TABS =================
tab1, tab2 = st.tabs(["🔮 Prediction", "📈 Insights"])

# ================= TAB 1 =================
with tab1:
    st.header("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency (days)", 1, 365, 30)

    with col2:
        frequency = st.number_input("Frequency", 1, 100, 10)

    with col3:
        monetary = st.number_input("Monetary Value", 10, 10000, 500)

    if st.button("Predict CLV"):

        input_data = np.array([[recency, frequency, monetary]])

        clv = reg_model.predict(input_data)[0]
        segment = clf_model.predict(input_data)[0]

        st.subheader("📊 Results")

        colA, colB = st.columns(2)
        colA.metric("Predicted CLV", f"{round(clv,2)}")
        colB.metric("Segment", ["Low", "Medium", "High"][segment])

        # Recommendation
        if segment == 2:
            st.success("High-value customer → Offer loyalty rewards 🎁")
        elif segment == 1:
            st.info("Medium-value → Use personalized offers 📩")
        else:
            st.warning("Low-value → Improve engagement with discounts 💸")

# ================= TAB 2 =================
with tab2:
    st.header("📈 Real-Time Customer Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.slider("Recency", 1, 365, 30)

    with col2:
        frequency = st.slider("Frequency", 1, 100, 10)

    with col3:
        monetary = st.slider("Monetary", 10, 10000, 500)

    input_data = np.array([[recency, frequency, monetary]])

    clv = reg_model.predict(input_data)[0]
    segment = clf_model.predict(input_data)[0]

    colA, colB = st.columns(2)
    colA.metric("Predicted CLV", f"{round(clv,2)}")
    colB.metric("Segment", ["Low", "Medium", "High"][segment])

    features = ['Recency', 'Frequency', 'Monetary']
    values = [recency, frequency, monetary]

    norm_values = [
        recency/365,
        frequency/100,
        monetary/10000
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Feature Comparison")
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(features, values)
        ax.tick_params(colors='white')
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
        st.pyplot(fig)

    with col2:
        st.subheader("📉 Normalized View")
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.bar(features, norm_values)
        ax2.tick_params(colors='white')
        ax2.set_facecolor('none')
        fig2.patch.set_alpha(0)
        st.pyplot(fig2)

    st.subheader("💡 Insights")

    if frequency > 30:
        st.success("High engagement customer")
    elif frequency > 10:
        st.info("Moderate engagement")
    else:
        st.warning("Low engagement → Risk")

    if recency < 30:
        st.success("Recently active")
    else:
        st.warning("Inactive customer")

    if monetary > 3000:
        st.success("High spender")
    else:
        st.info("Growth potential")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")