
import os
import json
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Package Prediction App",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# 2. APPLICATION HEADER AND TOURISM IMAGE
# ============================================================

st.title("🌍 Tourism Package Purchase Prediction App")

st.markdown(
    "### ✈️ Predict Customer Interest in Tourism Packages"
)

st.image(
    "https://images.unsplash.com/photo-1488646953014-85cb44e25828",
    caption="Explore the World with Tourism Packages",
    use_container_width=True
)

st.write("""
This Machine Learning application predicts the probability of
a customer purchasing a Tourism Package based on demographic
information, travel preferences, and sales interaction details.

Please enter the customer information below to generate a prediction.
""")

st.divider()

# ============================================================
# 3. DEFINE MODEL AND CONFIGURATION PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_tourism_model_v1.joblib"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "model_config.json"
)

# ============================================================
# 4. LOAD TRAINED MODEL AND CONFIGURATION
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_data
def load_config():

    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"Model configuration not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return config


try:
    model = load_model()
    config = load_config()

    classification_threshold = float(
        config["classification_threshold"]
    )

    if not 0 <= classification_threshold <= 1:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not support predict_proba()."
        )

    st.success("✅ Machine Learning model loaded successfully!")

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.exception(e)
    st.stop()

# ============================================================
# 5. CUSTOMER INPUT FORM
# ============================================================

st.subheader("👤 Customer Information")

with st.form("tourism_prediction_form"):

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        contact_type = st.selectbox(
            "Type of Contact",
            ["Self Enquiry", "Company Invited"]
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        duration_pitch = st.number_input(
            "Duration of Pitch (Minutes)",
            min_value=1,
            max_value=120,
            value=15
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer"
            ]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        persons_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=20,
            value=2
        )

        followups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            max_value=10,
            value=3
        )

        product_pitched = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Standard",
                "Deluxe",
                "Super Deluxe",
                "King"
            ]
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        property_star = st.selectbox(
            "Preferred Property Star",
            [3, 4, 5]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Unmarried", "Married", "Divorced"]
        )

        number_trips = st.number_input(
            "Number of Trips",
            min_value=0,
            max_value=30,
            value=3
        )

        passport = st.selectbox(
            "Passport",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        satisfaction = st.selectbox(
            "Pitch Satisfaction Score",
            [1, 2, 3, 4, 5]
        )

        own_car = st.selectbox(
            "Own Car",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=10,
            value=1
        )

        designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1000,
            max_value=1000000,
            value=25000,
            step=1000
        )

    st.divider()

    predict_button = st.form_submit_button(
        "🔮 Predict Tourism Package Purchase",
        use_container_width=True
    )

# ============================================================
# 6. PREPARE INPUT DATA
# ============================================================

if predict_button:

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact_type,
        "CityTier": city_tier,
        "DurationOfPitch": duration_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons_visiting,
        "NumberOfFollowups": followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_trips,
        "Passport": passport,
        "PitchSatisfactionScore": satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    # ========================================================
    # 7. GENERATE PREDICTION
    # ========================================================

    try:

        # Match features used during training
        expected_features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if expected_features is not None:

            expected_features = list(expected_features)

            missing_features = [
                feature
                for feature in expected_features
                if feature not in input_data.columns
            ]

            if missing_features:
                raise ValueError(
                    f"Missing model features: {missing_features}"
                )

            input_data = input_data[expected_features]

        # Identify positive class
        classes = list(model.classes_)

        positive_index = next(
            (
                i for i, value in enumerate(classes)
                if str(value) == "1"
            ),
            None
        )

        if positive_index is None:
            raise ValueError(
                "Positive target class 1 not found in model."
            )

        # Calculate probability
        probability = float(
            model.predict_proba(input_data)[0][positive_index]
        )

        # Apply trained classification threshold
        prediction = int(
            probability >= classification_threshold
        )

        # ====================================================
        # 8. DISPLAY PREDICTION RESULTS
        # ====================================================

        st.divider()

        st.subheader("📊 Tourism Package Prediction Result")

        if prediction == 1:

            st.success(
                "✅ Customer is likely to purchase "
                "the Tourism Package."
            )

        else:

            st.warning(
                "⚠️ Customer is unlikely to purchase "
                "the Tourism Package."
            )

        # Display probability and threshold
        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Probability of Purchasing Tourism Package",
                f"{probability:.2%}"
            )

        with result_col2:

            st.metric(
                "Classification Threshold",
                f"{classification_threshold:.0%}"
            )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )

        # ====================================================
        # 9. CUSTOMER PURCHASE PROBABILITY ANALYSIS
        # ====================================================

        if probability >= 0.75:

            st.info(
                "High purchase probability. "
                "Customer may be a strong potential buyer."
            )

        elif probability >= 0.50:

            st.info(
                "Moderate purchase probability. "
                "Customer may benefit from follow-up."
            )

        else:

            st.info(
                "Lower purchase probability. "
                "Additional engagement may be required."
            )

        # ====================================================
        # 10. DISPLAY CUSTOMER INPUT DETAILS
        # ====================================================

        with st.expander("🔍 View Customer Input Details"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Prediction Error: {e}")
        st.exception(e)

# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 Tourism Package Prediction App | "
    "Machine Learning Based Customer Purchase Prediction"
)
