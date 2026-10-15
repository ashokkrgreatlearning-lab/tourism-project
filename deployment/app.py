
import os
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
# 3. DEFINE MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_tourism_model_v1.joblib"
)

# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
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

# ============================================================
# 7. CUSTOMER PURCHASE PREDICTION
# ============================================================

if predict_button:

    try:

        # Match the input columns expected by the model
        expected_features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if expected_features is not None:

            expected_features = list(expected_features)

            # Add index column only if model expects it
            if "__index_level_0__" in expected_features:
                input_data["__index_level_0__"] = 0

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

        # Generate prediction
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("📊 Tourism Package Prediction Result")

        if str(prediction) == "1":

            st.success(
                "✅ Customer is likely to purchase "
                "the Tourism Package."
            )

        else:

            st.warning(
                "⚠️ Customer is unlikely to purchase "
                "the Tourism Package."
            )

        # ----------------------------------------------------
        # PURCHASE PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            classes = list(model.classes_)

            positive_index = next(
                (
                    i for i, value in enumerate(classes)
                    if str(value) == "1"
                ),
                None
            )

            if positive_index is not None:

                probability = model.predict_proba(
                    input_data
                )[0][positive_index]

                st.metric(
                    label="Probability of Purchasing Tourism Package",
                    value=f"{probability:.2%}"
                )

                st.progress(
                    min(max(float(probability), 0.0), 1.0)
                )

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
                        "Low purchase probability. "
                        "Additional engagement may be required."
                    )

        # ----------------------------------------------------
        # DISPLAY INPUT DATA
        # ----------------------------------------------------

        with st.expander("🔍 View Customer Input Details"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(f"Prediction Error: {e}")
        st.exception(e)

# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 Tourism Package Prediction App | "
    "Machine Learning Based Customer Purchase Prediction"
)
