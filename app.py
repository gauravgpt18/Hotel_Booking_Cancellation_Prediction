import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# SESSION STATE DEFAULTS
# --------------------------------------------------
DEFAULT_STATE = {
    "active_page": "Predictor",
    "prediction_result": None,
    "predicted_inputs": None,
    "inp_adults": 2,
    "inp_children": 0,
    "inp_wknd": 1,
    "inp_wk": 2,
    "inp_meal": "Meal Plan 1",
    "inp_room": "Room_Type 1",
    "inp_segment": "Online",
    "inp_lead": 45,
    "inp_price": 100.0,
    "inp_requests": 1,
    "inp_parking": "No",
    "inp_repeated": "No",
    "inp_pc": 0,
    "inp_pnotc": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def set_page(page_name: str):
    st.session_state.active_page = page_name


# --------------------------------------------------
# THEME-ADAPTIVE CSS (ENHANCED DARK + NATIVE STREAMLIT THEME SUPPORT)
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Keep Streamlit's top-right Settings menu visible for built-in Dark/Light mode */
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Main canvas: enhances Dark & Light themes smoothly using Streamlit CSS variables */
    .stApp {
        background-color: var(--background-color);
        background-image:
            radial-gradient(circle at 88% 8%, rgba(56, 189, 248, 0.13), transparent 38%),
            radial-gradient(circle at 10% 92%, rgba(99, 102, 241, 0.14), transparent 42%);
        background-attachment: fixed;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1260px;
    }

    /* Bordered Cards (Navbar, Header, Form Sections, Result Panel) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--secondary-background-color);
        background-color: color-mix(in srgb, var(--secondary-background-color) 78%, var(--background-color) 22%);
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent) !important;
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.18);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(56, 189, 248, 0.55) !important;
        box-shadow: 0 10px 28px rgba(14, 165, 233, 0.12);
    }

    /* Form Inputs & Selectboxes (Rounded + Theme-Safe) */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        border-color: color-mix(in srgb, var(--text-color) 18%, transparent) !important;
    }

    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }

    /* Primary Button: Refined Royal Indigo -> Cyan Gradient */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 11px !important;
        padding: 0.62rem 1.2rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
    }

    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        filter: brightness(1.07);
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.45) !important;
    }

    /* Secondary Button: Adapts cleanly to both Dark & Light modes */
    div.stButton > button[kind="secondary"] {
        background-color: color-mix(in srgb, var(--secondary-background-color) 85%, transparent) !important;
        color: var(--text-color) !important;
        border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent) !important;
        border-radius: 11px !important;
        padding: 0.56rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: #38bdf8 !important;
        background-color: color-mix(in srgb, #38bdf8 12%, var(--secondary-background-color) 88%) !important;
    }

    /* Metric Boxes with Sleek Accent Bar */
    [data-testid="stMetric"] {
        background-color: color-mix(in srgb, var(--background-color) 65%, var(--secondary-background-color) 35%);
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 0.85rem 1rem;
    }

    [data-testid="stMetricLabel"] p {
        font-weight: 600 !important;
        opacity: 0.85;
    }

    [data-testid="stMetricValue"] div {
        font-weight: 800 !important;
    }

    /* Native Table: Clean Theme-Adaptive Styling */
    [data-testid="stTable"] table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
    }

    [data-testid="stTable"] thead tr th {
        background-color: color-mix(in srgb, var(--secondary-background-color) 88%, #2563eb 12%) !important;
        color: var(--text-color) !important;
        font-weight: 700 !important;
        border-bottom: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent) !important;
        text-align: left !important;
        padding: 0.55rem 0.85rem !important;
    }

    [data-testid="stTable"] tbody tr td {
        background-color: color-mix(in srgb, var(--background-color) 75%, var(--secondary-background-color) 25%) !important;
        color: var(--text-color) !important;
        border-bottom: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent) !important;
        padding: 0.5rem 0.85rem !important;
    }

    [data-testid="stTable"] tbody tr:nth-child(even) td {
        background-color: color-mix(in srgb, var(--secondary-background-color) 60%, var(--background-color) 40%) !important;
    }

    /* Hide numeric row index column cleanly */
    [data-testid="stTable"] thead tr th:first-child,
    [data-testid="stTable"] tbody tr th {
        display: none;
    }

    /* Alert Containers */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* Cancellation Risk Progress Bar */
    [data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #f59e0b 55%, #ef4444 100%) !important;
    }

    /* Subtle Dividers */
    hr {
        border-color: color-mix(in srgb, var(--text-color) 14%, transparent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# MODEL CONSTANTS (STANDALONE - NO CSV REQUIRED)
# --------------------------------------------------
NUM_COLS = [
    "number_of_adults",
    "number_of_children",
    "number_of_weekend_nights",
    "number_of_week_nights",
    "car_parking_space",
    "lead_time",
    "repeated",
    "P_C",
    "P_not_C",
    "average_price",
    "special_requests",
]

TRAIN_MEANS = np.array([
    1.8456662537,
    0.1045542252,
    0.8128014398,
    2.2097285504,
    0.0312112451,
    85.4360349831,
    0.0244935926,
    0.0238735003,
    0.1505443036,
    103.4394057462,
    0.6168871422,
])

TRAIN_STDS = np.array([
    0.5181886979,
    0.3968526253,
    0.8724016150,
    1.4150071298,
    0.1738881896,
    86.0635562150,
    0.1545757301,
    0.3826565832,
    1.7937826560,
    34.9484968531,
    0.7861538187,
])

MEAL_OPTIONS = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
ROOM_OPTIONS = [
    "Room_Type 1",
    "Room_Type 2",
    "Room_Type 3",
    "Room_Type 4",
    "Room_Type 5",
    "Room_Type 6",
    "Room_Type 7",
]
SEGMENT_OPTIONS = ["Online", "Offline", "Corporate", "Complementary", "Aviation"]


# --------------------------------------------------
# LOAD MODEL BUNDLE SAFELY
# --------------------------------------------------
@st.cache_resource
def load_model_bundle(filename: str = "Hotel_Cancellation_model_prod.pkl"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            bundle = joblib.load(path)
            return bundle["model"], bundle["OHE"], bundle["LE"]

    raise FileNotFoundError(
        f"'{filename}' not found. Please place it in the same folder as this script ({script_dir})."
    )


try:
    model, ohe, le = load_model_bundle()
except Exception as err:
    st.error(f"Startup Error: {err}")
    st.stop()


# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------
def predict_booking(inputs: dict) -> dict:
    cat_df = pd.DataFrame(
        [[inputs["type_of_meal"], inputs["room_type"], inputs["market_segment_type"]]],
        columns=["type_of_meal", "room_type", "market_segment_type"],
    )
    ohe_encoded = ohe.transform(cat_df)
    ohe_cols = ohe.get_feature_names_out(["type_of_meal", "room_type", "market_segment_type"])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_cols)

    raw_num = np.array(
        [[
            inputs["number_of_adults"],
            inputs["number_of_children"],
            inputs["number_of_weekend_nights"],
            inputs["number_of_week_nights"],
            inputs["car_parking_space"],
            inputs["lead_time"],
            inputs["repeated"],
            inputs["p_c"],
            inputs["p_not_c"],
            inputs["average_price"],
            inputs["special_requests"],
        ]],
        dtype=float,
    )
    scaled_num = (raw_num - TRAIN_MEANS) / TRAIN_STDS
    num_df = pd.DataFrame(scaled_num, columns=NUM_COLS)

    features = pd.concat([ohe_df, num_df], axis=1)[list(model.feature_names_in_)]

    pred_idx = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    label = le.inverse_transform([pred_idx])[0]

    classes = list(le.classes_)
    cancel_prob = float(proba[classes.index("Canceled")] * 100)
    not_cancel_prob = float(proba[classes.index("Not_Canceled")] * 100)

    return {
        "label": label,
        "cancel_prob": cancel_prob,
        "not_cancel_prob": not_cancel_prob,
    }


# --------------------------------------------------
# TOP NAVIGATION BAR
# --------------------------------------------------
with st.container(border=True):
    nav_title, nav_pred, nav_specs, nav_guide = st.columns([4.2, 1.3, 1.4, 1.3])

    with nav_title:
        st.markdown("### 🏨 Hotel Booking Cancellation Predictor")

    with nav_pred:
        st.button(
            "Predictor",
            on_click=set_page,
            args=("Predictor",),
            type="primary" if st.session_state.active_page == "Predictor" else "secondary",
            use_container_width=True,
        )

    with nav_specs:
        st.button(
            "Model Details",
            on_click=set_page,
            args=("Model Details",),
            type="primary" if st.session_state.active_page == "Model Details" else "secondary",
            use_container_width=True,
        )

    with nav_guide:
        st.button(
            "User Guide",
            on_click=set_page,
            args=("User Guide",),
            type="primary" if st.session_state.active_page == "User Guide" else "secondary",
            use_container_width=True,
        )


# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------
with st.container(border=True):
    if st.session_state.active_page == "Predictor":
        st.subheader("Reservation Cancellation Risk Evaluation")
        st.caption(
            "Enter the booking details below and click **Predict Booking Status** to run the classification model."
        )
    elif st.session_state.active_page == "Model Details":
        st.subheader("Model Architecture & Feature Pipeline")
        st.caption(
            "Technical summary of the trained K-Nearest Neighbors classifier and preprocessing workflow."
        )
    else:
        st.subheader("Application Guide & Input Definitions")
        st.caption(
            "Quick guide for running predictions and understanding each reservation parameter."
        )


# --------------------------------------------------
# PAGE 1: PREDICTOR
# --------------------------------------------------
if st.session_state.active_page == "Predictor":
    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        # Card 1: Guest & Stay Duration
        with st.container(border=True):
            st.markdown("#### 👥 Guest & Stay Details")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                number_of_adults = st.number_input("Adults", min_value=0, max_value=10, step=1, key="inp_adults")
            with g2:
                number_of_children = st.number_input("Children", min_value=0, max_value=10, step=1, key="inp_children")
            with g3:
                number_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=14, step=1, key="inp_wknd")
            with g4:
                number_of_week_nights = st.number_input("Week Nights", min_value=0, max_value=30, step=1, key="inp_wk")

        # Card 2: Room, Meal & Channel
        with st.container(border=True):
            st.markdown("#### 🛎️ Room, Meal & Market Segment")
            r1, r2, r3 = st.columns(3)
            with r1:
                type_of_meal = st.selectbox("Meal Plan", MEAL_OPTIONS, key="inp_meal")
            with r2:
                room_type = st.selectbox("Room Type", ROOM_OPTIONS, key="inp_room")
            with r3:
                market_segment_type = st.selectbox("Market Segment", SEGMENT_OPTIONS, key="inp_segment")

        # Card 3: Pricing, Lead Time & History
        with st.container(border=True):
            st.markdown("#### 💳 Pricing, Lead Time & Guest History")
            p1, p2, p3 = st.columns(3)
            with p1:
                lead_time = st.slider("Lead Time (Days)", min_value=0, max_value=450, key="inp_lead")
                average_price = st.number_input("Average Price ($)", min_value=0.0, max_value=600.0, step=5.0, key="inp_price")
            with p2:
                special_requests = st.selectbox("Special Requests", [0, 1, 2, 3, 4, 5], key="inp_requests")
                car_parking_choice = st.selectbox("Car Parking Space", ["No", "Yes"], key="inp_parking")
                car_parking_space = 1 if car_parking_choice == "Yes" else 0
            with p3:
                repeated_choice = st.selectbox("Repeated Guest", ["No", "Yes"], key="inp_repeated")
                repeated = 1 if repeated_choice == "Yes" else 0
                p_c = st.number_input("Prev. Canceled (P-C)", min_value=0, max_value=20, step=1, key="inp_pc")
                p_not_c = st.number_input("Prev. Not-Canceled (P-not-C)", min_value=0, max_value=60, step=1, key="inp_pnotc")

        # Action Buttons
        btn_col1, btn_col2 = st.columns([3, 1])
        with btn_col1:
            predict_clicked = st.button("Predict Booking Status", type="primary", use_container_width=True)
        with btn_col2:
            clear_clicked = st.button("Clear Result", type="secondary", use_container_width=True)

    current_inputs = {
        "number_of_adults": int(number_of_adults),
        "number_of_children": int(number_of_children),
        "number_of_weekend_nights": int(number_of_weekend_nights),
        "number_of_week_nights": int(number_of_week_nights),
        "type_of_meal": type_of_meal,
        "room_type": room_type,
        "market_segment_type": market_segment_type,
        "lead_time": int(lead_time),
        "average_price": float(average_price),
        "special_requests": int(special_requests),
        "car_parking_space": int(car_parking_space),
        "repeated": int(repeated),
        "p_c": int(p_c),
        "p_not_c": int(p_not_c),
    }

    if clear_clicked:
        st.session_state.prediction_result = None
        st.session_state.predicted_inputs = None

    if predict_clicked:
        st.session_state.prediction_result = predict_booking(current_inputs)
        st.session_state.predicted_inputs = current_inputs.copy()

    # Right Panel: Displays prediction ONLY on button click
    with right_col:
        with st.container(border=True):
            st.markdown("#### 📊 Prediction Result")

            if st.session_state.prediction_result is None:
                st.info(
                    "Click **Predict Booking Status** after entering reservation details to view the prediction."
                )
            elif st.session_state.predicted_inputs != current_inputs:
                st.warning(
                    "Booking inputs have changed. Click **Predict Booking Status** to compute the updated result."
                )
            else:
                res = st.session_state.prediction_result
                pred_label = res["label"]
                cancel_prob = res["cancel_prob"]
                not_cancel_prob = res["not_cancel_prob"]

                if pred_label == "Canceled":
                    st.error(f"⚠️ High Cancellation Risk — Predicted Status: {pred_label}")
                else:
                    st.success(f"✅ Low Cancellation Risk — Predicted Status: {pred_label}")

                m1, m2 = st.columns(2)
                m1.metric("Cancellation Risk", f"{cancel_prob:.1f}%")
                m2.metric("Check-In Likelihood", f"{not_cancel_prob:.1f}%")

                st.write("**Risk Meter**")
                st.progress(int(round(cancel_prob)))

            st.divider()
            st.markdown("**Reservation Summary**")
            total_nights = number_of_weekend_nights + number_of_week_nights
            summary_df = pd.DataFrame(
                {
                    "Parameter": [
                        "Guests",
                        "Stay Length",
                        "Lead Time",
                        "Room & Meal",
                        "Market Segment",
                        "Room Rate",
                        "Special Requests & Parking",
                    ],
                    "Value": [
                        f"{number_of_adults} Adults, {number_of_children} Children",
                        f"{total_nights} Nights ({number_of_weekend_nights} Wknd / {number_of_week_nights} Wk)",
                        f"{lead_time} Days",
                        f"{room_type} | {type_of_meal}",
                        market_segment_type,
                        f"${average_price:.2f} / night",
                        f"{special_requests} Requests | Parking: {car_parking_choice}",
                    ],
                }
            )
            st.table(summary_df)


# --------------------------------------------------
# PAGE 2: MODEL DETAILS
# --------------------------------------------------
elif st.session_state.active_page == "Model Details":
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        with st.container(border=True):
            st.markdown("#### Classifier Specifications")
            specs_df = pd.DataFrame(
                {
                    "Attribute": ["Algorithm", "Neighbors (k)", "Distance Metric", "Weighting", "Output Classes"],
                    "Details": [
                        "K-Nearest Neighbors (KNeighborsClassifier)",
                        "5",
                        "Euclidean (Minkowski, p=2)",
                        "Uniform",
                        "Canceled (0), Not_Canceled (1)",
                    ],
                }
            )
            st.table(specs_df)

    with col_b:
        with st.container(border=True):
            st.markdown("#### Feature Engineering Pipeline")
            pipe_df = pd.DataFrame(
                {
                    "Stage": ["Categorical Features", "Numerical Features", "Total Input Columns", "Model Bundle"],
                    "Details": [
                        "One-Hot Encoded (16 columns via saved OHE)",
                        "Z-Score Standardized (11 columns)",
                        "27 aligned features",
                        "Hotel_Cancellation_model_prod.pkl (Standalone)",
                    ],
                }
            )
            st.table(pipe_df)


# --------------------------------------------------
# PAGE 3: USER GUIDE
# --------------------------------------------------
else:
    with st.container(border=True):
        st.markdown("#### How to Use This Application")
        st.write("1. Open the **Predictor** section from the top navigation bar.")
        st.write("2. Fill in guest count, stay duration, meal plan, room type, market segment, lead time, price, and prior booking history.")
        st.write("3. Click **Predict Booking Status** to compute the cancellation probability and predicted status.")
        st.write("4. If you change any input afterwards, click **Predict Booking Status** again to refresh the result.")
        st.write("5. To switch between Dark and Light modes, use Streamlit's built-in **⋮ (Settings) → Theme** option in the top-right corner.")

    with st.container(border=True):
        st.markdown("#### Key Field Descriptions")
        guide_df = pd.DataFrame(
            {
                "Field": ["Lead Time", "P-C", "P-not-C", "Special Requests"],
                "Meaning": [
                    "Number of days between the booking date and the arrival date.",
                    "Number of previous bookings canceled by the guest before this reservation.",
                    "Number of previous bookings not canceled by the guest before this reservation.",
                    "Total special requests made by the guest (e.g., high floor, extra bed).",
                ],
            }
        )
        st.table(guide_df)
