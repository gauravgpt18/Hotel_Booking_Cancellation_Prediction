"""Hotel Booking Cancellation Predictor (Streamlit).

Files expected next to this script:
    Hotel_Cancellation_model_prod.pkl   -> dict {"model": KNN, "OHE": OneHotEncoder, "LE": LabelEncoder}
    booking.csv                         -> original dataset (used for the insights tab)
"""
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Hotel_Cancellation_model_prod.pkl"
DATA_PATH = BASE_DIR / "booking.csv"

# --------------------------------------------------------------------------
# Feature definitions (CSV column names)
# --------------------------------------------------------------------------
CAT_COLS = ["type of meal", "room type", "market segment type"]
NUM_COLS = [
    "number of adults",
    "number of children",
    "number of weekend nights",
    "number of week nights",
    "car parking space",
    "lead time",
    "repeated",
    "P-C",
    "P-not-C",
    "average price",
    "special requests",
]
REQUIRED_COLS = CAT_COLS + NUM_COLS

# The pickle does not contain the StandardScaler that was used during training,
# so its parameters (fit on the 80 % training split) are stored here.
SCALER_MEAN = np.array([
    1.8456662532727022, 0.10455422350833682, 0.8128014330990769,
    2.2097285379633473, 0.031211244315832987, 85.4360272840037,
    0.024493592393551054, 0.023873501446878876, 0.1505443020531905,
    103.43940574617612, 0.6168871434477057,
])
SCALER_SCALE = np.array([
    0.518188697754022, 0.3968526242349167, 0.8724016122653853,
    1.4150071255608132, 0.17388818977748421, 86.06354830147754,
    0.15457573006526487, 0.38265658205107295, 1.7937826246376243,
    34.948496853051644, 0.786153816291951,
])


def to_model_name(col: str) -> str:
    """'number of adults' -> 'number_of_adults', 'P-not-C' -> 'P_not_C'."""
    return col.replace(" ", "_").replace("-", "_")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
    return bundle["model"], bundle["OHE"], bundle["LE"]


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
def featurize(raw: pd.DataFrame, model, ohe) -> pd.DataFrame:
    """Raw booking columns (CSV names) -> 27 model-ready features."""
    df = raw.copy()
    df.columns = [to_model_name(c) for c in df.columns]

    cat_names = [to_model_name(c) for c in CAT_COLS]
    num_names = [to_model_name(c) for c in NUM_COLS]

    cat = pd.DataFrame(
        ohe.transform(df[cat_names]),
        columns=ohe.get_feature_names_out(),
        index=df.index,
    )
    num = pd.DataFrame(
        (df[num_names].astype(float).to_numpy() - SCALER_MEAN) / SCALER_SCALE,
        columns=num_names,
        index=df.index,
    )
    X = pd.concat([cat, num], axis=1)
    return X[list(model.feature_names_in_)]


def cancel_probability(raw: pd.DataFrame, model, ohe, le) -> np.ndarray:
    X = featurize(raw, model, ohe)
    proba = model.predict_proba(X)
    canceled_label = int(le.transform(["Canceled"])[0])
    idx = list(model.classes_).index(canceled_label)
    return proba[:, idx]


def risk_label(p: float) -> str:
    if p >= 0.6:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def tab_single(model, ohe, le):
    meal_opts, room_opts, segment_opts = [list(c) for c in ohe.categories_]

    with st.form("booking_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Guests & stay")
            adults = st.number_input("Number of adults", 0, 4, 2)
            children = st.number_input("Number of children", 0, 10, 0)
            weekend = st.number_input("Weekend nights", 0, 7, 1)
            week = st.number_input("Week nights", 0, 20, 2)
            lead = st.number_input("Lead time (days before arrival)", 0, 500, 60)

        with c2:
            st.subheader("Room & pricing")
            room = st.selectbox("Room type", room_opts)
            meal = st.selectbox("Meal plan", meal_opts)
            price = st.number_input(
                "Average price per night", 0.0, 1000.0, 100.0, step=1.0
            )
            parking = st.selectbox(
                "Car parking space", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            requests = st.number_input("Special requests", 0, 5, 0)

        with c3:
            st.subheader("Guest history")
            segment = st.selectbox("Market segment", segment_opts, index=segment_opts.index("Online") if "Online" in segment_opts else 0)
            repeated = st.selectbox(
                "Repeated guest", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            p_c = st.number_input("Previous cancellations", 0, 20, 0)
            p_not_c = st.number_input("Previous bookings not canceled", 0, 60, 0)

        submitted = st.form_submit_button("Predict cancellation risk", type="primary")

    if not submitted:
        return

    if weekend + week == 0:
        st.warning("Total nights is 0 - please enter at least one night.")
        return

    raw = pd.DataFrame([{
        "number of adults": adults,
        "number of children": children,
        "number of weekend nights": weekend,
        "number of week nights": week,
        "type of meal": meal,
        "car parking space": parking,
        "room type": room,
        "lead time": lead,
        "market segment type": segment,
        "repeated": repeated,
        "P-C": p_c,
        "P-not-C": p_not_c,
        "average price": price,
        "special requests": requests,
    }])

    prob = float(cancel_probability(raw, model, ohe, le)[0])
    level = risk_label(prob)

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Cancellation probability", f"{prob:.0%}")
        st.metric("Risk level", level)
    with r2:
        if prob >= 0.5:
            st.error("This booking is likely to be **CANCELED**.")
        else:
            st.success("This booking is likely to be **KEPT (not canceled)**.")
        st.progress(prob)
        st.caption(
            "The model is a 5-nearest-neighbours classifier, so probabilities "
            "move in steps of 20%."
        )


def tab_batch(model, ohe, le):
    st.write(
        "Upload a CSV with the same columns as `booking.csv` "
        "(`Booking_ID`, dates and `booking status` are optional and ignored)."
    )
    file = st.file_uploader("CSV file", type="csv")
    if file is None:
        return

    df = pd.read_csv(file)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    for col, cats in zip(CAT_COLS, ohe.categories_):
        bad = sorted(set(df[col].dropna().unique()) - set(cats))
        if bad:
            st.error(f"Unknown values in '{col}': {bad}. Allowed: {list(cats)}")
            return

    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    probs = cancel_probability(df[REQUIRED_COLS], model, ohe, le)

    out = df.copy()
    out["cancel_probability"] = probs
    out["prediction"] = np.where(probs >= 0.5, "Canceled", "Not_Canceled")
    out["risk_level"] = [risk_label(p) for p in probs]

    a, b, c = st.columns(3)
    a.metric("Bookings scored", f"{len(out):,}")
    b.metric("Predicted cancellations", f"{(probs >= 0.5).sum():,}")
    c.metric("Predicted cancel rate", f"{(probs >= 0.5).mean():.1%}")

    st.dataframe(out.head(500))
    st.download_button(
        "Download predictions (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def tab_insights():
    df = load_data()
    if df is None:
        st.info("`booking.csv` not found next to app.py - insights are unavailable.")
        return

    df = df.copy()
    df["canceled"] = (df["booking status"] == "Canceled").astype(int)

    a, b, c, d = st.columns(4)
    a.metric("Total bookings", f"{len(df):,}")
    b.metric("Cancellation rate", f"{df['canceled'].mean():.1%}")
    c.metric("Avg lead time", f"{df['lead time'].mean():.0f} days")
    d.metric("Avg price / night", f"{df['average price'].mean():.0f}")

    def rate_by(series, title):
        st.markdown(f"**{title}**")
        st.bar_chart(df.groupby(series, observed=True)["canceled"].mean())

    left, right = st.columns(2)
    with left:
        rate_by("market segment type", "Cancellation rate by market segment")
        rate_by("room type", "Cancellation rate by room type")
    with right:
        buckets = pd.cut(
            df["lead time"],
            [-1, 7, 30, 90, 180, 365, 1000],
            labels=["0-7", "8-30", "31-90", "91-180", "181-365", "366+"],
        )
        rate_by(buckets, "Cancellation rate by lead time (days)")
        rate_by("special requests", "Cancellation rate by number of special requests")

    with st.expander("Preview data"):
        st.dataframe(df.drop(columns="canceled").head(100))


def main():
    st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.caption("Estimate how likely a hotel booking is to be canceled.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH.name}. Put it in the same folder as app.py.")
        st.stop()

    model, ohe, le = load_artifacts()

    with st.sidebar:
        st.header("About")
        st.write(
            "Model: K-Nearest Neighbours (k=5) trained on 29,028 hotel bookings "
            "with 27 encoded features. Accuracy on held-out data is about 80%."
        )
        st.write("Use the tabs to score one booking, score a CSV, or explore the data.")

    t1, t2, t3 = st.tabs(["Single booking", "Batch (CSV)", "Data insights"])
    with t1:
        tab_single(model, ohe, le)
    with t2:
        tab_batch(model, ohe, le)
    with t3:
        tab_insights()


if __name__ == "__main__":
    main()
