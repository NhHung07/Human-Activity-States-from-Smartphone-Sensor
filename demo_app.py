import io
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


APP_TITLE = "Activity State Demo"
TRAIN_PATH = "./data/train.csv"
SCALER_PATH = "scaler.pkl"
SELECTOR_PATH = "selector.pkl"
MODEL_PATH = "lr_model.pkl"

@st.cache_resource
def load_artifacts():
	if not os.path.exists(TRAIN_PATH):
		raise FileNotFoundError(f"Missing {TRAIN_PATH}")
	if not os.path.exists(SCALER_PATH):
		raise FileNotFoundError(f"Missing {SCALER_PATH}")
	if not os.path.exists(SELECTOR_PATH):
		raise FileNotFoundError(f"Missing {SELECTOR_PATH}")
	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError(f"Missing {MODEL_PATH}")

	train_df = pd.read_csv(TRAIN_PATH)
	feature_cols = [c for c in train_df.columns if c not in ["Activity", "subject"]]

	scaler = joblib.load(SCALER_PATH)
	selector = joblib.load(SELECTOR_PATH)
	model = joblib.load(MODEL_PATH)

	label_encoder = LabelEncoder()
	label_encoder.fit(train_df["Activity"])

	return feature_cols, scaler, selector, model, label_encoder


def prepare_features(raw_df, feature_cols):
	df = raw_df.copy()
	drop_cols = [c for c in ["Activity", "subject"] if c in df.columns]
	if drop_cols:
		df = df.drop(columns=drop_cols)

	missing = [c for c in feature_cols if c not in df.columns]
	extra = [c for c in df.columns if c not in feature_cols]
	if missing:
		raise ValueError("Missing columns: " + ", ".join(missing[:10]))

	df = df.reindex(columns=feature_cols)
	if extra:
		st.warning(
			"Extra columns will be ignored: " + ", ".join(extra[:10])
		)
	return df


def main():
	st.set_page_config(page_title=APP_TITLE, layout="wide")
	st.title(APP_TITLE)
	st.caption("Upload a CSV to get activity predictions.")

	with st.spinner("Loading artifacts..."):
		feature_cols, scaler, selector, model, label_encoder = load_artifacts()

	st.subheader("Upload CSV")
	uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

	if uploaded is None:
		st.info("Upload a CSV file to run predictions.")
		return

	try:
		raw_df = pd.read_csv(uploaded)
	except Exception as exc:
		st.error(f"Failed to read CSV: {exc}")
		return

	st.write("Preview")
	st.dataframe(raw_df.head(100))

	try:
		features = prepare_features(raw_df, feature_cols)
		scaled = scaler.transform(features)
		selected = selector.transform(scaled)
	except Exception as exc:
		st.error(f"Feature preparation failed: {exc}")
		return

	preds = model.predict(selected)
	pred_labels = label_encoder.inverse_transform(preds)

	results = raw_df.copy()
	results["predicted_activity"] = pred_labels

	st.subheader("Predictions")
	st.dataframe(results.head(100))

	csv_bytes = results.to_csv(index=False).encode("utf-8")
	st.download_button(
		"Download predictions CSV",
		data=csv_bytes,
		file_name="predictions.csv",
		mime="text/csv",
	)


if __name__ == "__main__":
	main()
