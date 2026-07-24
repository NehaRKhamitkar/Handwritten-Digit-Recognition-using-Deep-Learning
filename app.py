"""
app.py
-------
Streamlit app: Handwritten Digit Recognition with Deep Learning (CNN).

Two live input modes:
  1. Draw  - draw a digit with your mouse/finger on a canvas
  2. Webcam - show a handwritten digit (paper) to your webcam

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this folder (including digit_model.h5) to a GitHub repo and
    point share.streamlit.io at app.py. See README.md for details.
"""

import numpy as np
import cv2
from PIL import Image
import streamlit as st
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

from preprocessing import preprocess_to_mnist

MODEL_PATH = "digit_model.h5"

st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✏️",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model_input, model):
    probs = model.predict(model_input, verbose=0)[0]
    pred_digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return pred_digit, confidence, probs


def show_result(pred_digit, confidence, probs, processed_img):
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(processed_img, width=140, caption="What the model sees (28x28)")

    with col2:
        st.metric("Prediction", pred_digit, f"{confidence * 100:.1f}% confidence")

    df = pd.DataFrame({"digit": list(range(10)), "probability": probs})
    df = df.set_index("digit")
    st.bar_chart(df)

    if confidence < 0.6:
        st.warning(
            "Low confidence — try drawing the digit bigger, thicker, "
            "and more centered for a better result."
        )


def main():
    st.title("✏️ Handwritten Digit Recognition")
    st.caption(
        "A CNN trained on MNIST predicts your digit in real time — "
        "draw it or show it to your webcam."
    )

    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"Could not load '{MODEL_PATH}'. Run `python train_model.py` "
            f"first to generate the model file, then place it next to app.py.\n\n"
            f"Details: {e}"
        )
        st.stop()

    tab_draw, tab_webcam = st.tabs(["🖌️ Draw", "📷 Webcam"])

    # ---------------------------------------------------------------
    # TAB 1: DRAW
    # ---------------------------------------------------------------
    with tab_draw:
        st.write("Draw a single digit (0-9) below, then click **Predict**.")

        canvas_col, controls_col = st.columns([2, 1])

        with controls_col:
            stroke_width = st.slider("Brush size", 8, 30, 18, key="stroke_w")
            if st.button("🗑️ Clear canvas", use_container_width=True):
                st.session_state["canvas_key"] = st.session_state.get("canvas_key", 0) + 1

        with canvas_col:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.get('canvas_key', 0)}",
            )

        predict_clicked = st.button("🔮 Predict", type="primary", key="predict_draw")

        if predict_clicked:
            if canvas_result.image_data is None or canvas_result.image_data[..., :3].sum() == 0:
                st.info("Please draw a digit first.")
            else:
                rgba = canvas_result.image_data.astype(np.uint8)
                gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)

                # Canvas is already white strokes on black -> already_white_on_black=True
                model_input, processed_img = preprocess_to_mnist(
                    gray, already_white_on_black=True
                )
                pred_digit, confidence, probs = predict(model_input, model)
                show_result(pred_digit, confidence, probs, processed_img)

    # ---------------------------------------------------------------
    # TAB 2: WEBCAM
    # ---------------------------------------------------------------
    with tab_webcam:
        st.write(
            "Write a large digit on plain paper with a dark pen, then hold "
            "it up to the camera and take a photo. Good, even lighting and "
            "a digit that fills most of the frame works best."
        )

        cam_image = st.camera_input("Show a digit to the webcam", key="cam")

        if cam_image is not None:
            pil_img = Image.open(cam_image).convert("L")
            gray = np.array(pil_img)

            # A normal photo is dark ink on a light/white page ->
            # already_white_on_black=False so it gets inverted before thresholding
            model_input, processed_img = preprocess_to_mnist(
                gray, already_white_on_black=False
            )
            pred_digit, confidence, probs = predict(model_input, model)
            show_result(pred_digit, confidence, probs, processed_img)

    st.divider()
    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            - **Model**: Convolutional Neural Network trained on the MNIST dataset (~99% test accuracy)
            - **Framework**: TensorFlow / Keras
            - **Input pipeline**: Both the canvas and webcam images are converted to
              grayscale, thresholded (Otsu), cropped to the digit's bounding box,
              resized to 20x20 preserving aspect ratio, padded onto a 28x28 canvas,
              and re-centered by center-of-mass — matching how MNIST itself was built.
            - **Tip**: A single, bold, centered digit works far better than
              small or off-center writing.
            """
        )


if __name__ == "__main__":
    main()
