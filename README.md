# Handwritten Digit Recognition (Streamlit + CNN)

Real-time handwritten digit recognition using a CNN trained on MNIST,
with **two live input modes**:

- 🖌️ **Draw** — sketch a digit with your mouse/finger on an in-browser canvas
- 📷 **Webcam** — hold up a handwritten digit on paper to your camera

## Project structure

```
digit-recognition-app/
├── app.py              # Streamlit app (draw + webcam + prediction UI)
├── preprocessing.py     # MNIST-style image preprocessing (shared by both modes)
├── train_model.py       # Trains the CNN and saves digit_model.h5
├── requirements.txt
└── README.md
```

## 1. Train the model (run once, locally)

```bash
pip install tensorflow numpy
python train_model.py
```

This downloads MNIST, trains a CNN (~10-15 min on CPU, faster on GPU),
and saves **`digit_model.h5`** in the project folder. Expect ~99%
test accuracy.

> You only need to do this once. Commit `digit_model.h5` to your repo
> (it's usually just a few MB) so the deployed app doesn't need to
> retrain.

## 2. Run the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 3. Deploy to Streamlit Community Cloud

1. Push this whole folder (including `digit_model.h5`) to a public
   GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in,
   click **New app**.
3. Select your repo/branch, set **Main file path** to `app.py`.
4. Click **Deploy**.

Notes:
- Webcam access (`st.camera_input`) requires HTTPS, which Streamlit
  Community Cloud provides automatically — it will also prompt the
  visitor's browser for camera permission.
- If your repo/model exceeds GitHub's normal file-size comfort zone,
  use [Git LFS](https://git-lfs.com/) for `digit_model.h5`.
- `tensorflow-cpu` is used in `requirements.txt` to keep the deployed
  image smaller/faster than full `tensorflow`.

## How prediction works

1. Whichever image you provide (canvas or webcam frame) is converted
   to grayscale.
2. Otsu's method automatically finds a good black/white threshold
   (handles varying lighting for webcam shots).
3. The digit's bounding box is found and cropped tightly.
4. It's resized to fit in a 20×20 box (preserving aspect ratio) and
   pasted onto a 28×28 canvas — the same convention MNIST itself
   uses.
5. The digit is re-centered by center-of-mass for the best alignment.
6. The CNN (trained on exactly this format) predicts probabilities
   for digits 0–9; the app shows the top prediction, confidence, and
   a full probability bar chart.

## Tips for best accuracy

- **Draw large and bold** — use a thick brush and fill much of the canvas.
- **Center the digit** — avoid drawing in a corner.
- **Webcam**: good lighting, dark ink on plain light paper, digit filling
  most of the frame, camera roughly perpendicular to the page.
- If confidence is low, the app will tell you — just retry with a
  bigger/clearer digit.

## Customizing / improving further

- Swap in a deeper CNN or transfer-learning backbone in `train_model.py`.
- Add temperature scaling or calibration if you want more trustworthy
  confidence scores.
- Add a "history" panel using `st.session_state` to log past predictions.
- Add multi-digit support by segmenting connected components in
  `preprocessing.py` before feeding each crop to the model.
