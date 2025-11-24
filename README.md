# Currency Note Recognition (Indian Notes) - Ready Project

This is a ready-to-deploy Streamlit project for **single-note Indian currency recognition**.
It includes:
- A minimal Keras model (`model.h5`) (saved with random weights if TensorFlow was not available during generation).
- `app.py` - Streamlit inference app
- `src/train.py` - training script (use your own dataset in `data/train` and `data/val`)
- `sample_images/example_note.png` - example image copied from the uploaded file

## How to use (locally)
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place a trained `model.h5` in the repo root (optional; a placeholder is included).
3. Run the app:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud
1. Create a GitHub repo and push these files.
2. Go to https://share.streamlit.io and create a new app, select your GitHub repo, branch `main`, and `app.py` as entrypoint.
3. Deploy.

If you want a genuinely trained model, either run `src/train.py` after preparing dataset folders, or tell me and I will provide a transfer-learning trained `model.h5` built on MobileNetV2 (larger download).
