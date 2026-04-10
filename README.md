# VoiceIQ · Gender Recognition

A deep learning web app that identifies the gender of a speaker from audio input using a CNN trained on mel spectrogram features.

## 🚀 Deploy on Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/voiceiq.git
git push -u origin main
```

> ⚠️ **Include your `gender_model.h5`** file in the repo root before committing (it's under 25MB so it fits in GitHub directly).

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **"Deploy!"**

That's it — Streamlit Cloud will install dependencies from `requirements.txt` automatically.

## 📁 File Structure

```
voiceiq/
├── app.py                  ← Main Streamlit app
├── gender_model.h5         ← Your trained model (add this!)
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Dark theme config
└── .gitignore
```

## 🔬 Tech Stack

- **Streamlit** — Web UI
- **TensorFlow / Keras** — Model inference
- **Librosa** — Audio processing & mel spectrograms
- **Plotly** — Interactive charts

## 📝 Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```
