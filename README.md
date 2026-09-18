# 🩺 Breast Cancer Detection Using ResNet-50 V2

<img width="1917" height="1005" alt="Screenshot 2026-09-18 143520" src="https://github.com/user-attachments/assets/cffa41f4-651d-407b-b466-0e386eeeeb68" />


A deep learning project that classifies breast scan images as **Benign** or **Malignant** using a ResNet-50 V2 backbone (transfer learning), with a Streamlit app for interactive testing and Grad-CAM visual explanations.

> ⚠️ **Disclaimer:** This project is for research/portfolio purposes only. It is **not** a diagnostic tool and should never be used for real medical decisions. Always consult a qualified radiologist.

---

## Features

- Transfer learning on **ResNet-50 V2** (ImageNet-pretrained backbone)
- Binary classification: Benign vs. Malignant
- Interactive **Streamlit** web app for testing the trained model
- **Grad-CAM** heatmaps to visualize which regions of the scan influenced the prediction
- Dark-themed UI with live model stats (size, parameter count) and adjustable heatmap opacity

---

## Project Structure

```
.
├── app.py                 # Streamlit app (upload image → prediction + Grad-CAM)
├── requirements.txt       # Python dependencies
├── cancer_model.h5        # Trained model (not tracked in git — see .gitignore)
├── .gitignore
└── README.md
```

---

## Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd "Breast Cancer Detection Using ResNet-50 V2"
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your trained model file (`cancer_model.h5`) in the project folder. It's excluded from git via `.gitignore` because of its size — download/train it separately.

---

## Running the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload a breast scan image, and click **Run Prediction** to see the classification result and Grad-CAM heatmap.

---

## Model Details

| Item | Value |
|---|---|
| Backbone | ResNet-50 V2 (ImageNet pretrained) |
| Task | Binary classification (Benign / Malignant) |
| Input | RGB image, resized to the model's expected input shape |
| Explainability | Grad-CAM on the backbone's last convolutional layer |

> Fill in your actual training details here once finalized: dataset used, number of training/validation samples, epochs, accuracy/precision/recall/F1, and any augmentation strategy.

---

## Notes

- The app auto-detects the model's expected input shape and channel count at load time, so it adapts if you retrain with a different input size.
- Grad-CAM automatically locates the last convolutional layer, including when the ResNet-50 V2 backbone is nested as a sub-model inside a `Sequential` or `Functional` wrapper.
- This is a demo/testing tool, not a production inference service — there's no authentication, batching, or deployment hardening here.

---

## License

Add your preferred license here (e.g. MIT).
