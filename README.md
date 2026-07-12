# 🌿 Cassava Leaf Disease Classification

A deep learning project that classifies cassava leaf diseases from real field photographs using CNN from scratch and EfficientNet-B0 transfer learning. Built on a real-world Kaggle dataset of 21,397 crowdsourced images from Tanzanian farmers.

---

## 📌 Problem Statement

Cassava is the second largest provider of carbohydrates in Africa, feeding over 800 million people. However, viral diseases cause yield losses of up to 100% in severe cases. Diagnosing diseases requires expert knowledge that most small-scale farmers lack.

This project builds an automated classifier that takes a photo of a cassava leaf and predicts which of 5 conditions it has — enabling early intervention by farmers with no agricultural expertise.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) |
| Images | 21,397 real field photos |
| Collected by | Makerere AI Lab — crowdsourced from Tanzanian farmers using regular cameras |
| Annotated by | Agricultural experts at National Crops Resources Research Institute (NaCRRI) |
| Split | 80% train (17,117) / 20% val (4,280) — stratified |

**Class Distribution:**

| Class | Label | Count | % |
|---|---|---|---|
| Cassava Bacterial Blight | CBB | 1,087 | 5.08% |
| Cassava Brown Streak Disease | CBSD | 2,189 | 10.23% |
| Cassava Green Mottle | CGM | 2,386 | 11.15% |
| Cassava Mosaic Disease | CMD | 13,158 | 61.49% |
| Healthy | — | 2,577 | 12.04% |

> ⚠️ Severe class imbalance — CMD dominates at 61.49%. A naive model predicting CMD always would score 61.5% accuracy. We explicitly tracked per-class metrics to go beyond this.

---

## 🧠 Approach

### Step 1 — EDA
- Loaded and verified all 21,397 images
- Confirmed class distribution and severe CMD imbalance
- Visually inspected one sample per class — CMD showed distinct yellow mosaic pattern, CBB/CBSD were visually subtle even to the human eye

### Step 2 — CNN from Scratch
- Built a 4-block convolutional network (32→64→128→256 filters) with BatchNorm and Dropout
- Trained for 10 epochs
- **Result: 61.50% val accuracy** — model collapsed to predicting CMD for every image (majority class shortcut)
- This is expected and by design — proves that 21K images and this level of class imbalance is too hard for a CNN with no prior knowledge

### Step 3 — Transfer Learning (Frozen)
- Loaded EfficientNet-B0 pretrained on ImageNet
- Froze all backbone layers, only trained the classifier head (6,405 params out of 4M total)
- **Result: ~71% val accuracy** — significant improvement from prior knowledge alone

### Step 4 — Fine-tuning (Full Network)
- Unfroze entire EfficientNet-B0 backbone
- Retrained at lower learning rate (lr=1e-4) to gently update pretrained weights
- **Result: 85.28% val accuracy** — target achieved ✅

### Step 5 — Explainability (Grad-CAM)
- Applied Gradient-weighted Class Activation Mapping to visualize what the model looks at
- CMD: model focused on the central mosaic pattern ✅
- CBB: narrow attention on leaf edges — explains why CBB has lowest recall
- Confirms model learned real biological features, not background noise

---

## 📊 Results

| Model | Val Accuracy | Trainable Params |
|---|---|---|
| CNN from Scratch | 61.50% | 25.9M |
| EfficientNet B0 (frozen) | ~71% | 6,405 |
| EfficientNet B0 (fine-tuned) | **85.28%** | 4.0M |

**Per-class breakdown (fine-tuned model):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CBB | 0.53 | 0.60 | 0.56 | 217 |
| CBSD | 0.76 | 0.71 | 0.73 | 438 |
| CGM | 0.80 | 0.72 | 0.75 | 477 |
| CMD | 0.94 | 0.95 | 0.95 | 2,632 |
| Healthy | 0.64 | 0.66 | 0.65 | 516 |
| **Macro avg** | **0.73** | **0.73** | **0.73** | 4,280 |

> Macro avg (0.73) is the unbiased metric — it weights all classes equally regardless of CMD dominance. This is the number that matters.

---

## 🏆 Leaderboard Context

| Solution | Accuracy |
|---|---|
| Top ensemble (winning solution) | 91.3% |
| Strong single EfficientNet B4 | ~88-89% |
| **This project (EfficientNet B0, no ensembling)** | **85.28%** |
| Naive majority-class baseline | 61.5% |

Top solutions used EfficientNet B4/B5, Vision Transformers, multiple model ensembling, and test-time augmentation. This project uses a single EfficientNet B0 — the gap is explained and expected.

---

## ⚠️ Known Limitations

- **CBB is the weakest class** (F1=0.56) — only 5% of training data and visually subtle symptoms even for human experts
- **Mild overfitting** — train accuracy reached 90% vs val 85%. Fixable with stronger augmentation, weight decay, or early stopping at epoch 7
- **Overconfident predictions** — model outputs high confidence (95%+) on many individual predictions. Temperature scaling would calibrate this
- **Single model** — no ensembling, no test-time augmentation
- **Label noise** — the original dataset has documented label noise (diagnosing from photos alone is difficult even for experts)

---

## 🚀 How to Run Locally

### Prerequisites
```bash
git clone https://github.com/aryanparija/Cassava-Leaf-Disease-Classification.git
cd Cassava-Leaf-Disease-Classification
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

> ⚠️ You need the model weights file `efficientnet_b0_85.pth` in the `models/` folder. Download it from [Google Drive](https://drive.google.com/drive/folders/your-link-here) and place it at `models/efficientnet_b0_85.pth`.

### Run FastAPI
```bash
uvicorn app.main:app --reload
```
API will be available at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

### Run Streamlit
Open a new terminal:
```bash
streamlit run app/streamlit_app.py
```
App will open at `http://localhost:8501`

> ⚠️ FastAPI must be running before launching Streamlit — Streamlit calls the FastAPI `/predict` endpoint.

### Test the API directly
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@your_image.jpg"
```

---

## 📁 Project Structure
cassava-leaf-disease-classification/
├── notebooks/
│   ├── 01_eda.ipynb              # EDA, class distribution, sample images
│   └── 02_modelling.ipynb        # CNN scratch, EfficientNet, fine-tuning, Grad-CAM
├── app/
│   ├── main.py                   # FastAPI prediction endpoint
│   └── streamlit_app.py          # Streamlit frontend UI
├── models/                       # Model weights (not tracked in git — see above)
├── data/                         # Dataset (not tracked in git)
│   ├── train_images/             # 21,397 images
│   ├── train.csv                 # image_id + label
│   └── label_num_to_disease_map.json
├── .gitignore
└── README.md

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Model | EfficientNet-B0 (ImageNet pretrained) |
| Explainability | Grad-CAM |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Data | pandas, numpy, scikit-learn, Pillow |
| Training | Google Colab (T4 GPU) |

---

## 👤 Author

**Aryan Parija** — (https://www.linkedin.com/in/aryanparija2006/) [Linkedin]
GitHub: [github.com/aryanparija](https://github.com/aryanparija)