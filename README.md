# 🌿 Cassava Leaf Disease Classification

A deep learning project that classifies cassava leaf diseases from real field photographs using CNN from scratch and EfficientNet-B0 transfer learning. Built on a real-world Kaggle dataset of 21,397 crowdsourced images from Tanzanian farmers.

**🔗 Live demo:** [cassava-leaf-disease-classification-aryanparija.streamlit.app](https://cassava-leaf-disease-classification-aryanparija.streamlit.app/)

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
- Froze all backbone layers, only trained the classifier head (6,405 params out of 4,013,953 total)
- **Result: 71.75% val accuracy** — significant improvement from prior knowledge alone

### Step 4 — Fine-tuning (Full Network)
- Unfroze entire EfficientNet-B0 backbone (4,013,953 trainable params)
- Retrained with StepLR scheduler (lr=1e-3, halved every 5 epochs) to gently update pretrained weights
- **Result: 85.19% val accuracy** ✅

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
| EfficientNet B0 (frozen) | 71.75% | 6,405 |
| EfficientNet B0 (fine-tuned) | **85.19%** | 4,013,953 |

**Per-class breakdown (fine-tuned model):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CBB | 0.58 | 0.57 | 0.57 | 217 |
| CBSD | 0.81 | 0.71 | 0.76 | 438 |
| CGM | 0.80 | 0.69 | 0.74 | 477 |
| CMD | 0.94 | 0.95 | 0.95 | 2,632 |
| Healthy | 0.63 | 0.72 | 0.67 | 516 |
| **Accuracy** | | | **0.85** | 4,280 |
| **Macro avg** | **0.75** | **0.73** | **0.74** | 4,280 |
| **Weighted avg** | **0.85** | **0.85** | **0.85** | 4,280 |

> Macro avg F1 (0.74) is the unbiased metric — it weights all classes equally regardless of CMD dominance. This is the number that matters most given the class imbalance.

---

## 🏆 Leaderboard Context

| Solution | Accuracy |
|---|---|
| Top ensemble (winning solution) | 91.3% |
| Strong single EfficientNet B4 | ~88-89% |
| **This project (EfficientNet B0, no ensembling)** | **85.19%** |
| Naive majority-class baseline | 61.5% |

Top solutions used EfficientNet B4/B5, Vision Transformers, multiple model ensembling, and test-time augmentation. This project uses a single EfficientNet B0 — the gap is explained and expected.

---

## ⚠️ Known Limitations

- **CBB is the weakest class** (F1=0.57) — only 5% of training data and visually subtle symptoms even for human experts
- **Mild overfitting** — train accuracy reached 89.6% vs val 85.2% by the final epoch. Fixable with stronger augmentation, weight decay, or early stopping around epoch 8 (where val accuracy was already 85.02%)
- **Overconfident predictions** — model outputs high confidence (95%+) on many individual predictions. Temperature scaling would calibrate this
- **Single model** — no ensembling, no test-time augmentation
- **Label noise** — the original dataset has documented label noise (diagnosing from photos alone is difficult even for experts)

---

## 🌐 Deployment

The Streamlit app is deployed on **Streamlit Community Cloud** and is fully self-contained — it does **not** depend on the FastAPI service to run. On first load, it downloads the fine-tuned model weights directly from the **Hugging Face Hub** repo [`aryanparija/cassava-leaf-disease-efficientnet-b0`](https://huggingface.co/aryanparija/cassava-leaf-disease-efficientnet-b0) and caches them, so there's no manual weights download step for the deployed version.

The FastAPI service (`app/main.py`) is a separate, independent inference API — useful for programmatic/API access — and is not required for the Streamlit app to function.

---

## 🚀 How to Run Locally

### Option A — Streamlit app only (recommended, simplest)

```bash
git clone https://github.com/aryanparija/Cassava-Leaf-Disease-Classification.git
cd Cassava-Leaf-Disease-Classification
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Model weights are downloaded automatically from Hugging Face Hub on first run — no manual setup needed. App opens at `http://localhost:8501`.

### Option B — FastAPI service (standalone prediction API)

The FastAPI service loads weights from a local file rather than Hugging Face Hub, so you'll need to place the weights manually first:

1. Download `efficientnet_b0_85.pth` from the [Hugging Face Hub repo](https://huggingface.co/aryanparija/cassava-leaf-disease-efficientnet-b0)
2. Place it at `models/efficientnet_b0_85.pth`
3. Install the extra API dependencies not included in the base `requirements.txt`:
   ```bash
   pip install fastapi uvicorn python-multipart
   ```
4. Run the service:
   ```bash
   uvicorn app.main:app --reload
   ```
   API available at `http://127.0.0.1:8000`, interactive docs at `http://127.0.0.1:8000/docs`

**Test the API directly:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@your_image.jpg"
```

> Note: the Streamlit app and FastAPI service are independent of each other — the Streamlit app does **not** call the FastAPI endpoint, each loads and runs its own copy of the model.

---

## 📁 Project Structure

```
cassava-leaf-disease-classification/
├── notebooks/
│   ├── 01_eda.ipynb              
│   └── 02_modelling.ipynb        
├── app/
│   ├── main.py                   
│   └── streamlit_app.py          
├── models/                       
├── data/                         
│   ├── train_images/             
│   ├── train.csv                 
│   └── label_num_to_disease_map.json
├── requirements.txt               
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Model | EfficientNet-B0 (ImageNet pretrained), hosted on Hugging Face Hub |
| Explainability | Grad-CAM |
| API (standalone) | FastAPI, Uvicorn |
| Frontend / Deployed App | Streamlit, Streamlit Community Cloud |
| Data | pandas, numpy, scikit-learn, Pillow |
| Training | Google Colab (T4 GPU) |

---

## 👤 Author

Aryan Parija — [LinkedIn](https://www.linkedin.com/in/aryanparija2006/)
GitHub: [github.com/aryanparija](https://github.com/aryanparija)