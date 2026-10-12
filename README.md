<h1 align="center">🌱 Crop Health AI</h1>

<p align="center">
  AI-powered crop disease screening with explainable results, severity estimation, farm monitoring, and expert verification.
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/Backend-FastAPI-009688">
  <img alt="PyTorch" src="https://img.shields.io/badge/ML-PyTorch-EE4C2C">
  <img alt="Model" src="https://img.shields.io/badge/Model-EfficientNet--B0-purple">
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Supported Classes](#-supported-classes)
- [Machine Learning](#-machine-learning)
- [System Architecture](#-system-architecture)
- [Scan Workflow](#-scan-workflow)
- [Screenshots](#-screenshots)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Train Models](#-train-models)
- [Evaluate Model](#-evaluate-model)
- [Run the Application](#-run-the-application)
- [Run Tests](#-run-tests)
- [User Roles](#-user-roles)
- [Main API Endpoints](#-main-api-endpoints)
- [Planned Crop Assistant](#-planned-crop-assistant)
- [Future Improvements](#-future-improvements)
- [Limitations](#-limitations)
- [Responsible AI](#-responsible-ai)
- [Documentation](#-documentation)
- [Author](#-author)
- [License](#-license)
- [Vision](#-vision)

---

## 🌾 Overview

**Crop Health AI** is a practical crop-health platform that goes beyond simple plant disease classification. It combines computer vision, explainable AI, uncertainty handling, farm monitoring, recommendations, follow-up workflows, and expert verification in one application.

A farmer uploads or captures a leaf photo and receives an AI screening result with a confidence score, severity estimate, Grad-CAM explanation, and structured recommendations. Uncertain or severe cases can be reviewed by an agricultural extension officer.

---

## ✨ Features

- 📷 Leaf image upload and live camera scanning
- 🧠 Disease classification using a fine-tuned EfficientNet-B0 model
- 🔥 Grad-CAM heatmaps that show which leaf regions influenced the prediction
- 🔍 Image quality check before inference
- ❓ Healthy / Diseased / Uncertain outcome with confidence score
- 🌡 Severity and affected-area estimation
- 🌾 Farm and crop plot management
- 📜 Scan history with full result snapshots
- 👨‍🌾 Expert verification workflow (human-in-the-loop)
- 🔁 Follow-up monitoring
- 📚 Structured recommendation system
- 🌍 Multilingual interface (English, Swahili, Hindi, Tamil)
- 📱 Progressive Web App features

---

### 🌡 Severity Estimation

The system estimates:

- Mild
- Moderate
- Severe

It also estimates the approximate affected area percentage.

> **Note:** Severity and affected-area outputs are project-defined image-based estimates and are **not** validated agronomic field severity measurements.

### 🌾 Farm & Plot Management

Farmers can create farms and crop plots.

```text
Farmer
  └── Farm
       ├── Maize Plot
       ├── Potato Plot
       └── Tomato Plot
```

Scans can be linked to specific farms and plots for better tracking.

### 📜 Scan History

Each scan can store:

- Farmer
- Farm
- Plot
- Crop
- Prediction
- Confidence
- Severity
- Affected area
- Timestamp
- Recommendation snapshot
- Verification status
- Grad-CAM output
- Image quality information

### 👨‍🌾 Expert Verification

The system includes an agricultural officer / extension officer workflow.

Experts can review:

- Pending cases
- Low-confidence scans
- Severe scans

Experts can:

- Confirm AI predictions
- Correct AI predictions
- Add expert verification

This creates a human-in-the-loop AI system.

### 🔁 Follow-Up Monitoring

Farmers can schedule follow-up checks after a diagnosis. Future scans can help show whether a crop condition is:

- Improving
- Stable
- Worsening

### 📚 Recommendation System

The system provides structured guidance for:

- Immediate management
- Prevention
- Monitoring
- Follow-up
- Expert escalation

Recommendations are stored separately from the model output.

### 🌍 Multilingual Interface

The frontend includes support for:

- English
- Swahili
- Hindi
- Tamil

---

## 🧪 Supported Classes

| Crop | Classes |
|---|---|
| **Maize** | `maize_common_rust`, `maize_gray_leaf_spot`, `maize_northern_leaf_blight`, `maize_healthy` |
| **Potato** | `potato_early_blight`, `potato_late_blight`, `potato_healthy` |
| **Tomato** | `tomato_bacterial_spot`, `tomato_early_blight`, `tomato_late_blight`, `tomato_healthy` |

---

## 🧠 Machine Learning

### Dataset

The baseline model uses selected images from the **PlantVillage** dataset. Approximately **12,600** selected images were used across the supported crop classes.

The dataset was divided into:

- Training set
- Validation set
- Test set

### Training Strategy

The project uses **transfer learning**. Instead of training a deep neural network from scratch, pretrained CNN architectures were fine-tuned using crop disease images.

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
CNN Model
    ↓
Prediction
    ↓
Compare with Ground Truth
    ↓
Calculate Loss
    ↓
Backpropagation
    ↓
Update Model Weights
    ↓
Repeat
```

### Model Comparison

Three CNN architectures were compared:

| Model | Result |
|---|---|
| MobileNetV3-Small | Strong |
| ResNet18 | Strong |
| **EfficientNet-B0** | **Best** |

EfficientNet-B0 was selected as the final model.

### Final Evaluation

The final EfficientNet-B0 model achieved approximately:

| Metric | Score |
|---|---|
| Accuracy | 99.05% |
| Macro Precision | 98.69% |
| Macro Recall | 98.54% |
| Macro F1 | 98.60% |
| Macro Specificity | 99.90% |
| Accepted Accuracy | 99.25% |

> These results are from a held-out **PlantVillage-distribution** test set. They do **not** mean the model has 99% accuracy in real farm environments.

Real-world performance may be affected by:

- Lighting
- Complex backgrounds
- Camera quality
- Leaf orientation
- Disease stage
- Pest damage
- Nutrient deficiencies
- Unknown diseases
- Multiple symptoms

---

## 🏗 System Architecture

```text
                ┌──────────────────────┐
                │      Farmer UI       │
                │   HTML / CSS / JS    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    FastAPI Backend   │
                └──────────┬───────────┘
                           │
           ┌───────────────┼────────────────┐
           │               │                │
           ▼               ▼                ▼
    Authentication     AI Inference      Database
                           │
                           ▼
                   EfficientNet-B0
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         Prediction     Grad-CAM     Severity
              │
              ▼
        Recommendations
```

---

## 🔄 Scan Workflow

```text
Login
  ↓
Select Farm / Plot
  ↓
Select Crop
  ↓
Upload Image / Use Camera
  ↓
Image Quality Check
  ↓
AI Inference
  ↓
Healthy / Diseased / Uncertain
  ↓
Confidence
  ↓
Severity
  ↓
Affected Area
  ↓
Grad-CAM
  ↓
Recommendations
  ↓
Save Scan
  ↓
Follow-Up / Expert Review
```

---

## 🖼 Screenshots

Create this folder in the repository:

```bash
mkdir -p docs/screenshots
```

Then save your screenshots using these exact filenames.

| Landing Page | Farmer Dashboard |
|---|---|
| ![Landing Page](docs/screenshots/landing-page.png) | ![Farmer Dashboard](docs/screenshots/dashboard.png) |

| Crop Scan Page | Camera Scan |
|---|---|
| ![Crop Scan Page](docs/screenshots/scan-page.png) | ![Camera Scan](docs/screenshots/camera-scan.png) |

| AI Analysis | Disease Result |
|---|---|
| ![AI Analysis](docs/screenshots/analyzing.png) | ![Disease Result](docs/screenshots/disease-result.png) |

| Grad-CAM Heatmap | Scan History |
|---|---|
| ![Grad-CAM Heatmap](docs/screenshots/gradcam.png) | ![Scan History](docs/screenshots/history.png) |

| Farm Management | Expert Dashboard |
|---|---|
| ![Farm Management](docs/screenshots/farms.png) | ![Expert Dashboard](docs/screenshots/expert-dashboard.png) |

| Admin Dashboard |
|---|
| ![Admin Dashboard](docs/screenshots/admin-dashboard.png) |

Required filenames:

```text
landing-page.png
dashboard.png
scan-page.png
camera-scan.png
analyzing.png
disease-result.png
gradcam.png
history.png
farms.png
expert-dashboard.png
admin-dashboard.png
```

---

## 🛠 Technology Stack

**Frontend**

- HTML5
- CSS3
- JavaScript
- Browser Camera API
- Progressive Web App features

**Backend**

- Python
- FastAPI
- Uvicorn
- SQLAlchemy

**Machine Learning**

- PyTorch
- Torchvision
- EfficientNet-B0
- ResNet18
- MobileNetV3-Small
- Grad-CAM

**Database**

- Development: SQLite
- Future production option: PostgreSQL

---

## 📁 Project Structure

```text
crop-health-ai/
│
├── backend/
│   ├── main.py
│   ├── recommendations.py
│   └── ...
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   ├── manifest.json
│   ├── sw.js
│   └── locales/
│
├── ml/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   └── training/
│
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── smoke_test.py
│
├── tests/
├── knowledge_base/
├── artifacts/
├── docs/
│
├── config.yaml
├── requirements.txt
├── run.py
└── README.md
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aakashyadav-dev/CROP-HEALTH-AI.git
cd CROP-HEALTH-AI
```

### 2. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

Download the dataset:

```bash
python scripts/download_dataset.py
```

Prepare the dataset:

```bash
python scripts/prepare_dataset.py
```

---

## 🏋️ Train Models

```bash
python scripts/train.py
```

The training pipeline compares the supported CNN architectures and saves model artifacts.

---

## 📊 Evaluate Model

```bash
python scripts/evaluate.py
```

Evaluation includes:

- Accuracy
- Macro Precision
- Macro Recall
- Macro Specificity
- Macro F1
- Test Loss
- Coverage
- Accepted Accuracy
- Inference Time

---

## ▶️ Run the Application

```bash
python run.py
```

| Page | URL |
|---|---|
| Application | http://127.0.0.1:8000 |
| API documentation | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

---

## 🧪 Run Tests

```bash
pytest -q
```

---

## 🔐 User Roles

### Farmer

Can:

- Register and log in
- Create farms
- Create crop plots
- Scan leaves
- Use camera scanning
- View AI results
- View scan history
- Schedule follow-ups
- Submit feedback
- Receive notifications

### Extension Officer

Can:

- Review pending scans
- Review low-confidence scans
- Review severe scans
- Confirm AI predictions
- Correct predictions
- Provide expert verification

### Admin

Can:

- View users
- View scans
- View model versions
- View follow-ups
- View platform analytics

---

## 🔌 Main API Endpoints

**Authentication**

```http
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me
```

**Farms**

```http
POST /api/v1/farms
GET  /api/v1/farms
GET  /api/v1/farms/{farm_id}
```

**Scans**

```http
POST /api/v1/scans
GET  /api/v1/scans
GET  /api/v1/scans/{scan_id}
GET  /api/v1/scans/{scan_id}/gradcam
GET  /api/v1/scans/{scan_id}/image
GET  /api/v1/scans/{scan_id}/report.pdf
```

**Expert Review**

```http
GET  /api/v1/expert/pending
GET  /api/v1/expert/low-confidence
GET  /api/v1/expert/severe
POST /api/v1/expert/scans/{scan_id}/review
```

**Analytics**

```http
GET /api/v1/analytics/dashboard
GET /api/v1/analytics/health-trend
GET /api/v1/analytics/public-stats
```

---

## 🤖 Planned Crop Assistant

A future version will include a grounded AI Crop Assistant.

Example farmer questions:

- What should I do today?
- Can this disease spread?
- When should I scan again?
- Why is this severe?
- Should I contact an expert?

The assistant will use context such as:

```text
Current Scan
     +
Crop
     +
Disease
     +
Confidence
     +
Severity
     +
Farm / Plot
     +
Recommendation Knowledge Base
```

---

## 🔮 Future Improvements

- Real Kenyan field dataset
- Expert-verified farmer images
- Two-stage crop + disease classifier
- More crops and diseases
- Crop Assistant chatbot
- Weather-based disease risk
- Better severity estimation
- Recovery trend visualization
- Offline-friendly scanning
- ONNX / TensorFlow Lite deployment
- County-level disease surveillance
- PostgreSQL production deployment

---

## ⚠️ Limitations

The current model is primarily trained using PlantVillage-style images. Real-world field images may differ because of:

- Natural backgrounds
- Different lighting
- Camera quality
- Leaf orientation
- Multiple symptoms
- Pest damage
- Nutrient deficiencies
- Unknown diseases

Therefore, Crop Health AI should be treated as a **screening and decision-support tool**, not a replacement for a qualified agronomist or agricultural extension officer.

---

## 🧑‍⚕️ Responsible AI

Crop Health AI follows a human-in-the-loop approach:

```text
AI Screening
      ↓
Confidence Check
      ↓
Uncertain / Severe Case
      ↓
Expert Review
      ↓
Confirmed or Corrected Result
```

The system does not assume AI predictions are always correct.

---

## 📚 Documentation

| File | Description |
|---|---|
| [API.md](API.md) | API endpoints and usage |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and components |
| [DATASET.md](DATASET.md) | Dataset sources and preparation |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide |
| [EVALUATION.md](EVALUATION.md) | Model metrics and results |
| [LIMITATIONS.md](LIMITATIONS.md) | Known limitations in detail |
| [MODEL.md](MODEL.md) | Model architecture and training |
| [docs/](docs/) | Additional documentation and screenshots |

---

## 👨‍💻 Author

**Aakash Yadav**

GitHub: [@aakashyadav-dev](https://github.com/aakashyadav-dev)

---

## 📄 License

The source code is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

Dataset usage should follow the original dataset provider's license terms.

---

## 🌱 Vision

Crop Health AI aims to move beyond simple plant disease classification by combining computer vision, explainable AI, uncertainty handling, farm monitoring, recommendations, follow-up workflows, and expert verification into one practical crop-health platform.

<p align="center">
  Made with 🌱 AI + Computer Vision + Agriculture
</p>
