# 🤟 Real-Time Indian Sign Language Recognition Using MediaPipe Hands and LSTM with Reference Sequence Matching

**AI-powered Indian Sign Language (ISL) recognition using hand landmarks, temporal deep learning, and Dynamic Time Warping (DTW).**

This project recognizes signs from the INCLUDE-50 dataset using a hand-focused LSTM model and compares predicted sign sequences against stored reference sequences. It supports real-time webcam capture through a local client and exposes inference through a FastAPI service.

> **Project status:** The model, feature pipeline, API, and automated regression tests have been locally verified as documented below. Live webcam sign accuracy has not been measured. Docker, Kubernetes, and remote GitHub Actions execution must be verified in their respective environments.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [AI Model](#-ai-model)
- [Recognition Pipeline](#-recognition-pipeline)
- [Reference Sequence Matching](#-reference-sequence-matching)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the API](#-running-the-api)
- [Real-Time Webcam Recognition](#-real-time-webcam-recognition)
- [API Documentation](#-api-documentation)
- [Frontend](#-frontend)
- [Testing and Verification](#-testing-and-verification)
- [Measured Model Results](#-measured-model-results)
- [CI/CD and Deployment](#-cicd-and-deployment)
- [Troubleshooting](#-troubleshooting)
- [Known Limitations](#-known-limitations)
- [Future Enhancements](#-future-enhancements)

## 🚀 Overview

Indian Sign Language Recognition aims to help computers interpret sign sequences by analysing hand movements over time.

This project uses MediaPipe Hands to extract hand landmarks from video frames, an LSTM neural network to classify temporal sequences, and DTW reference matching to compare a captured sequence with stored examples.

The application separates camera capture from model inference:

- **Local webcam client:** Accesses the user's camera and extracts hand landmarks.
- **Inference API:** Receives landmark sequences and performs model prediction and reference matching.
- **Browser frontend:** Supports webcam interaction, expected-sign validation, stable predictions, and a no-sign state.

This architecture allows the inference API to run locally or on a reachable server without requiring a remote server to access the user's physical webcam.

## ✨ Key Features

- 🤟 Indian Sign Language classification across 50 classes.
- ✋ Hand landmark extraction using MediaPipe Hands.
- 🧠 LSTM-based temporal sequence classification.
- 🔎 DTW-based reference sequence matching.
- 🎥 Real-time webcam capture through a local client.
- 🎯 Expected-sign validation.
- 🗳️ Stable prediction voting to reduce unstable frame-by-frame output.
- 🛑 No-sign state when hand landmarks are missing or insufficient.
- ⚡ FastAPI inference endpoints.
- 🧪 Automated tests and model contract validation.
- 🐳 Docker build configuration.
- ☸️ Kubernetes manifests and GitHub Actions CI/CD workflows.

## 🏗️ System Architecture

```text
                INCLUDE-50 DATASET
                        |
                        v
             MediaPipe Hands Extraction
                        |
                        v
             126 Features per Frame
                        |
                        v
               45-Frame Sequence
                  (45 x 126)
                        |
                        v
                Hand-Focused LSTM
                        |
                        v
               Prediction: 50 Classes
                        |
                        v
              DTW Reference Matching
                        |
                        v
             MATCH / NOT MATCH / NO SIGN
                        |
             +----------+----------+
             |                     |
             v                     v
       Local Webcam Client    FastAPI Service
             |                     |
             +----------+----------+
                        |
                        v
                Browser Frontend
```

### High-level components

| Component | Responsibility |
|---|---|
| MediaPipe Hands | Detects and extracts hand landmarks |
| Feature pipeline | Creates fixed-length temporal sequences |
| LSTM model | Predicts a sign class |
| Reference database | Stores reference hand sequences |
| DTW matcher | Compares captured and reference sequences |
| FastAPI | Provides health and inference endpoints |
| Webcam client | Captures video on the camera-equipped machine |
| Frontend | Displays predictions and expected-sign results |
| GitHub Actions | Automates configured CI and deployment workflows |

## 📚 Dataset

The project uses the **INCLUDE-50 Indian Sign Language corpus**.

| Dataset property | Value |
|---|---|
| Sign classes | 50 |
| Reference hand sequences | 943 |
| Features per frame | 126 |
| Frames per sequence | 45 |
| Sequence shape | `(45, 126)` |
| Numeric data type | `float32` |
| Reference database | `data/include50/reference_hand_sequences` |

### Hand feature representation

Each frame contains 126 values:

- Left hand: 21 landmarks × 3 coordinates = 63 features.
- Right hand: 21 landmarks × 3 coordinates = 63 features.

Total:

`63 + 63 = 126 features per frame`

A sequence of 45 frames therefore has the shape:

`(45, 126)`

The reference sequences are organized by sign class so the matching stage can compare an input sequence with relevant stored examples.

## 🧠 AI Model

The active hand-focused LSTM model is:

```text
models/include50_hand_lstm_best.keras
```

Class mapping:

```text
models/include50_hand_classes.json
```

### Model contract

| Property | Value |
|---|---|
| Model type | LSTM |
| Input shape | `(None, 45, 126)` |
| Output classes | 50 |
| Sequence length | 45 frames |
| Features per frame | 126 |
| Validation accuracy | 97.76% |
| Held-out test accuracy | 92.74% |

The validation and held-out test metrics describe the corresponding dataset evaluation. They are **not webcam accuracy measurements**.

## 🔄 Recognition Pipeline

### Step 1 — Capture video

The local webcam client reads frames from the user's camera.

### Step 2 — Extract hand landmarks

MediaPipe Hands extracts the left and right hand landmarks when available.

### Step 3 — Build a temporal sequence

The feature pipeline prepares a 45-frame sequence containing 126 features per frame.

### Step 4 — Validate input quality

The application checks whether the sequence contains sufficient valid landmark information. If no meaningful sign signal is available, the system should return a no-sign state instead of forcing a class prediction.

### Step 5 — Predict the sign

The LSTM model predicts one of the 50 configured sign classes.

### Step 6 — Match the reference sequence

DTW compares the captured sequence against stored reference sequences and produces a similarity or distance-based matching result according to the configured implementation.

### Step 7 — Stabilize and display the result

The frontend uses stable voting and expected-sign validation to determine whether the observed prediction matches the target sign.

Possible states include:

- `MATCHING`
- `NOT MATCHING`
- `NO SIGN DETECTED`

The exact displayed status depends on the current prediction, reference-matching result, and input-quality checks.

## 🔎 Reference Sequence Matching with DTW

Dynamic Time Warping (DTW) compares temporal sequences that may differ in execution speed or alignment.

In this project, the captured hand sequence is compared with stored reference sequences to help evaluate whether the observed movement corresponds to an expected sign.

Conceptually:

1. Receive the captured 45-frame sequence.
2. Select reference sequences from the reference database.
3. Calculate DTW distance between the input and reference sequences.
4. Apply the configured matching logic.
5. Return the prediction and reference-matching information.

DTW complements the LSTM prediction; it is not itself a guarantee of correct sign recognition. Matching thresholds must be evaluated using representative data.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and model inference |
| MediaPipe Hands | Hand landmark extraction |
| TensorFlow / Keras | LSTM model loading and inference |
| NumPy | Numerical feature sequences |
| FastAPI | HTTP inference service |
| Uvicorn | ASGI server |
| OpenCV | Webcam and video processing |
| DTW | Temporal reference sequence comparison |
| HTML, CSS, JavaScript | Browser interface, as implemented |
| Pytest | Automated tests |
| Docker | Containerization |
| Kubernetes | Container orchestration |
| GitHub Actions | CI/CD automation |

## 📁 Project Structure

```text
ISL-Recognition-Research/
│
├── app/
│   ├── api.py
│   └── inference/
│       └── realtime_hand_predict.py
│
├── frontend/
│
├── models/
│   ├── include50_hand_lstm_best.keras
│   └── include50_hand_classes.json
│
├── data/
│   └── include50/
│       └── reference_hand_sequences/
│
├── tests/
│
├── reports/
│   └── results/
│       └── webcam_final_validation.md
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml
│   └── service.yaml
│
├── Dockerfile
└── README.md
```

*This is a representative layout based on the project configuration described here. Confirm the current repository contents before relying on every path.*

## ⚙️ Installation

### Prerequisites

- Python environment compatible with the project's installed TensorFlow and MediaPipe dependencies.
- Git.
- A webcam for live local capture.
- The trained model and class mapping files.
- The reference sequence database.
- Node.js and npm if frontend dependencies require them.
- Docker and/or Kubernetes only if testing those deployment targets.

### 1. Clone the repository

```powershell
git clone <YOUR-REPOSITORY-URL>
cd ISL-Recognition-Research
```

Replace `<YOUR-REPOSITORY-URL>` with the actual repository URL.

### 2. Activate the Python environment

The configured local commands use `venv311`.

```powershell
.\venv311\Scripts\python.exe --version
```

If the environment has not been created, create a virtual environment using the Python version supported by the project's dependencies, then install the repository's requirements.

For example:

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install --upgrade pip
```

Install dependencies using the repository's requirements file if one is provided:

```powershell
.\venv311\Scripts\python.exe -m pip install -r requirements.txt
```

Do not recreate an existing working environment without checking the project's dependency versions first.

### 3. Verify required model and data files

Confirm these paths exist:

```text
models/include50_hand_lstm_best.keras
models/include50_hand_classes.json
data/include50/reference_hand_sequences/
```

The API requires the model, class mapping, and reference database to be available in the configured locations.

## ▶️ Running the API

Start the hand-only inference API from the repository root:

```powershell
.\venv311\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

The local API will be available at:

```text
http://127.0.0.1:8000
```

### Health check

In another terminal:

```powershell
curl.exe http://127.0.0.1:8000/health/hand
```

The health endpoint reports:

- `model_loaded`
- `class_count`
- `reference_database_available`
- `reference_classes`
- `reference_sequences`

Use these fields to verify that the active hand model and reference database are loaded.

## 🎥 Real-Time Webcam Recognition

Run the standalone webcam client:

```powershell
.\venv311\Scripts\python.exe app\inference\realtime_hand_predict.py
```

The client must run on the machine that has access to the physical webcam.

### Recommended environment

- Ensure the camera is not exclusively occupied by another application.
- Use clear lighting.
- Keep the hands visible within the camera frame.
- Avoid excessive motion blur.
- Follow the sequence warm-up instructions displayed by the application.
- Keep the model, class mapping, and inference API available.

**Camera privacy:** A remote server or Kubernetes pod cannot directly access the webcam attached to the user's laptop. Camera capture must run locally, and the client can then send the extracted landmark sequence to the inference API.

## 🔌 API Documentation

### `GET /health/hand`

Checks the readiness of the hand-recognition pipeline.

Example:

```bash
curl http://127.0.0.1:8000/health/hand
```

The response should report model loading and reference database status. Consult the actual API response for the current values.

### `POST /predict/hand-sequence`

Accepts a hand-landmark sequence with the required shape:

`(45, 126)`

The response includes:

| Field | Purpose |
|---|---|
| `prediction` | Predicted sign class |
| `confidence` | Model confidence or configured confidence score |
| `reference_match` | Reference matching result |
| `expected_sign` | Expected sign information, when supplied or available |

The request must use the precise payload schema implemented by `app/api.py`. The API should reject invalid shapes or non-finite numeric inputs according to its validation logic.

The frontend must use `/predict/hand-sequence` for the hand-only model rather than the older 258-feature endpoint.

## 🖥️ Frontend

The browser application in `frontend/` supports:

- Live webcam interaction.
- Hand-only inference.
- Expected-sign validation.
- Stable prediction voting.
- Matching and non-matching status.
- A no-sign state when the input landmarks are insufficient.

### Frontend and backend relationship

```text
Browser Camera
      |
      v
Hand Landmark Extraction
      |
      v
45 x 126 Sequence
      |
      v
POST /predict/hand-sequence
      |
      v
LSTM + DTW Reference Matching
      |
      v
Stable Result Display
```

The frontend must not send the old 258-feature sequence to the hand-only endpoint.

## 🧪 Testing and Verification

The project verification notes dated **2026-09-05** document the following results.

| Check | Status |
|---|---|
| Automated tests | PASS — 9 passed |
| Python compilation | PASS |
| Model contract | PASS |
| 50-class mapping | PASS |
| MediaPipe Holistic initialization | PASS |
| 45-frame feature pipeline validation | PASS |
| API health check | PASS |
| Workflow and Kubernetes YAML parsing | PASS |
| Docker runtime verification | BLOCKED |
| Kubernetes runtime verification | BLOCKED |
| Webcam startup | PASS |
| Human-labeled live webcam sign accuracy | NOT MEASURED |

These are recorded verification results, not a guarantee that the same results will occur on every machine or after code changes.

### Python compilation

```powershell
.\venv311\Scripts\python.exe -m compileall app tests
```

### Automated tests

```powershell
.\venv311\Scripts\python.exe -m pytest -q
```

Use the test command configured by the repository if it differs from the example above.

### Important verification distinctions

- A passing model contract confirms that the model interface and expected tensor dimensions are compatible.
- Passing regression tests validates the tested software paths.
- Webcam startup confirms camera initialization, not sign-recognition accuracy.
- A successful API health response does not by itself verify every inference scenario.
- Docker and Kubernetes should be reported as blocked when their runtimes or clusters are unavailable.

## 📈 Measured Model Results

The project notes also record a V2 production-model evaluation on an active test split of 124 sequences.

| Metric | Recorded result |
|---|---:|
| Top-1 accuracy | 91.13% |
| Top-3 accuracy | 98.39% |
| Macro F1 | 0.872 |
| Test sequences | 124 |

The project also records separate validation and held-out test metrics for the active hand-focused LSTM model:

- Validation accuracy: **97.76%**
- Held-out test accuracy: **92.74%**

These results should be presented as separate evaluation records unless the evaluation reports establish that they refer to the same model checkpoint, dataset split, and protocol.

### Observed weaker classes

The recorded V2 test split reports:

- `77. Boy` and `42. T-Shirt` had 0% recall.
- `1. Dog`, `2. Death`, `2. quiet`, `40. Paint`, `55. White`, and `78. Girl` had recall below 100%.

These findings are based on the recorded 124-sequence test split. They do not establish that those classes always fail, nor do they establish webcam performance.

### Recorded-video acceptance test

The project notes report that the browser acceptance test processed a labelled `1. loud` video in continuous mode:

- Valid frames: `45/45`
- Prediction: `1. loud`
- Recorded confidence: `100.0%`
- History events: 1
- API latency: under 120 ms

This validates the documented recorded-video test path. It does not replace human-labeled live webcam trials.

### Webcam accuracy

**Live webcam sign accuracy has not been measured.**

A proper evaluation requires multiple human-labeled trials across signs, participants, lighting conditions, signing speeds, and camera positions. Dataset accuracy must not be presented as real-time webcam accuracy.

## 🔁 CI/CD and Deployment

The repository includes the following GitHub Actions workflows:

```text
.github/workflows/ci.yml
.github/workflows/cd.yml
```

### Continuous Integration

The CI workflow is configured for:

- Python dependency installation.
- Python compilation checks.
- Automated tests.
- Model contract validation.
- Docker image build.

A local YAML parse or local test run does not prove that GitHub Actions has completed successfully. Verify the workflow run in the repository's **Actions** tab.

### Continuous Deployment

The CD workflow is designed to:

1. Publish a container image to a configured registry.
2. Use an immutable image reference based on `${{ github.sha }}`.
3. Update the Kubernetes deployment to use that image.
4. Apply the Kubernetes manifests when the required credentials and deployment conditions are available.

### Required GitHub Actions secrets

Configure the following secrets in the repository's Actions settings when using the corresponding workflow:

| Secret | Purpose |
|---|---|
| `REGISTRY_LOGIN_SERVER` | Container registry address |
| `REGISTRY_USERNAME` | Registry authentication username |
| `REGISTRY_PASSWORD` | Registry authentication password |
| `KUBE_CONFIG_DATA` | Kubernetes cluster configuration |

Never commit registry credentials, kubeconfig contents, or other secrets into source files.

### Deployment architecture

```text
Developer Push
      |
      v
GitHub Repository
      |
      v
GitHub Actions CI
      |
      v
Tests + Model Contract + Docker Build
      |
      v
Container Registry
      |
      v
GitHub Actions CD
      |
      v
Kubernetes Cluster
      |
      v
Hand Inference API
      ^
      |
Local Webcam Client
      |
      v
Browser / User Interface
```

The remote inference API can process submitted landmark sequences, but it cannot directly use the webcam attached to a user's laptop. Run the camera client on the local machine.

### Local Kubernetes example with kind

Only run these commands when Docker, `kind`, and `kubectl` are installed and working.

```powershell
kind create cluster --name isl-recognition

docker build -t isl-recognition:test .

kind load docker-image isl-recognition:test --name isl-recognition

kubectl apply -f k8s/namespace.yaml `
              -f k8s/configmap.yaml `
              -f k8s/deployment.yaml `
              -f k8s/service.yaml

kubectl -n isl-recognition rollout status deployment/isl-recognition-api --timeout=180s
```

Verify that the manifests reference the correct image and that the deployment configuration matches the current application. A local test image must be available to the cluster, while a remote cluster normally requires a registry-accessible image.

## 🛠️ Troubleshooting

### API health endpoint fails

Check:

- Whether port `8000` is already in use.
- Whether the model file exists.
- Whether the class mapping loads.
- Whether the reference database path is correct.
- Whether TensorFlow and other dependencies are compatible.
- The API startup logs.

### Webcam warm-up stalls

- Improve lighting.
- Keep the hands fully visible.
- Avoid excessive motion blur.
- Check MediaPipe detection.
- Ensure the local client has camera permission.

### Invalid sequence shape

For the active hand-only API, verify the sequence shape is exactly:

`(45, 126)`

The repository's older Holistic feature pipeline uses 258 features, but that is a different input contract. Do not send a `(45, 258)` sequence to the hand-only endpoint.

### Non-finite sequence values

Ensure the landmark sequence contains finite numeric values. Missing or low-quality hand landmarks should be handled by the input-quality gate rather than converted into arbitrary sign predictions.

### Docker unavailable

Record Docker verification as `BLOCKED` if the Docker daemon cannot run. A Dockerfile existing in the repository is not evidence of a successful container build.

### Kubernetes unavailable

Check:

```powershell
kubectl config current-context
kubectl get nodes
```

If there is no active context or reachable cluster, record Kubernetes verification as `BLOCKED`.

### GitHub Actions deployment fails

Verify:

- Required secrets are configured.
- Registry authentication succeeds.
- The published image exists.
- The workflow uses the expected immutable image reference.
- Kubernetes credentials and permissions are valid.
- The target cluster is reachable.

## ⚠️ Known Limitations

1. **Fixed model:** The repository uses a trained baseline and does not automatically retrain the model.
2. **Dataset-to-webcam gap:** Dataset metrics do not establish live webcam accuracy.
3. **Input quality:** Lighting, occlusion, signing speed and MediaPipe detection quality affect results.
4. **Class imbalance or limited samples:** Some classes have weaker measured recall on the recorded test split.
5. **Camera location:** Physical webcam capture must run on the user's machine.
6. **Deployment dependency:** Docker and Kubernetes results depend on available runtimes and credentials.
7. **CI/CD verification:** Remote workflow execution must be confirmed through GitHub Actions.
8. **Reference matching:** DTW thresholds require evaluation and may need calibration for different users and conditions.
9. **Recognition scope:** This is a sign-classification system, not a complete natural-language ISL translation system. Grammar, continuous sentence interpretation, facial expressions and full-body non-manual signals are not established by the hand-only pipeline.

## 🔮 Future Enhancements

- Collect human-labeled webcam evaluation data.
- Improve per-class performance through data analysis and model evaluation.
- Calibrate DTW matching thresholds.
- Add participant-independent test splits.
- Improve low-confidence and no-sign handling.
- Expand temporal sequence evaluation.
- Add model versioning and reproducible evaluation reports.
- Add deployment health monitoring and automated smoke tests.
- Improve browser camera diagnostics.
- Explore broader ISL sentence-level recognition only after collecting suitable data and validating the required features.

## 📜 Responsible Use

This project is intended for research, learning, and demonstration. Sign recognition results can be wrong and should not be treated as authoritative communication or accessibility support without proper human evaluation.

Respect the dataset's licence and usage terms. Obtain informed permission for recordings of signers, protect personal video and landmark data, and document model limitations before deploying the system for real users.

---

**Project:** Real-Time Indian Sign Language Recognition Using MediaPipe Hands and LSTM with Reference Sequence Matching  
**Dataset:** INCLUDE-50  
**Model:** Hand-focused LSTM  
**Input:** 45 frames × 126 hand features  
**Classes:** 50  
**Inference API:** FastAPI  
**Reference matching:** Dynamic Time Warping (DTW)

**Built to explore temporal sign classification, reference matching, and local webcam inference with transparent model and deployment verification.**
