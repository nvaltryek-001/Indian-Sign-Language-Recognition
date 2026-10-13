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
|
