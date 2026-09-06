# Real-Time Indian Sign Language Recognition Using MediaPipe Hands and LSTM with Reference Sequence Matching

## Project title

Real-Time Indian Sign Language Recognition Using MediaPipe Hands and LSTM with Reference Sequence Matching

## System pipeline

```text
INCLUDE-50
    -> MediaPipe Hands
    -> 126 hand features per frame
    -> 45-frame sequence
    -> LSTM
    -> 50-class prediction
    -> DTW reference matching
    -> Real-time webcam
    -> MATCH / NOT MATCH
```

## Dataset

This project uses the INCLUDE-50 sign corpus with 943 hand reference sequences across 50 classes. The extracted hand sequences are stored as NumPy arrays of shape `(45, 126)` using float32 values. The reference database is under `data/include50/reference_hand_sequences` and is organized by sign class.

## Model

The active model is the hand-focused LSTM at `models/include50_hand_lstm_best.keras`, with the class mapping in `models/include50_hand_classes.json`.

- Model input shape: `(None, 45, 126)`
- Output classes: 50
- Validation accuracy: 97.76%
- Held-out test accuracy: 92.74%

These metrics are dataset/model evaluation metrics and are not claimed webcam accuracy.

## What the pipeline does

- MediaPipe Hands extracts normalized hand landmarks from live video frames.
- Each frame uses 126 features: 21 left-hand landmarks × 3 coordinates + 21 right-hand landmarks × 3 coordinates.
- The model consumes a 45-frame temporal sequence and predicts one of 50 classes.
- DTW compares the captured sequence against reference sequences to produce a similarity score.
- The webcam client runs locally, predicts on a valid sequence, and reports `MATCHING`, `NOT MATCHING`, or `NO SIGN DETECTED` depending on signal quality and stability.
- When no hand is detected or the frame quality is insufficient, the system must not force a random 50-class label; it returns a no-sign state instead.

## Local runtime

Use the hand-only inference API:

```bash
.\venv311\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/health/hand
```

Standalone webcam script:

```powershell
.\venv311\Scripts\python.exe app\inference\realtime_hand_predict.py
```

## API contract

`GET /health/hand` reports:

- `model_loaded`
- `class_count`
- `reference_database_available`
- `reference_classes`
- `reference_sequences`

`POST /predict/hand-sequence` accepts a 45 × 126 hand sequence and returns:

- `prediction`
- `confidence`
- `reference_match`
- `expected_sign`

## Frontend

The browser app in `frontend/` uses the hand-only route `/predict/hand-sequence` and not the old 258-feature endpoint. It supports a live webcam, expected-sign validation, stable voting, and a no-sign state when landmarks are insufficient.

## Important note on webcam testing

The physical webcam cannot be runtime-tested from this environment because camera access is unavailable here. The code is prepared for local execution, but actual webcam capture results are not claimed as verified in this container.

## Deployment notes

The repository includes Docker and Kubernetes manifests for deployment, but they are validated only to the extent supported by the current runtime environment. Do not claim cloud deployment success without a real cluster and credentialed deployment run.

## Testing summary

The project has verified:

- 943 hand sequences exist
- 50 classes exist
- model loads successfully
- class map loads successfully
- DTW reference database loads successfully
- API health endpoint responds correctly
- regression tests pass

The project does not claim webcam accuracy without a real camera test on a local machine.

The repository includes:

- `.github/workflows/ci.yml` for Python dependency installation, compilation checks, tests, model contract validation, and Docker image build
- `.github/workflows/cd.yml` for registry publishing and conditional Kubernetes deployment

Required GitHub secrets include:

- `REGISTRY_LOGIN_SERVER`
- `REGISTRY_USERNAME`
- `REGISTRY_PASSWORD`
- `KUBE_CONFIG_DATA`

These values must be supplied in the repository or environment settings. No credentials are checked into source files.

## Deployment

The end-to-end deployment flow is:

```text
GitHub
  -> GitHub Actions CI
  -> Docker build / registry publish
  -> GitHub Actions CD
  -> Kubernetes deployment
  -> inference API
```

Use the repository container image and then apply the manifests to a reachable cluster. Remote deployment can expose the API, but it cannot access the user's laptop camera directly.

## Troubleshooting

- If `/health` fails, check the model path, TF runtime, and container logs.
- If webcam capture stalls in warm-up, improve lighting and keep both hands and the full pose visible.
- If a sequence is rejected, verify the shape is exactly `(45, 258)` and contains finite values only.
- If Docker or Kubernetes is unavailable locally, record it as `BLOCKED` instead of claiming it passed.

For a local Kubernetes test cluster, install Minikube or kind, start it, load or push the image, and then apply the manifests. For example with kind:

```powershell
kind create cluster --name isl-recognition
docker build -t isl-recognition:test .
kind load docker-image isl-recognition:test --name isl-recognition
kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml
kubectl -n isl-recognition rollout status deployment/isl-recognition-api --timeout=180s
```

The CD workflow replaces the deployment image with the immutable registry image before applying it. A local cluster without image loading requires a registry-accessible image.

## Known Limitations

- The model is a fixed production baseline and is not retrained automatically in this repository.
- Local webcam capture depends on lighting and MediaPipe detection quality.
- Kubernetes cannot access a local camera without a client process running on the user machine.
- Docker and Kubernetes verification remain environment-dependent and should be reported as blocked when no runtime is available.

## Important

The production workflow is designed for local webcam capture and remote inference, not for remote webcam access from a Kubernetes pod. The user must run the capture client on the same machine that has the camera, and the resulting landmark sequence is sent to the inference service.

## Verification Status

Local verification on 2026-09-05:

- Tests: PASS (`9 passed`)
- Python compilation: PASS (`python -m compileall app tests`)
- Model contract: PASS (`(45, 258)` input, 50 outputs, finite normalized predictions, class keys `0..49`)
- Feature pipeline: PASS (MediaPipe Holistic initializes; 132 pose + 63 left hand + 63 right hand = 258 features; 45-frame validation and quality gate pass)
- API: PASS (`/health` returned `{"status":"ok","model_loaded":true}`; port 8000 was already occupied by the running API during the final check)
- Workflow and Kubernetes YAML parsing: PASS
- Docker: BLOCKED (Docker daemon unavailable on the verification host)
- Kubernetes: BLOCKED (no current context or reachable cluster on the verification host)
- Live webcam startup: PASS (camera opened and the application initialized at 960x540; no labeled human sign trials were available)
- Webcam sign accuracy: NOT MEASURED (see `reports/results/webcam_final_validation.md`)

The CD workflow publishes and deploys the same immutable image reference, `${{ github.sha }}`, when the required registry and Kubernetes secrets are present. GitHub Actions execution itself must be verified by a remote workflow run.

## Measured Model Results

The V2 production model was evaluated on the active test split with 124 sequences:

- Top-1 accuracy: 91.13%
- Top-3 accuracy: 98.39%
- Macro F1: 0.872
- Weak observed classes: `77. Boy` and `42. T-Shirt` had 0% recall in this split; `1. Dog`, `2. Death`, `2. quiet`, `40. Paint`, `55. White`, and `78. Girl` were below 100% recall.

These are dataset results, not webcam accuracy. Human-labeled webcam trials are still required to measure domain shift.

The browser acceptance test also processed the repository's labeled `1. loud` video in continuous mode: `45/45` valid frames, `1. loud` at `100.0%`, one history event, and API latency under 120 ms. This validates the recorded-video path; it does not replace live human webcam trials.
