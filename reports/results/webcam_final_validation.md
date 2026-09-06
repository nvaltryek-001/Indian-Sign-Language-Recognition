# Webcam Final Validation

Date: 2026-09-05

## Status

PARTIAL / BLOCKED for sign-accuracy measurement.

The local camera was available and a real startup smoke test passed:

- Camera opened: PASS
- Camera probe frame: PASS
- Probe resolution: 640x480
- Application resolution: 960x540
- Production model loaded once: PASS
- Model input: `(45, 258)`
- Model output: 50 classes
- MediaPipe Holistic startup: PASS
- Webcam resource cleanup: implemented with `finally`
- Quality-gated capture path: present
- User feedback on failed landmark quality: `Improve hand visibility`
- FPS display: present

The application was stopped after startup. No sign performer was available during this verification session, so no labeled webcam trials were collected.

## Browser acceptance smoke test

On 2026-09-06 the browser client was opened at `http://127.0.0.1:5173` with the API on port 8000. API status became online, camera permission succeeded, MediaPipe reached ready state, and the privacy-rendered canvas processed 156 frames. The available automated camera feed contained no hands during this run (`pose=0`, `leftHand=0`, `rightHand=0`), so the capture button remained disabled and no invalid sequence was sent. This validates the rejection path, not human sign recognition accuracy.

## Browser uploaded-video acceptance

The labeled repository video `data/include50/videos/Adjectives/1. loud/MVI_5177.MOV` was processed through the browser uploader on 2026-09-06. The sequential extractor generated `45/45` frames, with diagnostics of `68` total processed frames, `67` pose detections, `52` left-hand detections, and `59` right-hand detections. The API returned the V2 result `1. loud` with `100.0%` confidence in `117 ms`, and the UI displayed `RECOGNIZED`. This validates browser MediaPipe, temporal sampling, API transport, model inference, and result rendering for that labeled video; it is not a human webcam trial.

## Continuous mode acceptance

The same labeled `1. loud` video was processed with Continuous mode enabled. Motion-based segmentation produced one segment, exactly `45/45` sampled frames, one history event, and the browser displayed `RECOGNIZED: 1. loud` at `100.0%`. Face visibility defaulted to ON, and the OFF toggle switched the preview to the local face mask without changing landmark extraction. Multiple distinct human signs in one live session were not available for this run, so multi-sign accuracy remains unmeasured.

## Requested Sign Trials

| Intended sign | Landmark quality | Top prediction | Confidence | Top-3 | Match |
| --- | --- | --- | --- | --- | --- |
| loud | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | BLOCKED |
| quiet | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | BLOCKED |
| happy | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | BLOCKED |
| good | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | BLOCKED |
| new | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | BLOCKED |

## Accuracy

Webcam validation accuracy: NOT COMPUTED.

Computing accuracy requires labeled human-performed trials. No accuracy percentage is claimed from the startup smoke test.

## Existing Labeled Artifact

The repository contains six previously recorded `1. Dog` sequences in `data/webcam50_validation/webcam_results.csv`. The production V2 model classified 0 of 6 correctly (0.00% on this single-class sample), generally as `4. Bird`. This is evidence of webcam domain shift, not a representative five-sign validation set, and it is not sufficient evidence to retrain or promote a new model.

## Reproduction

Run from the repository root with a camera attached:

```powershell
python app/inference/realtime_predict.py
```

Use Q to exit. Use T to display the top-five list. Record the intended sign, quality status, top prediction, confidence, top-three predictions, and whether the top prediction matches the intended label for each trial.
