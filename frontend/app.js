"use strict";

/*
 * ============================================================
 * ISL VISION
 * INCLUDE-50 REAL-TIME FRONTEND
 * ============================================================
 *
 * REAL CAMERA
 *     ↓
 * MEDIAPIPE HANDS
 *     ↓
 * LEFT 63 + RIGHT 63
 *     ↓
 * 126 FEATURES / FRAME
 *     ↓
 * 45 FRAME BUFFER
 *     ↓
 * FASTAPI
 *     ↓
 * LSTM + DTW
 *
 * ============================================================
 */

const API_BASE =
    window.ISL_API_BASE_URL ||
    "http://127.0.0.1:8000";


const SEQUENCE_LENGTH = 45;
const FEATURES_PER_FRAME = 126;


/*
 * INCLUDE-50 CLASS LIST
 */

const SIGNS = [
    "1. Dog",
    "2. loud",
    "3. Car",
    "4. Election",
    "5. train ticket",
    "6. House",
    "7. Death",
    "8. quiet",
    "9. Court",
    "10. Store or Shop",
    "11. Window",
    "12. happy",
    "13. Pen",
    "14. Bank",
    "15. Hat",
    "16. Bird",
    "17. I",
    "18. Paint",
    "19. T-Shirt",
    "20. Shoes",
    "21. it",
    "22. you (plural)",
    "23. Red",
    "24. Hello",
    "25. Cow",
    "26. Good Morning",
    "27. Fan",
    "28. Black",
    "29. Cell phone",
    "30. Thank you",
    "31. White",
    "32. Father",
    "33. Summer",
    "34. Fall",
    "35. Brother",
    "36. Monday",
    "37. Boy",
    "38. Girl",
    "39. Year",
    "40. long",
    "41. short",
    "42. big large",
    "43. Teacher",
    "44. small little",
    "45. Time",
    "46. hot",
    "47. Priest",
    "48. new",
    "49. good",
    "50. dry"
];


/*
 * ============================================================
 * STATE
 * ============================================================
 */

let mediaStream = null;

let cameraRunning = false;

let mediaPipeHands = null;

let mediaPipeReady = false;

let lastResults = null;

let sequence = [];

let capturing = false;

let handFrames = 0;

let processingPrediction = false;

let expectedSign =
    SIGNS[0];

let threshold =
    0.35;

let history = [];

let currentLeftHand = false;

let currentRightHand = false;


/*
 * ============================================================
 * DOM
 * ============================================================
 */

const $ = id =>
    document.getElementById(id);


/*
 * ============================================================
 * INITIALIZE
 * ============================================================
 */

document.addEventListener(
    "DOMContentLoaded",
    initialize
);


async function initialize() {

    populateExpectedSigns();

    populateSignGrid();

    setupEvents();

    setupThreshold();

    setupModals();

    setStatus(
        "mediaPipeStatus",
        "Ready",
        true
    );

    /*
     * MediaPipe Hands
     */

    try {

        if (
            typeof Hands ===
            "undefined"
        ) {

            throw new Error(
                "MediaPipe Hands script was not loaded."
            );
        }


        mediaPipeHands =
            new Hands({

                locateFile:
                    file =>
                        `vendor/mediapipe-hands/${file}`

            });


        mediaPipeHands.setOptions({

            maxNumHands: 2,

            modelComplexity: 1,

            minDetectionConfidence:
                0.5,

            minTrackingConfidence:
                0.5

        });


        mediaPipeHands.onResults(
            handleMediaPipeResults
        );


        mediaPipeReady = true;


        setStatus(
            "mediaPipeStatus",
            "Loaded",
            true
        );


    }

    catch (error) {

        console.error(
            "MediaPipe initialization failed:",
            error
        );

        mediaPipeReady = false;

        setStatus(
            "mediaPipeStatus",
            "Error",
            false
        );

    }


    /*
     * Backend health
     */

    await checkBackend();


    updateModelStatus();
}


/*
 * ============================================================
 * EVENTS
 * ============================================================
 */

function setupEvents() {

    $("startCameraBtn")
        .addEventListener(
            "click",
            startCamera
        );


    $("stopCameraBtn")
        .addEventListener(
            "click",
            stopCamera
        );


    $("captureBtn")
        .addEventListener(
            "click",
            startCapture
        );


    $("clearResultBtn")
        .addEventListener(
            "click",
            clearResult
        );


    $("clearHistoryBtn")
        .addEventListener(
            "click",
            clearHistory
        );


    $("uploadBtn")
        .addEventListener(
            "click",
            () =>
                $("videoInput").click()
        );


    $("videoInput")
        .addEventListener(
            "change",
            handleVideoUpload
        );


    $("referenceBtn")
        .addEventListener(
            "click",
            showReferenceInfo
        );


    $("expectedSign")
        .addEventListener(
            "change",
            event => {

                expectedSign =
                    event.target.value;

                selectSignInGrid(
                    expectedSign
                );

            }
        );


    $("signSearch")
        .addEventListener(
            "input",
            event => {

                renderSignGrid(
                    event.target.value
                );

            }
        );


    $("settingsBtn")
        .addEventListener(
            "click",
            showSettingsInfo
        );


    $("helpBtn")
        .addEventListener(
            "click",
            () =>
                openModal("helpModal")
        );


    $("aboutBtn")
        .addEventListener(
            "click",
            () =>
                openModal("aboutModal")
        );

}


/*
 * ============================================================
 * EXPECTED SIGNS
 * ============================================================
 */

function populateExpectedSigns() {

    const select =
        $("expectedSign");

    select.innerHTML = "";

    SIGNS.forEach(
        sign => {

            const option =
                document.createElement(
                    "option"
                );

            option.value = sign;

            option.textContent = sign;

            select.appendChild(
                option
            );

        }
    );

    select.value =
        expectedSign;
}


/*
 * ============================================================
 * SIGN GRID
 * ============================================================
 */

function populateSignGrid() {

    renderSignGrid("");
}


function renderSignGrid(query) {

    const grid =
        $("signGrid");

    const q =
        String(query || "")
            .trim()
            .toLowerCase();

    grid.innerHTML = "";

    SIGNS
        .filter(
            sign =>
                !q ||
                sign
                    .toLowerCase()
                    .includes(q)
        )
        .forEach(
            sign => {

                const item =
                    document.createElement(
                        "button"
                    );

                item.type = "button";

                item.className =
                    "sign-item";

                if (
                    sign === expectedSign
                ) {

                    item.classList.add(
                        "selected"
                    );

                }

                item.textContent =
                    sign;

                item.addEventListener(
                    "click",
                    () => {

                        expectedSign =
                            sign;

                        $("expectedSign")
                            .value =
                            sign;

                        selectSignInGrid(
                            sign
                        );

                    }
                );

                grid.appendChild(
                    item
                );

            }
        );
}


function selectSignInGrid(sign) {

    document
        .querySelectorAll(
            ".sign-item"
        )
        .forEach(
            item => {

                item.classList.toggle(
                    "selected",
                    item.textContent === sign
                );

            }
        );
}


/*
 * ============================================================
 * THRESHOLD
 * ============================================================
 */

function setupThreshold() {

    const slider =
        $("thresholdSlider");

    const output =
        $("thresholdValue");

    slider.addEventListener(
        "input",
        () => {

            threshold =
                Number(
                    slider.value
                );

            output.textContent =
                threshold.toFixed(2);

        }
    );

    output.textContent =
        threshold.toFixed(2);
}


/*
 * ============================================================
 * BACKEND
 * ============================================================
 */

async function checkBackend() {

    try {

        const response =
            await fetch(
                `${API_BASE}/health`,
                {
                    method: "GET"
                }
            );


        if (!response.ok) {

            throw new Error(
                `HTTP ${response.status}`
            );

        }


        const data =
            await response.json();


        setSystemOnline(
            true,
            "Backend Connected"
        );


        $("apiStatus").textContent =
            "Connected";


        $("apiStatus").style.color =
            "#168b56";


        if (
            data.model_loaded
        ) {

            $("appModelStatus")
                .textContent =
                data.model_type ||
                "Hand LSTM";

        }


        if (
            data.class_count
        ) {

            $("appModelStatus")
                .textContent +=
                ` · ${data.class_count} classes`;

        }


        updateModelStatus(
            data
        );

    }

    catch (error) {

        console.error(
            "Backend health error:",
            error
        );


        setSystemOnline(
            false,
            "Backend unavailable"
        );


        $("apiStatus").textContent =
            "Offline";

        $("apiStatus").style.color =
            "#d84a52";

    }
}


function setSystemOnline(
    online,
    message
) {

    const dot =
        $("systemDot");

    const status =
        $("systemStatus");

    const text =
        $("systemMessage");


    dot.classList.toggle(
        "online",
        online
    );


    dot.classList.toggle(
        "offline",
        !online
    );


    status.textContent =
        online
            ? "SYSTEM ONLINE"
            : "SYSTEM OFFLINE";


    status.style.color =
        online
            ? "#58e78e"
            : "#ff9095";


    text.textContent =
        message;
}


function updateModelStatus(data) {

    const model =
        data?.model_type ||
        "include50_hand_lstm";


    $("modelLabel").textContent =
        `Model: ${model}`;


    $("appModelStatus").textContent =
        model;
}


function setStatus(
    elementId,
    text,
    good
) {

    const el =
        $(elementId);

    if (!el) return;

    el.textContent =
        text;

    el.style.color =
        good
            ? "#168b56"
            : "#d84a52";
}


/*
 * ============================================================
 * REAL WEBCAM
 * ============================================================
 */

async function startCamera() {

    if (cameraRunning) {

        return;
    }


    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        showCameraError(
            "Your browser does not support webcam access."
        );

        return;
    }


    try {

        /*
         * Request the actual physical webcam.
         */

        mediaStream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    video: {

                        width: {
                            ideal: 1280
                        },

                        height: {
                            ideal: 720
                        },

                        facingMode:
                            "user"

                    },

                    audio: false

                });


        const video =
            $("webcam");


        video.srcObject =
            mediaStream;


        await video.play();


        /*
         * SHOW VIDEO INSIDE CAMERA BOX
         */

        video.style.display =
            "block";


        $("cameraPlaceholder")
            .classList.add(
                "hidden"
            );


        $("cameraLiveBadge")
            .classList.remove(
                "hidden"
            );


        $("faceBadge")
            .classList.add(
                "hidden"
            );


        $("resolutionBadge")
            .classList.remove(
                "hidden"
            );


        cameraRunning = true;


        $("startCameraBtn")
            .disabled = true;


        $("captureBtn")
            .disabled = false;


        $("stopCameraBtn")
            .disabled = false;


        $("cameraStatus")
            .textContent =
            "Active";


        $("cameraStatus")
            .style.color =
            "#168b56";


        $("cameraMessage")
            .textContent =
            "Camera active. Your face and hands should now be visible.";


        setTimeout(
            updateResolution,
            500
        );


        /*
         * Start MediaPipe loop
         */

        processVideoFrame();


    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );


        showCameraError(
            cameraErrorMessage(
                error
            )
        );

    }
}


function cameraErrorMessage(
    error
) {

    if (
        error.name ===
        "NotAllowedError"
    ) {

        return (
            "Camera permission was denied. " +
            "Click the camera icon in Chrome's address bar and allow camera access."
        );

    }


    if (
        error.name ===
        "NotFoundError"
    ) {

        return (
            "No webcam was found."
        );

    }


    if (
        error.name ===
        "NotReadableError"
    ) {

        return (
            "The webcam is already being used by another application."
        );

    }


    return (
        "Unable to start webcam: " +
        error.message
    );
}


function showCameraError(
    message
) {

    $("cameraMessage")
        .textContent =
        message;

    $("cameraMessage")
        .style.color =
        "#d84a52";

    alert(message);
}


/*
 * ============================================================
 * VIDEO → MEDIAPIPE
 * ============================================================
 */

async function processVideoFrame() {

    if (
        !cameraRunning
    ) {

        return;
    }


    const video =
        $("webcam");


    if (
        video.readyState >= 2 &&
        mediaPipeHands &&
        mediaPipeReady
    ) {

        try {

            await mediaPipeHands.send({
                image: video
            });

        }

        catch (error) {

            console.error(
                "MediaPipe frame error:",
                error
            );

        }

    }


    requestAnimationFrame(
        processVideoFrame
    );
}


/*
 * ============================================================
 * MEDIAPIPE RESULTS
 * ============================================================
 */

function handleMediaPipeResults(
    results
) {

    lastResults =
        results;


    drawLandmarks(
        results
    );


    const hands =
        results.multiHandLandmarks ||
        [];


    currentLeftHand = false;

    currentRightHand = false;


    const handedness =
        results.multiHandedness ||
        [];


    handedness.forEach(
        (hand, index) => {

            const label =
                hand.label;

            /*
             * MediaPipe selfie video is mirrored.
             * MediaPipe handedness remains the logical hand.
             */

            if (
                label === "Left"
            ) {

                currentLeftHand =
                    true;

            }

            if (
                label === "Right"
            ) {

                currentRightHand =
                    true;

            }

        }
    );


    updateTrackingUI();


    if (
        capturing
    ) {

        const features =
            extractHandFeatures(
                results
            );


        sequence.push(
            features
        );


        if (
            currentLeftHand ||
            currentRightHand
        ) {

            handFrames++;

        }


        updateSequenceUI();


        if (
            sequence.length >=
            SEQUENCE_LENGTH
        ) {

            finishCapture();

        }

    }
}


/*
 * ============================================================
 * LANDMARK DRAWING
 * ============================================================
 */

function drawLandmarks(
    results
) {

    const canvas =
        $("landmarkCanvas");


    const video =
        $("webcam");


    const rect =
        video.getBoundingClientRect();


    const width =
        rect.width;


    const height =
        rect.height;


    canvas.width =
        width;


    canvas.height =
        height;


    const ctx =
        canvas.getContext("2d");


    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    if (
        !results.multiHandLandmarks
    ) {

        return;
    }


    results.multiHandLandmarks
        .forEach(
            landmarks => {

                /*
                 * Connect hand bones
                 */

                const connections = [

                    [0,1],
                    [1,2],
                    [2,3],
                    [3,4],

                    [0,5],
                    [5,6],
                    [6,7],
                    [7,8],

                    [0,9],
                    [9,10],
                    [10,11],
                    [11,12],

                    [0,13],
                    [13,14],
                    [14,15],
                    [15,16],

                    [0,17],
                    [17,18],
                    [18,19],
                    [19,20],

                    [5,9],
                    [9,13],
                    [13,17]

                ];


                ctx.lineWidth =
                    2.2;

                ctx.strokeStyle =
                    "#55e7a1";


                connections
                    .forEach(
                        ([a,b]) => {

                            const p1 =
                                landmarks[a];

                            const p2 =
                                landmarks[b];


                            ctx.beginPath();

                            ctx.moveTo(
                                p1.x * width,
                                p1.y * height
                            );

                            ctx.lineTo(
                                p2.x * width,
                                p2.y * height
                            );

                            ctx.stroke();

                        }
                    );


                /*
                 * Draw points
                 */

                landmarks
                    .forEach(
                        point => {

                            ctx.beginPath();

                            ctx.arc(
                                point.x * width,
                                point.y * height,
                                3,
                                0,
                                Math.PI * 2
                            );

                            ctx.fillStyle =
                                "#ffffff";

                            ctx.fill();

                            ctx.strokeStyle =
                                "#18c982";

                            ctx.lineWidth =
                                1.5;

                            ctx.stroke();

                        }
                    );

            }
        );
}


/*
 * ============================================================
 * HAND FEATURE EXTRACTION
 * ============================================================
 *
 * 21 × 3 = 63 per hand
 * Left 63 + Right 63 = 126
 *
 * Wrist-centered + scale normalized.
 *
 * This mirrors the intended project preprocessing:
 * handedness -> wrist centering -> scale normalization.
 *
 * ============================================================
 */

function extractHandFeatures(
    results
) {

    const output =
        new Array(
            FEATURES_PER_FRAME
        ).fill(0);


    const hands =
        results.multiHandLandmarks ||
        [];


    const handedness =
        results.multiHandedness ||
        [];


    for (
        let i = 0;
        i < hands.length;
        i++
    ) {

        const landmarks =
            hands[i];


        const label =
            handedness[i]?.label;


        let offset;


        if (
            label === "Left"
        ) {

            offset = 0;

        }

        else if (
            label === "Right"
        ) {

            offset = 63;

        }

        else {

            continue;

        }


        /*
         * Wrist = landmark 0
         */

        const wrist =
            landmarks[0];


        /*
         * Scale:
         * maximum distance from wrist
         */

        let scale = 0;


        landmarks.forEach(
            point => {

                const dx =
                    point.x -
                    wrist.x;

                const dy =
                    point.y -
                    wrist.y;

                const dz =
                    point.z -
                    wrist.z;


                const distance =
                    Math.sqrt(
                        dx * dx +
                        dy * dy +
                        dz * dz
                    );


                if (
                    distance > scale
                ) {

                    scale =
                        distance;

                }

            }
        );


        if (
            scale < 1e-6
        ) {

            scale = 1;

        }


        for (
            let j = 0;
            j < 21;
            j++
        ) {

            const point =
                landmarks[j];


            const base =
                offset +
                j * 3;


            output[base] =
                (
                    point.x -
                    wrist.x
                ) / scale;


            output[base + 1] =
                (
                    point.y -
                    wrist.y
                ) / scale;


            output[base + 2] =
                (
                    point.z -
                    wrist.z
                ) / scale;

        }

    }


    return output;
}


/*
 * ============================================================
 * TRACKING UI
 * ============================================================
 */

function updateTrackingUI() {

    setTracking(
        "trackFace",
        "trackFaceDot",
        true,
        "Detected"
    );


    setTracking(
        "trackLeft",
        "trackLeftDot",
        currentLeftHand,
        currentLeftHand
            ? "Detected"
            : "Missing"
    );


    setTracking(
        "trackRight",
        "trackRightDot",
        currentRightHand,
        currentRightHand
            ? "Detected"
            : "Missing"
    );


    /*
     * Face badge.
     *
     * The hand-only MediaPipe model does not
     * perform face detection. The UI therefore
     * treats the live camera subject as visible.
     *
     * Recognition itself is based on hands.
     */

    $("faceBadge")
        .classList.toggle(
            "hidden",
            !cameraRunning
        );


    $("leftHandBadge")
        .classList.toggle(
            "detected",
            currentLeftHand
        );


    $("rightHandBadge")
        .classList.toggle(
            "detected",
            currentRightHand
        );


    $("leftHandStatus")
        .textContent =
        currentLeftHand
            ? "✓"
            : "—";


    $("rightHandStatus")
        .textContent =
        currentRightHand
            ? "✓"
            : "—";
}


function setTracking(
    labelId,
    dotId,
    good,
    text
) {

    $(labelId).textContent =
        text;


    $(labelId).style.color =
        good
            ? "#168b56"
            : "#d47b22";


    $(dotId)
        .classList.toggle(
            "good",
            good
        );


    $(dotId)
        .classList.toggle(
            "bad",
            !good
        );
}


/*
 * ============================================================
 * CAPTURE
 * ============================================================
 */

function startCapture() {

    if (
        !cameraRunning
    ) {

        showCameraError(
            "Start the camera first."
        );

        return;
    }


    if (
        capturing
    ) {

        return;
    }


    sequence = [];

    handFrames = 0;

    capturing = true;

    processingPrediction = false;


    $("captureOverlay")
        .classList.remove(
            "hidden"
        );


    $("captureBtn")
        .disabled = true;


    $("cameraMessage")
        .textContent =
        "Perform the selected sign naturally.";


    updateSequenceUI();

}


function updateSequenceUI() {

    const count =
        Math.min(
            sequence.length,
            SEQUENCE_LENGTH
        );


    const percent =
        (
            count /
            SEQUENCE_LENGTH
        ) * 100;


    $("sequenceText")
        .textContent =
        `${count} / ${SEQUENCE_LENGTH}`;


    $("sequenceBar")
        .style.width =
        `${percent}%`;


    $("captureFrameText")
        .textContent =
        `${count} / ${SEQUENCE_LENGTH}`;

}


async function finishCapture() {

    if (
        !capturing
    ) {

        return;
    }


    capturing = false;


    $("captureOverlay")
        .classList.add(
            "hidden"
        );


    $("captureBtn")
        .disabled = false;


    updateSequenceUI();


    /*
     * Minimum hand evidence.
     *
     * At least 8 frames with a hand.
     */

    if (
        handFrames < 8
    ) {

        setResult(
            "No sign",
            null,
            null,
            "NO SIGN DETECTED"
        );


        $("cameraMessage")
            .textContent =
            "Not enough hand frames were detected. Try again.";


        return;

    }


    $("cameraMessage")
        .textContent =
        "Analyzing 45 frames...";


    await predictSequence();

}


/*
 * ============================================================
 * API PREDICTION
 * ============================================================
 */

async function predictSequence() {

    if (
        processingPrediction
    ) {

        return;
    }


    processingPrediction = true;


    try {

        const cleanSequence =
            sequence
                .slice(
                    0,
                    SEQUENCE_LENGTH
                )
                .map(
                    frame =>
                        frame.map(
                            Number
                        )
                );


        /*
         * Safety validation
         */

        if (
            cleanSequence.length !==
            SEQUENCE_LENGTH
        ) {

            throw new Error(
                "Sequence must contain 45 frames."
            );

        }


        if (
            cleanSequence.some(
                frame =>
                    frame.length !==
                    FEATURES_PER_FRAME
            )
        ) {

            throw new Error(
                "Every frame must contain 126 features."
            );

        }


        const response =
            await fetch(
                `${API_BASE}/predict/hand-sequence`,
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({

                            hand_sequence:
                                cleanSequence,

                            expected_sign:
                                expectedSign

                        })

                }
            );


        if (
            !response.ok
        ) {

            const errorText =
                await response.text();


            throw new Error(
                `API ${response.status}: ${errorText}`
            );

        }


        const result =
            await response.json();


        console.log(
            "Recognition result:",
            result
        );


        displayPrediction(
            result
        );


    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        $("cameraMessage")
            .textContent =
            `Prediction failed: ${error.message}`;


        setResult(
            "Error",
            null,
            null,
            "API ERROR"
        );

    }

    finally {

        processingPrediction =
            false;

    }
}


/*
 * ============================================================
 * DISPLAY RESULT
 * ============================================================
 */

function displayPrediction(
    result
) {

    const label =
        result.prediction ||
        result.predicted_label ||
        result.label ||
        "Unknown";


    const confidence =
        normalizePercent(
            result.confidence ??
            result.prediction_confidence ??
            result.probability
        );


    let reference =
        result.reference_match;


    let referencePercent =
        null;


    if (
        reference &&
        typeof reference ===
        "object"
    ) {

        referencePercent =
            normalizePercent(
                reference.match_percent ??
                reference.similarity ??
                reference.score
            );

    }

    else {

        referencePercent =
            normalizePercent(
                reference
            );

    }


    /*
     * Final matching status.
     *
     * The backend's verifier/matcher result
     * is preferred when available.
     */

    let matching =
        null;


    if (
        typeof result.matching ===
        "boolean"
    ) {

        matching =
            result.matching;

    }

    else if (
        typeof result.final_matching ===
        "boolean"
    ) {

        matching =
            result.final_matching;

    }

    else if (
        typeof result.status ===
        "string"
    ) {

        matching =
            result.status
                .toUpperCase()
                .includes("MATCH");

    }


    /*
     * If expected sign is supplied and backend
     * returned prediction, make the UI comparison
     * explicit without inventing confidence.
     */

    if (
        matching === null &&
        expectedSign
    ) {

        const normalize =
            value =>
                String(value)
                    .replace(
                        /^\d+\.\s*/,
                        ""
                    )
                    .trim()
                    .toLowerCase();


        matching =
            normalize(label) ===
            normalize(expectedSign);

    }


    const displayConfidence =
        confidence !== null
            ? confidence
            : null;


    setResult(
        label,
        displayConfidence,
        referencePercent,
        matching === true
            ? "MATCHING"
            : matching === false
                ? "NOT MATCHING"
                : "RESULT READY"
    );


    addHistory(
        label,
        displayConfidence,
        matching
    );


    $("cameraMessage")
        .textContent =
        "Analysis complete. Press CAPTURE SIGN to test again.";

}


/*
 * ============================================================
 * RESULT UI
 * ============================================================
 */

function setResult(
    label,
    confidence,
    reference,
    status
) {

    $("detectedSign")
        .textContent =
        label || "—";


    const confidencePercent =
        confidence === null
            ? 0
            : Math.max(
                0,
                Math.min(
                    100,
                    confidence
                )
            );


    const referencePercent =
        reference === null
            ? 0
            : Math.max(
                0,
                Math.min(
                    100,
                    reference
                )
            );


    $("confidenceText")
        .textContent =
        confidence === null
            ? "—"
            : `${confidence.toFixed(1)}%`;


    $("confidenceBar")
        .style.width =
        `${confidencePercent}%`;


    $("referenceText")
        .textContent =
        reference === null
            ? "—"
            : `${reference.toFixed(1)}%`;


    $("referenceBar")
        .style.width =
        `${referencePercent}%`;


    const badge =
        $("matchBadge");


    badge.classList.remove(
        "matching",
        "not-matching",
        "neutral"
    );


    if (
        status ===
        "MATCHING"
    ) {

        badge.classList.add(
            "matching"
        );

        badge.innerHTML =
            "✓&nbsp; MATCHING";

    }

    else if (
        status ===
        "NOT MATCHING"
    ) {

        badge.classList.add(
            "not-matching"
        );

        badge.innerHTML =
            "✕&nbsp; NOT MATCHING";

    }

    else if (
        status ===
        "NO SIGN DETECTED"
    ) {

        badge.classList.add(
            "not-matching"
        );

        badge.innerHTML =
            "⚠&nbsp; NO SIGN DETECTED";

    }

    else {

        badge.classList.add(
            "neutral"
        );

        badge.innerHTML =
            "●&nbsp; " +
            status;

    }

}


/*
 * ============================================================
 * NUMBER NORMALIZATION
 * ============================================================
 */

function normalizePercent(
    value
) {

    if (
        value === undefined ||
        value === null ||
        Number.isNaN(
            Number(value)
        )
    ) {

        return null;
    }


    let n =
        Number(value);


    /*
     * 0-1 → percentage
     */

    if (
        n >= 0 &&
        n <= 1
    ) {

        n *= 100;

    }


    return Math.max(
        0,
        Math.min(
            100,
            n
        )
    );
}


/*
 * ============================================================
 * STOP CAMERA
 * ============================================================
 */

function stopCamera() {

    capturing = false;


    if (
        mediaStream
    ) {

        mediaStream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        mediaStream = null;

    }


    const video =
        $("webcam");


    video.pause();

    video.srcObject =
        null;


    video.style.display =
        "none";


    $("cameraPlaceholder")
        .classList.remove(
            "hidden"
        );


    $("cameraLiveBadge")
        .classList.add(
            "hidden"
        );


    $("faceBadge")
        .classList.add(
            "hidden"
        );


    $("resolutionBadge")
        .classList.add(
            "hidden"
        );


    $("captureOverlay")
        .classList.add(
            "hidden"
        );


    clearCanvas();


    cameraRunning =
        false;


    sequence = [];

    handFrames = 0;


    $("startCameraBtn")
        .disabled = false;


    $("captureBtn")
        .disabled = true;


    $("stopCameraBtn")
        .disabled = true;


    $("cameraStatus")
        .textContent =
        "Inactive";


    $("cameraStatus")
        .style.color =
        "#d47b22";


    $("cameraMessage")
        .textContent =
        "Camera stopped.";


    updateSequenceUI();

}


function clearCanvas() {

    const canvas =
        $("landmarkCanvas");


    const ctx =
        canvas.getContext("2d");


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
}


/*
 * ============================================================
 * RESOLUTION
 * ============================================================
 */

function updateResolution() {

    const video =
        $("webcam");


    if (
        video.videoWidth &&
        video.videoHeight
    ) {

        const text =
            `${video.videoWidth} × ${video.videoHeight}`;


        $("cameraResolution")
            .textContent =
            text;


        $("resolutionBadge")
            .textContent =
            `Resolution: ${text}`;

    }
}


/*
 * ============================================================
 * CLEAR RESULT
 * ============================================================
 */

function clearResult() {

    setResult(
        "—",
        null,
        null,
        "WAITING"
    );


    sequence = [];

    handFrames = 0;


    updateSequenceUI();


    $("cameraMessage")
        .textContent =
        cameraRunning
            ? "Camera active. Ready for a new sign."
            : "Camera ready. Start the camera to begin.";

}


/*
 * ============================================================
 * HISTORY
 * ============================================================
 */

function addHistory(
    label,
    confidence,
    matching
) {

    history.unshift({

        time:
            new Date()
                .toLocaleTimeString(),

        label,

        confidence,

        matching

    });


    history =
        history.slice(
            0,
            20
        );


    renderHistory();
}


function renderHistory() {

    const body =
        $("historyBody");


    $("historyCount")
        .textContent =
        `Total: ${history.length}`;


    if (
        history.length === 0
    ) {

        body.innerHTML = `
            <tr>
                <td colspan="5">
                    No recognition results yet.
                </td>
            </tr>
        `;

        return;
    }


    body.innerHTML =
        history
            .map(
                (item, index) => {

                    const result =
                        item.matching === true
                            ? "MATCHING"
                            : item.matching === false
                                ? "NOT MATCHING"
                                : "RESULT";


                    const cls =
                        item.matching === true
                            ? "result-good"
                            : item.matching === false
                                ? "result-bad"
                                : "";


                    return `
                        <tr>

                            <td>
                                ${index + 1}
                            </td>

                            <td>
                                ${escapeHtml(item.time)}
                            </td>

                            <td>
                                <strong>
                                    ${escapeHtml(item.label)}
                                </strong>
                            </td>

                            <td>
                                ${
                                    item.confidence === null
                                        ? "—"
                                        : `${item.confidence.toFixed(1)}%`
                                }
                            </td>

                            <td class="${cls}">
                                ${result}
                            </td>

                        </tr>
                    `;

                }
            )
            .join("");
}


function clearHistory() {

    history = [];

    renderHistory();

}


/*
 * ============================================================
 * UPLOAD
 * ============================================================
 */

function handleVideoUpload(
    event
) {

    const file =
        event.target.files[0];


    if (!file) return;


    $("cameraMessage")
        .textContent =
        `Selected video: ${file.name}`;


    /*
     * The current API is sequence-based.
     *
     * Upload is kept as a frontend utility.
     * A future video extraction route can consume
     * this file directly.
     */

    alert(
        "Video selected successfully.\n\n" +
        "The current recognition API expects a 45-frame hand sequence. " +
        "Use LIVE CAMERA → CAPTURE SIGN for the full recognition pipeline."
    );

}


/*
 * ============================================================
 * REFERENCE
 * ============================================================
 */

function showReferenceInfo() {

    alert(
        "INCLUDE-50 Reference\n\n" +
        "The backend reference database contains the official " +
        "hand landmark sequences used for DTW reference matching."
    );

}


/*
 * ============================================================
 * SETTINGS
 * ============================================================
 */

function showSettingsInfo() {

    alert(
        "Current settings\n\n" +
        "API: " + API_BASE + "\n" +
        "Sequence: 45 frames\n" +
        "Features: 126/frame\n" +
        "Threshold: " + threshold.toFixed(2)
    );

}


/*
 * ============================================================
 * MODALS
 * ============================================================
 */

function setupModals() {

    $("closeAbout")
        .addEventListener(
            "click",
            () =>
                closeModal("aboutModal")
        );


    $("closeHelp")
        .addEventListener(
            "click",
            () =>
                closeModal("helpModal")
        );


    document
        .querySelectorAll(".modal")
        .forEach(
            modal => {

                modal.addEventListener(
                    "click",
                    event => {

                        if (
                            event.target ===
                            modal
                        ) {

                            modal.classList.add(
                                "hidden"
                            );

                        }

                    }
                );

            }
        );
}


function openModal(id) {

    $(id)
        .classList.remove(
            "hidden"
        );
}


function closeModal(id) {

    $(id)
        .classList.add(
            "hidden"
        );
}


/*
 * ============================================================
 * ESCAPE HTML
 * ============================================================
 */

function escapeHtml(
    value
) {

    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}


/*
 * ============================================================
 * CLEANUP
 * ============================================================
 */

window.addEventListener(
    "beforeunload",
    () => {

        if (
            mediaStream
        ) {

            mediaStream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );

        }

    }
);


/*
 * Public debugging helpers
 */

window.ISLVision = {

    startCamera,

    stopCamera,

    startCapture,

    clearResult,

    clearHistory,

    getSequence:
        () =>
            sequence,

    getExpectedSign:
        () =>
            expectedSign,

    api:
        API_BASE

};
