import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
from math import ceil

# optional import for model inference
try:
    import tensorflow as tf
    from model_utils import preprocess_image_for_model, load_model_if_exists
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from model_utils import heuristic_rotten_mask, heuristic_rotten_score

st.set_page_config(page_title="Apple Health Detector", layout="centered")

st.title("Apple Health Detector — Good vs Bad Apple")
st.write("Upload images or videos. Use the heuristic demo for quick checks or a trained model for better accuracy.")

with st.sidebar:
    st.header("Settings")

    mode = st.selectbox(
        "Mode",
        [
            "Heuristic (images)",
            "Model (images, models/model.h5)",
            "Heuristic (videos)",
            "Model (videos, models/model.h5)",
            "Train a model (script)"
        ],
        index=0
    )

    # Sampling controls for videos
    if "videos" in mode:
        st.write("Video sampling settings")
        target_fps = st.slider("Frames per second to analyze", 1, 10, 3)
        max_frames = st.slider("Max frames per video to analyze", 30, 600, 150)

    # Heuristic knobs
    if mode.startswith("Heuristic"):
        st.write("Heuristic thresholds")
        brown_weight = st.slider("Brownness weight", 0.0, 1.0, 0.7)
        dark_weight = st.slider("Darkness weight", 0.0, 1.0, 0.3)
        score_threshold = st.slider("Rot detection threshold (lower = more sensitive)", 0.01, 0.5, 0.08)

    # Model info
    if mode.startswith("Model"):
        st.write("TensorFlow available: {}".format(TF_AVAILABLE))
        st.caption("Model path: models/model.h5")

    st.markdown("---")
    st.write("Dataset structure for training (train.py):")
    st.text("dataset/\n  train/\n    good/\n    bad/\n  val/\n    good/\n    bad/")

def analyze_frame_heuristic(frame_rgb, brown_weight, dark_weight):
    mask = heuristic_rotten_mask(frame_rgb)
    score = heuristic_rotten_score(frame_rgb, brown_weight=brown_weight, dark_weight=dark_weight)
    return mask, score

def prob_bad_from_preds(preds):
    # Support sigmoid or 2-class softmax
    if np.isscalar(preds):
        return float(preds)
    preds = np.array(preds).ravel()
    if preds.size == 1:
        return float(preds[0])
    elif preds.size >= 2:
        # assume preds[0] = prob_bad
        return float(preds[0])
    return float(preds.mean())

def analyze_frame_model(frame_rgb, model):
    x = preprocess_image_for_model(frame_rgb, target_size=(224, 224))
    out = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    return prob_bad_from_preds(out)

def sample_video_frames(cap, target_fps=3, max_frames=150):
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(input_fps / target_fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    idx = 0
    collected = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            collected += 1
            if collected >= max_frames:
                break
        idx += 1
    return frames, total

# ======== IMAGE MODES ========
if mode in ("Heuristic (images)", "Model (images, models/model.h5)"):
    uploaded = st.file_uploader("Upload one or more images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload at least one image to get started.")
    else:
        model = None
        if mode.startswith("Model"):
            if not TF_AVAILABLE:
                st.error("TensorFlow is not available. Install tensorflow to use model inference.")
            else:
                model = load_model_if_exists("models/model.h5")
                if model is None:
                    st.warning("No model at models/model.h5. Use train.py to train and save a model there.")

        for uploaded_file in uploaded:
            st.write("###", uploaded_file.name)
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
            img_np = np.array(image)

            if mode.startswith("Heuristic"):
                mask = heuristic_rotten_mask(img_np)
                rotten_score = heuristic_rotten_score(img_np, brown_weight=brown_weight, dark_weight=dark_weight)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Rot/bruise mask")
                    st.image(mask, use_container_width=True)
                with col2:
                    label = "BAD (rotten/bruised)" if rotten_score >= score_threshold else "GOOD"
                    st.metric("Prediction", label)
                    st.caption(f"Heuristic score: {rotten_score:.3f} (threshold: {score_threshold:.3f})")

            else:
                if model is None:
                    st.stop()
                prob_bad = analyze_frame_model(img_np, model)
                pred_label = "BAD" if prob_bad >= 0.5 else "GOOD"
                st.metric("Prediction", pred_label, delta=f"bad: {prob_bad:.3f}")

# ======== VIDEO MODES ========
elif mode in ("Heuristic (videos)", "Model (videos, models/model.h5)"):
    videos = st.file_uploader("Upload one or more videos", accept_multiple_files=True,
                              type=["mp4", "mov", "avi", "mkv"])
    if not videos:
        st.info("Upload at least one video.")
    else:
        model = None
        if mode.startswith("Model"):
            if not TF_AVAILABLE:
                st.error("TensorFlow is not available. Install tensorflow to use model inference.")
            else:
                model = load_model_if_exists("models/model.h5")
                if model is None:
                    st.warning("No model at models/model.h5. Use train.py to train and save a model there.")

        for vid in videos:
            st.write("###", vid.name)

            # Save to a temp file for OpenCV
            suffix = os.path.splitext(vid.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                tfile.write(vid.read())
                tmp_path = tfile.name

            # Show the raw video player
            st.video(tmp_path)

            cap = cv2.VideoCapture(tmp_path)
            frames, total_frames = sample_video_frames(cap, target_fps=target_fps, max_frames=max_frames)
            cap.release()

            if len(frames) == 0:
                st.error("Could not read frames from the video.")
                continue

            st.caption(f"Analyzing ~{len(frames)} sampled frames (of {total_frames} total).")
            progress = st.progress(0)

            frame_results = []
            example_images = []  # store a handful of visualizations

            for i, fr in enumerate(frames):
                if mode.startswith("Heuristic"):
                    mask, score = analyze_frame_heuristic(fr, brown_weight, dark_weight)
                    is_bad = float(score >= score_threshold)
                    frame_results.append(is_bad)

                    # keep some examples (every Nth frame)
                    if i % max(1, len(frames)//6) == 0:
                        # stack side-by-side
                        vis = np.concatenate([fr, mask], axis=1)
                        example_images.append(vis)

                else:
                    if model is None:
                        st.warning("Model not loaded.")
                        break
                    prob_bad = analyze_frame_model(fr, model)
                    is_bad = float(prob_bad >= 0.5)
                    frame_results.append(is_bad)
                    if i % max(1, len(frames)//6) == 0:
                        # put text on the frame
                        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                        txt = f"bad={prob_bad:.2f}"
                        cv2.putText(bgr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        example_images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                progress.progress((i+1)/len(frames))

            # Aggregate decision at video level
            bad_ratio = float(np.mean(frame_results)) if frame_results else 0.0
            verdict = "BAD" if bad_ratio >= 0.5 else "GOOD"

            colA, colB = st.columns(2)
            with colA:
                st.metric("Video-level Prediction", verdict)
            with colB:
                st.metric("Bad-frame ratio", f"{bad_ratio:.2f}")

            st.write("Examples from the analysis")
            for img in example_images:
                st.image(img, use_container_width=True)

# ======== TRAIN ========
else:
    st.info("Training isn't run inside the app.\n\nUse:\n`python train.py --dataset dataset --save_path models/model.h5`")
