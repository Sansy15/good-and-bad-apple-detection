# app_improved.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
import tensorflow as tf
from typing import Optional, Tuple, List

# Try to import helper funcs from your model_utils (you already have this)
try:
    from model_utils import (
        preprocess_image_for_model,
        load_model_if_exists,
        heuristic_rotten_mask,
        heuristic_rotten_score
    )
except Exception:
    # minimal fallbacks if user didn't include model_utils
    def preprocess_image_for_model(img, target_size=(224,224)):
        img = cv2.resize(img, target_size)
        x = img.astype("float32") / 255.0
        return x

    def load_model_if_exists(path):
        if os.path.exists(path):
            return tf.keras.models.load_model(path, compile=False)
        return None

    def heuristic_rotten_mask(img):
        # simple threshold-based brown detection fallback
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([5, 50, 20]); upper = np.array([25, 255, 255])  # brown-ish range
        mask = cv2.inRange(hsv, lower, upper)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask_rgb

    def heuristic_rotten_score(img, brown_weight=0.7, dark_weight=0.3):
        mask = heuristic_rotten_mask(img)
        score = np.count_nonzero(mask[:,:,0]) / (mask.shape[0]*mask.shape[1])
        return float(score)

# Check TF availability
TF_AVAILABLE = True
try:
    _ = tf.__version__
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Apple Health Detector (improved)", layout="wide")
st.title("Apple Health Detector — Good vs Bad Apple (with visualization)")

with st.sidebar:
    st.header("Mode & Settings")
    mode = st.selectbox(
        "Mode",
        [
            "Heuristic (images)",
            "Model (images)",
            "Heuristic (videos)",
            "Model (videos)",
        ],
        index=1
    )

    if "videos" in mode:
        target_fps = st.slider("Frames per second to analyze", 1, 10, 3)
        max_frames = st.slider("Max frames (per video)", 30, 600, 150)

    if mode.startswith("Heuristic"):
        st.write("Heuristic thresholds")
        brown_weight = st.slider("Brownness weight", 0.0, 1.0, 0.7)
        dark_weight = st.slider("Darkness weight", 0.0, 1.0, 0.3)
        score_threshold = st.slider("Rot detection threshold", 0.01, 0.5, 0.08)

    if mode.startswith("Model"):
        st.write("Model inference")
        st.write(f"TensorFlow available: {TF_AVAILABLE}")
        model_path = st.text_input("Model path", "models/model.h5")
        explain_cam = st.checkbox("Show Grad-CAM overlays (model explainability)", value=True)

st.markdown("---")

# -------- Grad-CAM helper ----------
def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, 
                         last_conv_layer_name: Optional[str] = None) -> np.ndarray:
    """
    Produce Grad-CAM heatmap for a single preprocessed image array.
    img_array: preprocessed image (H,W,3) scaled 0..1
    model: a Keras model
    Returns a heatmap (H,W) normalized 0..1
    """
    # find a conv layer if not provided
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        # fallback: no conv layers
        return np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # convert to batch
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        # handle different output shapes (sigmoid vs softmax)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            # assume index 0 = prob_bad if two outputs; else take max class
            loss = tf.reduce_max(predictions, axis=-1)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)

    # global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    conv_outputs = conv_outputs.numpy()

    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
    heatmap /= np.max(heatmap)
    # resize to original size
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    return heatmap

def overlay_heatmap_on_image(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    rgb_img: HxWx3 (0..255)
    heatmap: HxW (0..1)
    """
    heat = np.uint8(255 * heatmap)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (heat_color.astype(np.float32) * alpha + rgb_img.astype(np.float32) * (1 - alpha))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

# --------- utilities ----------
def prob_bad_from_preds(preds: np.ndarray) -> float:
    preds = np.array(preds).ravel()
    if preds.size == 0:
        return 0.0
    if preds.size == 1:
        return float(preds[0])
    # if two-output softmax, assume index 0 = bad
    return float(preds[0])

def analyze_img_with_model(rgb_img: np.ndarray, model: tf.keras.Model) -> Tuple[float, Optional[np.ndarray]]:
    x = preprocess_image_for_model(rgb_img, target_size=(224,224))
    prob = prob_bad_from_preds(model.predict(np.expand_dims(x, axis=0), verbose=0)[0])
    return prob, x  # return preprocessed array for gradcam

# -------- Image mode ----------
if mode in ("Heuristic (images)", "Model (images)"):
    uploaded = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg","jpeg","png"])
    if uploaded:
        model = None
        if mode.startswith("Model"):
            if not TF_AVAILABLE:
                st.error("TensorFlow not available. Install it to use model mode.")
            else:
                model = load_model_if_exists(model_path)
                if model is None:
                    st.warning(f"No model found at {model_path}. Use train.py to create one.")

        for file in uploaded:
            st.write("####", file.name)
            image = Image.open(file).convert("RGB")
            img_np = np.array(image)
            col1, col2 = st.columns([1,1])
            with col1:
                st.image(img_np, use_container_width=True, caption="Original image")
            with col2:
                if mode.startswith("Heuristic"):
                    mask = heuristic_rotten_mask(img_np)
                    score = heuristic_rotten_score(img_np, brown_weight=brown_weight, dark_weight=dark_weight)
                    label = "BAD" if score >= score_threshold else "GOOD"
                    st.image(mask, use_container_width=True, caption=f"Rot mask — score {score:.3f}")
                    st.metric("Prediction", label)
                else:
                    if model is None:
                        st.write("Model not loaded.")
                        continue
                    prob_bad, pre_x = analyze_img_with_model(img_np, model)
                    label = "BAD" if prob_bad >= 0.5 else "GOOD"
                    st.metric("Prediction", label, delta=f"bad: {prob_bad:.3f}")

                    # Grad-CAM overlay (if checked)
                    if explain_cam:
                        heatmap = make_gradcam_heatmap(pre_x, model)
                        overlay = overlay_heatmap_on_image(img_np, heatmap, alpha=0.45)
                        st.image(overlay, use_container_width=True, caption="Grad-CAM overlay (where model looked)")

# -------- Video mode ----------
elif mode in ("Heuristic (videos)", "Model (videos)"):
    videos = st.file_uploader("Upload videos", accept_multiple_files=True, type=["mp4","mov","avi","mkv"])
    if videos:
        model = None
        if mode.startswith("Model"):
            if not TF_AVAILABLE:
                st.error("TensorFlow not available. Install it to use model mode.")
            else:
                model = load_model_if_exists(model_path)
                if model is None:
                    st.warning(f"No model found at {model_path}. Use train.py to create one.")

        for vid in videos:
            st.write("###", vid.name)
            # save to temp file for opencv
            suffix = os.path.splitext(vid.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                tfile.write(vid.read())
                tmp_path = tfile.name

            st.video(tmp_path)
            cap = cv2.VideoCapture(tmp_path)
            input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(input_fps / target_fps)))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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
            cap.release()

            if len(frames) == 0:
                st.error("Could not read frames from the video.")
                continue

            st.caption(f"Analyzing ~{len(frames)} sampled frames (of {total_frames} total).")
            progress = st.progress(0)
            frame_results = []
            example_images = []

            for i, fr in enumerate(frames):
                if mode.startswith("Heuristic"):
                    mask = heuristic_rotten_mask(fr)
                    score = heuristic_rotten_score(fr, brown_weight=brown_weight, dark_weight=dark_weight)
                    is_bad = float(score >= score_threshold)
                    frame_results.append(is_bad)
                    if i % max(1, len(frames)//6) == 0:
                        vis = np.concatenate([fr, mask], axis=1)
                        example_images.append(vis)
                else:
                    if model is None:
                        st.warning("Model not loaded.")
                        break
                    prob_bad, pre_x = analyze_img_with_model(fr, model)
                    is_bad = float(prob_bad >= 0.5)
                    frame_results.append(is_bad)
                    # make overlay example
                    if i % max(1, len(frames)//6) == 0:
                        if explain_cam:
                            heatmap = make_gradcam_heatmap(pre_x, model)
                            overlay = overlay_heatmap_on_image(fr, heatmap, alpha=0.45)
                            # draw text probability
                            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                            cv2.putText(overlay_bgr, f"bad={prob_bad:.2f}", (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                            example_images.append(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
                        else:
                            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                            cv2.putText(bgr, f"bad={prob_bad:.2f}", (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                            example_images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

                progress.progress((i+1)/len(frames))

            bad_ratio = float(np.mean(frame_results)) if frame_results else 0.0
            verdict = "BAD" if bad_ratio >= 0.5 else "GOOD"

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Video-level Prediction", verdict)
            with c2:
                st.metric("Bad-frame ratio", f"{bad_ratio:.2f}")

            st.write("Examples from the analysis")
            for im in example_images:
                st.image(im, use_container_width=True)

            # OPTIONAL: create a short result video with overlays (first N example frames)
            if example_images:
                out_fn = tmp_path + "_result_preview.mp4"
                h, w, _ = example_images[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_fn, fourcc, target_fps, (w, h))
                for im in example_images:
                    writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                writer.release()
                st.video(out_fn)

else:
    st.info("Pick a mode from the sidebar to begin. To train a model, run train.py externally.")
