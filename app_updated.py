
import streamlit as st
from PIL import Image
import io
import tempfile
import os

st.set_page_config(page_title="Image Detection / Segmentation App", layout="wide")

st.title("🩺 Automated Ear Disease Detection through Object Detection of Otoscopic Images")
st.write("Upload an image and (optionally) a model (.pt for Ultralytics YOLO). "
         "If you don't have a model handy, the app will run a dummy inference for demo.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    uploaded_image = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    model_file = st.file_uploader("Upload a model file (.pt) — optional", type=["pt","onnx","yaml"])
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    run = st.button("Run inference")

with col2:
    st.header("Preview / Result")
    if uploaded_image is None:
        st.info("Upload an image to get started. Use the sample screenshots (in repo) as a reference.")
    else:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Input image", use_column_width=True)
        if not run:
            st.caption("Click **Run inference** to detect/segment objects.")

def load_ultralytics_model(path):
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        return model
    except Exception as e:
        st.warning(f"Could not load Ultralytics model: {e}")
        return None

def dummy_inference_pil(image_pil):
    import PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
    im = image_pil.copy()
    draw = ImageDraw.Draw(im)
    w,h = im.size
    box = (int(w*0.15), int(h*0.15), int(w*0.75), int(h*0.75))
    draw.rectangle(box, outline="red", width=6)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=24)
    except:
        font = ImageFont.load_default()
    draw.text((box[0], box[1]-28), "demo_object:0.99", fill="red", font=font)
    return im

if run:
    if uploaded_image is None:
        st.error("Please upload an image first.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_image.getbuffer())
        tfile.flush()
        tfile.close()

        result_image = None
        model = None
        if model_file is not None:
            model_path = os.path.join(tempfile.gettempdir(), model_file.name)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())
            model = load_ultralytics_model(model_path)

        if model is not None:
            st.info("Running inference with Ultralytics YOLO model...")
            try:
                results = model.predict(source=tfile.name, conf=conf, save=False, verbose=False)
                r = results[0]
                annotated = r.plot()
                import numpy as np
                annotated_pil = Image.fromarray(annotated)
                result_image = annotated_pil
            except Exception as ex:
                st.warning(f"Could not plot results: {ex}")
        else:
            st.info("No compatible model loaded — running demo (dummy) inference.")
            result_image = dummy_inference_pil(Image.open(tfile.name).convert("RGB"))

        if result_image is not None:
            st.image(result_image, caption="Detection / Segmentation result", use_column_width=True)

st.markdown("---")
