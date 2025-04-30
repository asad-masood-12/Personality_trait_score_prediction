import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2
import plotly.graph_objects as go

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image):
    try:
        gray_image = np.array(image.convert('L'))
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            st.warning("No face detected.")
            return None
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        return image.crop((x, y, x + w, y + h))
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return None

def load_and_preprocess_image(image, target_size=(224, 224)):
    try:
        face = detect_and_crop_face(image)
        if face is None:
            return None
        face = face.resize(target_size)
        face = np.array(face) / 255.0
        return np.expand_dims(face, axis=0)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

def predict_personality_traits(model_path, image):
    model = load_model(model_path)
    processed_image = load_and_preprocess_image(image)
    if processed_image is None:
        return None
    predictions = model.predict(processed_image)[0]
    traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    return {trait: float(score) for trait, score in zip(traits, predictions)}

def draw_gauge(trait, value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': trait},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 20], 'color': "#FFDDDD"},
                {'range': [20, 40], 'color': "#FFAAAA"},
                {'range': [40, 60], 'color': "#FFFFAA"},
                {'range': [60, 80], 'color': "#AAFFAA"},
                {'range': [80, 100], 'color': "#66FF66"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Personality Trait Predictor", layout="centered")
    st.title('🧠 Personality Trait Prediction with Meters')
    st.markdown("Upload a **clear face image** to predict Big Five personality traits with interactive meter visuals.")

    uploaded_image = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        MODEL_PATH = r"D:\Asad\AUG AI\VGG16_Personality-Traits\Streamlit_VGG16\Model_2.0.h5"
        predictions = predict_personality_traits(MODEL_PATH, image)

        if predictions:
            st.subheader("🎯 Personality Trait Meters")

            # Move 'Extraversion' to second last
            traits_order = [trait for trait in predictions if trait != 'Extraversion']
            traits_order.insert(-1, 'Extraversion')

            for trait in traits_order:
                draw_gauge(trait, predictions[trait])
        else:
            st.error("❌ Prediction failed. Try a different image.")

if __name__ == "__main__":
    main()
