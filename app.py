import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("best_model.keras")

class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

st.title("Multi-class Fish Classification")
st.write("Upload an image of a fish to predict its category.")

uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict Fish Category"):

        # Preprocess image
        image = image.resize((224,224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Model prediction
        prediction = model.predict(image)

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.subheader("Prediction Result")
        st.success(f"Fish Category: {class_names[class_index]}")
        st.write("Confidence Score:", f"{confidence*100:.2f}%")