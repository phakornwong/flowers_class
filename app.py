import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# โหลดโมเดล
@st.cache_resource  # Cache โมเดลเพื่อโหลดเพียงครั้งเดียว
def load_model():
    model = tf.keras.models.load_model('flowers_model.h5')
    return model

model = load_model()

# คลาสของดอกไม้
class_names = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize ภาพให้ตรงกับที่โมเดลต้องการ
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # เพิ่ม batch dimension
    return image

# ส่วนของ Streamlit UI
st.title("Flower Classification App 🌸")
st.write("อัปโหลดภาพดอกไม้เพื่อให้โมเดลทำการทำนายชนิดของดอกไม้")
st.write("### ชนิดของดอกไม้ที่โมเดลรองรับ:")
st.markdown(
    """
    - 🌼 **Daisy**  
    - 🌻 **Dandelion**  
    - 🌹 **Roses**  
    - 🌞 **Sunflowers**  
    - 🌷 **Tulips**
    """
)

# อัปโหลดภาพ
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ปุ่มทำนาย
    if st.button("Predict"):
        # ประมวลผลภาพ
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        # แสดงผลลัพธ์
        st.write(f"### ผลการทำนาย:")
        st.write(f"Predicted Class: **{class_names[predicted_class]}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
