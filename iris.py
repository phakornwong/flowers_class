import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import os

# โหลดข้อมูล Iris เพื่อเตรียม Standardize
iris = load_iris()
scaler = StandardScaler()
scaler.fit(iris.data)
target_names = iris.target_names

# โหลดโมเดลที่เทรนไว้
model = tf.keras.models.load_model('iris_model.keras')

# Map รูปภาพกับพันธุ์ดอกไม้
image_map = {
    "setosa": "setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}

# ส่วนต้อนรับใน Streamlit
st.title("🌸 Iris Flower Classifier with Images")
st.write("กรอกข้อมูลด้านล่างเพื่อพยากรณ์ชนิดของดอกไม้")

# รับค่าจากผู้ใช้งาน
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# พยากรณ์เมื่อกดปุ่ม
if st.button("Predict"):
    # เตรียมข้อมูล
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_scaled = scaler.transform(input_data)  # Standardize ข้อมูล
    
    # ใช้โมเดลพยากรณ์
    predictions = model.predict(input_data_scaled)
    predicted_class = target_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # แสดงผลลัพธ์
    st.subheader("ผลลัพธ์การพยากรณ์:")
    st.write(f"ชนิดของดอกไม้: **{predicted_class.capitalize()}**")
    st.write(f"ความมั่นใจของโมเดล: **{confidence:.2f}%**")

    # แสดงรูปภาพ
    image_path = image_map[predicted_class]
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Predicted: {predicted_class.capitalize()}", use_column_width=True)
    else:
        st.write("ไม่พบรูปภาพของพันธุ์ดอกไม้ที่พยากรณ์ได้")
