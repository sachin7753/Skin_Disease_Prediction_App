import streamlit as st
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from utils.disease_info import disease_info

st.set_page_config(page_title="Skin E-Consultant", layout="centered")

USERS_FILE = "users.json"

def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
        with open(USERS_FILE, "w") as f:
            json.dump(users, f)
    return users

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

model = tf.keras.models.load_model("skin_model.h5")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    class_idx = np.argmax(prediction)
    class_name = list(disease_info.keys())[class_idx]
    confidence = np.max(prediction)
    return class_name, confidence

def disease_report(predicted_class, confidence):
    info = disease_info[predicted_class]
    st.markdown(f"### 🩺 Prediction Result: {predicted_class}")
    st.markdown(f"### 🔎 Confidence: {confidence:.2%}")
    if confidence < 0.6:
        st.warning("⚠️ The model is not confident in this prediction. Please consult a dermatologist.")

    st.subheader("🧠 Cause")
    st.write(info['Cause'])

    st.subheader("✅ Dos")
    for item in info['Dos']:
        st.markdown(f"- {item}")

    st.subheader("⛔ Don'ts")
    for item in info['Donts']:
        st.markdown(f"- {item}")

    st.subheader("💊 Medication")
    st.write(info['Medication'])

    st.subheader("🧬 Cure / Treatment")
    st.write(info['Cure'])

def main_app():
    st.markdown('<h1 style="white-space: nowrap;">📷 Skin Disease Detection & Consultation</h1>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        predicted_class, confidence = predict(img)
        st.markdown("---")
        disease_report(predicted_class, confidence)

def login(users):
    st.markdown('<h1 style="white-space: nowrap;">🔐 Login - Skin Disease Prediction App</h1>', unsafe_allow_html=True)
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    login_button = st.button("Login")

    if login_button:
        if username in users and users[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("✅ Login successful! Please continue.")
        else:
            st.error("❌ Invalid username or password.")

def register(users):
    st.markdown('<h1 style="white-space: nowrap;">📝 Register - Skin Disease Prediction App</h1>', unsafe_allow_html=True)
    new_username = st.text_input("New Username", key="reg_user")
    new_password = st.text_input("New Password", type="password", key="reg_pass")
    register_button = st.button("Register")

    if register_button:
        if new_username.strip() == "" or new_password.strip() == "":
            st.error("⚠️ Username and password cannot be empty.")
        elif new_username in users:
            st.error("⚠️ Username already exists. Please choose another.")
        else:
            users[new_username] = new_password
            save_users(users)
            st.success("✅ Registration successful! You can now login.")

def app():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    users = load_users()

    if st.session_state.authenticated:
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
        main_app()
    else:
        tab = st.sidebar.radio("Go to", ["Login", "Register"])
        if tab == "Login":
            login(users)
        else:
            register(users)

if __name__ == "__main__":
    app()
