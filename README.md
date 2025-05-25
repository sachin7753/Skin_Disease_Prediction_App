# AI-Powered Skin Disease Detection & E-Consultation App

A web app that uses deep learning to detect skin diseases from images and provides helpful medical guidance.

# Project Summary
--> This app allows users to upload photos of their skin conditions and uses a trained Convolutional Neural Network (CNN) model to predict the type of skin disease. It then provides detailed information about the disease, including causes, care tips, medications, and treatment options.

--> The app also includes secure user authentication with registration and login, keeping sessions active during use.

# Features
--> Upload images (JPG, PNG) for skin disease prediction

--> Deep learning model (CNN) trained on skin disease dataset

--> Detailed disease reports with causes, dos & don’ts, medication, and treatments

--> User registration and login with session management

--> Built using Streamlit for an interactive web interface

# Technologies Used

Streamlit: Web app framework for UI and backend
TensorFlow/Keras: Deep learning model development and prediction
PIL & NumPy: Image processing and array operations
JSON: User data and disease information storage

# How to Run
Clone the repo and install dependencies from requirements.txt
Make sure the model file skin_model.h5 and users.json are in the project folder

Run the app with:
streamlit run app.py  
Register or login, upload a skin image, and get predictions!


Disclaimer
This app is for educational purposes only and not a substitute for professional medical advice.
