# Skin_Disease_Prediction_App

Overview
A web-based app built with Streamlit and TensorFlow that uses deep learning to detect skin diseases from images and provide users with helpful medical information. It supports user registration and login for personalized access.

Features
Upload skin images for AI-powered disease prediction

Detailed disease report including causes, dos & don'ts, medication, and treatments

Secure user authentication system

User-friendly interface built with Streamlit

Tech Stack
Frontend & Backend: Streamlit

Machine Learning: TensorFlow / Keras (CNN model)

Image Processing: PIL, NumPy

Data Storage: JSON for user management and disease metadata

How to Run
Clone the repo

Install dependencies:
pip install -r requirements.txt  

Run the app:
streamlit run app.py  
Upload skin images, register/login, and start detecting!

Folder Structure
app.py: Main app code
skin_model.h5: Pretrained CNN model
users.json: Stores user credentials
utils/disease_info.py: Disease metadata and info

Contributing
Feel free to open issues or submit pull requests to improve features, add more diseases, or enhance UI.

