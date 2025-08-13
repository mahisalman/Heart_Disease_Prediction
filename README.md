# ❤️ Heart Disease Prediction API

A **FastAPI**-based machine learning application that predicts the likelihood of heart disease based on patient health metrics.  
The model is trained using a Random Forest Classifier on the [Kaggle Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

🚀 **Live API**: [Swagger Docs](https://heart-disease-prediction-joq2.onrender.com/docs#/)

---

## 📌 Features
- **Machine Learning Model**: Random Forest Classifier trained on heart disease dataset.
- **REST API** built with FastAPI.
- **Dockerized** for easy deployment.
- **Live on Render** with automatic documentation.

---

## 📂 Project Structure
```
├── app
│ ├── main.py # FastAPI app entry point
│ ├── schemas.py # Pydantic request/response models
├── model
│ └── heart_model.joblib # Trained ML model
├── heart.csv # Dataset
├── model.py # Model training script
├── requirements.txt # Python dependencies
├── Dockerfile # Docker image definition
├── docker-compose.yml # Optional docker-compose setup
└── README.md # Project documentation
```

---

## 🛠 Installation & Local Development

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction

2️⃣ Create and activate a virtual environment

python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Train the model (if not already trained)

python model.py

5️⃣ Run the API locally

uvicorn app.main:app --reload

API will be available at:
➡ http://127.0.0.1:8000
Swagger Docs: http://127.0.0.1:8000/docs
🐳 Run with Docker

docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api

🌐 Deployed API

Live Endpoint:
Base URL: https://heart-disease-prediction-joq2.onrender.com
API Endpoints:
Method	Endpoint	Description
GET	/health	Check if the API is running
GET	/info	Get model details and feature list
POST	/predict	Predict heart disease based on input features
📄 Example API Request
POST /predict

Request Body:

{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}

Response:

{
  "heart_disease": true
}

📚 Documentation

The interactive API documentation is available here:
👉 Swagger UI
---
---
I included:
- Deployment link
- Installation steps
- Docker instructions
- Example API request/response
- Swagger docs link
Do you want me to also include a **cURL command** example so users can test `/predict` directly from the terminal? That could make the README even more developer-friendly.
