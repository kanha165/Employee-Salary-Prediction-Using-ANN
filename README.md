# 💰 Salary Prediction using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

A **Deep Learning-based Salary Prediction System** built using **Artificial Neural Networks (ANN)** and deployed with **Streamlit**.

The model predicts the **salary of an employee** based on:

* 🎓 Education Level
* 💼 Job Title
* 📅 Years of Experience

---

# 🚀 Project Overview

This project demonstrates a **complete end-to-end Machine Learning pipeline**, from data preprocessing to deployment.

### 🔄 Workflow

```
User Input
   ↓
Data Preprocessing
   ↓
Feature Encoding & Scaling
   ↓
ANN Model
   ↓
Salary Prediction
```

---

# 📊 Dataset

The dataset contains employee salary-related information.

| Feature             | Description               |
| ------------------- | ------------------------- |
| Education Level     | Bachelor's, Master's, PhD |
| Job Title           | Employee role             |
| Years of Experience | Experience in years       |
| Salary              | Target variable           |

### 🎯 Target Variable

* `Salary`

---

# ⚙️ Data Preprocessing

### 1️⃣ Handling Categorical Data (One Hot Encoding)

Categorical features such as **Education Level** and **Job Title** are converted into numerical format.

#### Example:

```
Education Level = Bachelor's
```

⬇️ Converted to:

```
Education_Bachelor = 1  
Education_Master = 0  
Education_PhD = 0  
```

---

### 2️⃣ Feature Scaling

* Applied **StandardScaler / MinMaxScaler**
* Ensures all features are on the same scale
* Improves model performance and convergence

---

# 🧠 Model Architecture

The model uses a **Feedforward Artificial Neural Network (ANN)**.

```
Input Layer
   ↓
Dense Layer (64 neurons, ReLU)
   ↓
Dense Layer (32 neurons, ReLU)
   ↓
Output Layer (1 neuron)
```

---

# 🧪 Model Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

---

# 📈 Model Evaluation

The model is evaluated using:

* 📉 **Mean Squared Error (MSE)** → Measures prediction error
* 📊 **Mean Absolute Error (MAE)** → Average absolute difference

---

# 🌐 Web Application (Streamlit)

A user-friendly **Streamlit web app** is built for real-time predictions.

### Features:

* Interactive UI
* User input form
* Instant salary prediction
* Lightweight deployment

---

# ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/salary-prediction-ann.git
cd salary-prediction-ann
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```
├── app.py
├── model.pkl / model.h5
├── scaler.pkl
├── encoder.pkl
├── dataset.csv
├── requirements.txt
└── README.md
```

---

# 🔥 Key Highlights

* ✅ End-to-End ML Pipeline
* ✅ Deep Learning (ANN) Implementation
* ✅ Feature Engineering & Scaling
* ✅ Streamlit Deployment
* ✅ Real-world Use Case

---

# 🚧 Future Improvements

* 🔹 Add more features (Location, Skills, Company Size)
* 🔹 Use advanced models (XGBoost, Random Forest)
* 🔹 Improve accuracy with hyperparameter tuning
* 🔹 Deploy using Docker / Cloud

---

# 📌 Use Cases

* HR Salary Estimation
* Job Market Analysis
* Career Guidance Tools

---

# 👨‍💻 Author

**Kanha Patidar**

* 💼 LinkedIn: (https://www.linkedin.com/in/kanha-patidar-837421290/)
* 💻 GitHub: (https://github.com/kanha165)

---

# 📜 License

This project is licensed under the **MIT License**.

---

# ⭐ Support

If you like this project:

* ⭐ Star this repository
* 🔁 Share it on LinkedIn
* 💬 Give feedback

---
