# 💰 Salary Prediction using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

A **Deep Learning based Salary Prediction System** built using **Artificial Neural Networks (ANN)** and deployed with **Streamlit**.

The model predicts the **salary of an employee** based on:

- 🎓 Education Level
- 💼 Job Title
- 📅 Years of Experience

---

# 🚀 Project Overview

This project demonstrates a **complete Machine Learning pipeline**.

### Steps in the Project

1. Data Collection
2. Data Preprocessing
3. Feature Encoding
4. Feature Scaling
5. ANN Model Training
6. Model Evaluation
7. Web Application Deployment

### Workflow
User Input
↓
Data Preprocessing
↓
Feature Scaling
↓
ANN Model
↓
Salary Prediction



---

# 📊 Dataset

The dataset contains employee salary information.

| Feature | Description |
|------|------|
| Education Level | Bachelor's, Master's, PhD |
| Job Title | Employee role |
| Years of Experience | Experience in years |
| Salary | Target variable |

### Target Variable
salary


---

# 🧠 Model Architecture

The model uses a **Feedforward Artificial Neural Network (ANN)**.

Input Layer
↓
Dense Layer (64 neurons, ReLU)
↓
Dense Layer (32 neurons, ReLU)
↓
Output Layer (1 neuron)

### Model Implementation

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

⚙️ Data Preprocessing
1️⃣ One Hot Encoding

Categorical features such as Education Level and Job Title were converted into numerical format.

Example:

Education Level = Bachelor's

becomes

Education_Bachelor = 1
Education_Master = 0
Education_PhD = 0
