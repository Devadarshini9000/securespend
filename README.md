# 🔐 SecureSpend – Credit Card Fraud Detection System

**SecureSpend** is a machine learning-based web app that detects fraudulent credit card transactions using anonymized transaction features and advanced resampling techniques. It provides real-time fraud predictions through an interactive Streamlit dashboard.

---

## 🚀 Features

- 🧠 Uses **Random Forest Classifier** for fraud detection
- 🧮 Incorporates **SMOTE** to handle class imbalance
- 📊 Scales transaction amounts with `StandardScaler`
- 🧪 Accepts PCA-transformed features (V1–V28)
- 🖥️ Built with an intuitive **Streamlit interface**

---

## 📁 Project Structure
├── credit_train.py # Trains the fraud detection model
├── credit_test.py # Streamlit app for real-time prediction
├── credit_preprocess.py # Cleans data, applies scaling + SMOTE
├── fraud_model.pkl # Trained fraud detection model
├── cscaler.pkl # Scaler for 'Amount' normalization
├── preprocessed_fraud.csv # Resampled and preprocessed dataset

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- imbalanced-learn (SMOTE)
- Streamlit
- Joblib

---

## 🧪 How It Works

1. **Preprocessing (`credit_preprocess.py`)**
   - Loads dataset and drops missing values
   - Scales the transaction `Amount` field
   - Uses **SMOTE** to address class imbalance
   - Saves preprocessed dataset as `preprocessed_fraud.csv`

2. **Model Training (`credit_train.py`)**
   - Splits dataset into training and testing sets
   - Trains a **Random Forest model** (or Logistic Regression)
   - Evaluates accuracy and classification metrics
   - Saves model to `fraud_model.pkl`

3. **Fraud Prediction UI (`credit_test.py`)**
   - Accepts `Amount` and PCA features (V1 to V28) from user
   - Scales and formats inputs
   - Predicts whether a transaction is **genuine or fraudulent**

---

## 💻 How to Run the App

### 🔹 1. Clone the Repo

```bash
git clone https://github.com/yourusername/securespend.git
cd securespend

###🔹 2. Install Requirements
bash
pip install pandas numpy scikit-learn streamlit joblib imbalanced-learn

### 🔹 3. Train the Model
bash
python credit_train.py
(This will automatically preprocess data and generate the model)

###🔹 4. Launch the Streamlit App
bash
streamlit run credit_test.py

### 🎯 Use Cases
🛡️ Real-time fraud screening for online payments

💳 Bank/Fintech transaction monitoring

📊 Data science portfolio showcasing classification & imbalance handling

## 👩‍💻 Author
**Devadarshini P**  
[🔗 LinkedIn](https://www.linkedin.com/in/devadarshini-p-707b15202/)  
[💻 GitHub](https://github.com/Devadarshini9000)

“Catch the fraud before it catches you.” – SecureSpend
