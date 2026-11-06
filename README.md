# M.Tech AIML - Machine Learning: Assignment 2

The project demonstrates a real-world machine learning deployment workflow including data preprocessing, feature scaling, model training, evaluation metrics calculation, building an interactive frontend using **Streamlit**, and deploying the app on **Streamlit Community Cloud**.

---

## Submission Links
* **GitHub Repository Link:** [https://github.com/hshariharan/ML-Assignment_2](https://github.com/hshariharan/ML-Assignment_2)[cite: 482, 484]
* **Live Streamlit App Link:** [Insert your live Streamlit app URL here][cite: 482]

---

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy a robust binary classification model capable of predicting whether a breast tissue mass is malignant or benign based on digitized fine needle aspirate (FNA) cell nuclei characteristics. The system evaluates five distinct machine learning algorithms and exposes them through an interactive web interface for real-time testing.

---

## b. Dataset Description
* **Dataset Used:** Breast Cancer Wisconsin (Diagnostic) Dataset (sourced via Scikit-Learn / UCI Machine Learning Repository).
* **Total Instances:** 569 samples[cite: 488] (Split: 455 training samples, 114 test samples)[cite: 488].
* **Feature Size:** 30 continuous input features describing nuclear characteristics (e.g., radius, texture, perimeter, area, smoothness, concavity).
* **Target Variable:** Binary classification (`0` = Malignant, `1` = Benign).

---

## c. Project Structure
```text
project-folder/
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── support_vector_machine.pkl
│   └── scaler.pkl
├── app.py
├── requirements.txt
├── test_data.csv
└── README.md

d. Models Used & Performance Comparison
All five classification models were trained using standard preprocessing protocols (StandardScaler) and evaluated on the held-out test dataset (test_data.csv).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |



