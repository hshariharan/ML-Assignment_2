# M.Tech AIML - Machine Learning: Assignment 2

The project demonstrates a real-world machine learning deployment workflow including data preprocessing, feature scaling, model training, evaluation metrics calculation, building an interactive frontend using **Streamlit**, and deploying the app on **Streamlit Community Cloud**.

---

## Submission Links
* **GitHub Repository Link:** [https://github.com/hshariharan/ML-Assignment_2](https://github.com/hshariharan/ML-Assignment_2)
* **Live Streamlit App Link:** https://ml-assignment2-bkspvpaeiadmdpejebevu5.streamlit.app

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
ML-Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── support_vector_machine.pkl
│   └── scaler.pkl
│
└── train_model.py


```

## d. Models Used & Performance Comparison
All five classification models were trained using standard preprocessing protocols (StandardScaler) and evaluated on the held-out test dataset (`test_data.csv`).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |

---

 ## Model Performance Observations & Winner

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved exceptional performance (98.25% accuracy and 0.9954 AUC). The linear decision boundary fits the scaled biomedical feature space very cleanly with minimal false positives or false negatives. |
| Decision Tree | Exhibited lower accuracy (91.23%) relative to linear models due to its inherent susceptibility to variance and partitioning boundaries on continuous numerical data without pruning constraints. |
| KNN | Performed strongly with 95.61% accuracy. Feature scaling (`StandardScaler`) played a vital role in ensuring Euclidean distance computations treated all 30 features with equal relative weight. |
| Naive Bayes | Achieved 92.98% accuracy. While its conditional independence assumption is heavily violated by correlated medical features, it still provided a reliable baseline and high AUC (0.9868). |
| Support Vector Machine | Matched Logistic Regression with 98.25% accuracy and 0.9623 MCC. The Radial Basis Function (RBF) kernel successfully mapped complex non-linear interactions while maintaining a maximum margin separator. |
| **Overall, Winner for your dataset?** | **Logistic Regression & Support Vector Machine (Tie)**. Both models achieved the highest accuracy, F1-score, and Matthews Correlation Coefficient (MCC), proving most optimal for medical diagnosis classification tasks where false negatives must be tightly controlled. |
```

