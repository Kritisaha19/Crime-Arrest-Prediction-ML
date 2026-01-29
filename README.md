# Crime-Arrest-Prediction-ML

## 📌 Project Overview

This project aims to predict whether a reported crime results in an arrest using real-world crime data. By analyzing factors such as crime type, location, victim details, weapon used, and time of occurrence, the model estimates the probability of an arrest and provides insights into patterns influencing arrest outcomes.

---

## Problem Statement

Crime datasets are large, complex, and often imbalanced. Understanding which factors increase the likelihood of an arrest is challenging using manual analysis alone.
The problem addressed in this project is:

> **Can we use machine learning to predict arrest outcomes and identify key factors influencing arrests?**

---

## Solution Approach

To solve this problem, a **binary classification model** is built using Logistic Regression. The project follows a complete machine learning pipeline:

* Data preprocessing and cleaning
* Feature encoding and scaling
* Handling class imbalance
* Model training and evaluation
* Interpretation and visualization of results

The focus is not only on prediction accuracy but also on **understanding model behavior and insights**.

---

## Tech Stack

* **Programming Language:** Python
* **Libraries & Tools:**

  * Pandas, NumPy (Data handling)
  * Matplotlib, Seaborn (Visualization)
  * Scikit-learn (Machine Learning)
* **Model Used:** Logistic Regression

---

## Project Workflow

1. **Data Loading:** Import real-world crime dataset
2. **Feature Selection:** Choose relevant crime, victim, location, and time features
3. **Target Creation:** Convert arrest status into a binary outcome
4. **Data Preprocessing:**

   * Handle missing values
   * Encode categorical variables
   * Scale numerical features
5. **Train-Test Split:** Maintain class distribution using stratification
6. **Model Training:** Logistic Regression with class imbalance handling
7. **Evaluation:**

   * Accuracy, Precision, Recall, F1-score
   * Confusion Matrix
   * ROC–AUC Curve
8. **Analysis & Visualization:**

   * Feature importance
   * Time-based arrest trends
   * Threshold analysis
   * Error and confidence analysis

---

## Key Insights

* Certain crime types and locations show higher arrest probabilities
* Time of occurrence influences arrest likelihood
* Class imbalance significantly affects prediction behavior
* Model confidence correlates with prediction correctness

---

## Limitations & Ethics

* Label encoding may introduce unintended ordinal relationships
* Crime data may contain social or reporting biases
* The model is intended for **analytical insights only**, not real-world decision-making

---

## How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Crime-Arrest-Prediction-ML.git
cd Crime-Arrest-Prediction-ML
```

### 2️⃣ Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3️⃣ Add the dataset

Place the crime dataset CSV file in the project directory and update the file path in the code if needed.

### 4️⃣ Run the script

```bash
python crime_arrest_prediction.py
```

The script will train the model and generate evaluation metrics and visualizations.

---

## Future Improvements

* Use one-hot or target encoding for categorical features
* Experiment with advanced models (Random Forest, XGBoost)
* Add cross-validation and hyperparameter tuning
* Build a dashboard for interactive analysis

---

## 📫 Contact

Feel free to explore the repository and reach out for discussions, feedback, or collaboration!

---
