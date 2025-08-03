# COVID-19 Outcome Prediction using Supervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-1.4.2-blueviolet.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the code and resources for the project "COVID-19 Prediction Using Supervised Machine Learning," developed as part of the Machine Learning course (CCS2113) at Albukhary International University. The project focuses on leveraging supervised machine learning algorithms to predict COVID-19 infection based on patient symptoms and exposure data.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [How to Run](#-how-to-run)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Author](#-author)
- [License](#-license)

## 🎯 Project Overview

The primary goal of this study is to develop a robust machine learning model capable of accurately predicting whether a person is infected with COVID-19. The prediction is based on a set of 20 binary features that represent common symptoms, pre-existing health conditions, and exposure history. This project demonstrates a complete machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and comparison.

## 📊 Dataset

The model was trained and evaluated on the **"COVID-19 Symptoms and Presence Dataset"** sourced from Kaggle.

- **Total Samples:** 5,434
- **Features:** 20 binary attributes (encoded as 1 for "Yes" and 0 for "No").
- **Target Variable:** `COVID-19` (1 for Positive, 0 for Negative).

### Features Summary
The features include a mix of symptoms, health history, and exposure factors:
- **Symptoms:** Fever, Dry Cough, Sore Throat, Breathing Problem, Headache, Fatigue, Running Nose, Gastrointestinal Issues.
- **Health History:** Asthma, Chronic Lung Disease, Heart Disease, Diabetes, Hypertension.
- **Exposure Factors:** Abroad Travel, Contact with COVID Patient, Attended Large Gathering, Visited Public Exposed Places, Family working in Public Exposed Places.
- **Precautionary Measures:** Wearing Masks, Sanitization from Market.

### Class Distribution
The original dataset exhibited a significant class imbalance, which was addressed during preprocessing:
- **Positive (1):** 4,250 samples (78.23%)
- **Negative (0):** 1,184 samples (21.77%)

## ⚙️ Methodology

The project followed a structured machine learning pipeline:

1.  **Data Preprocessing:**
    * **Label Encoding:** Converted categorical "Yes"/"No" values into binary 1/0.
    * **Train-Test Split:** The data was split into 80% for training and 20% for testing, using stratified sampling to maintain class proportions.
    * **Feature Scaling:** `StandardScaler` was applied to standardize the feature values, ensuring that all features contribute equally to model performance.

2.  **Handling Class Imbalance:**
    * **SMOTE (Synthetic Minority Oversampling Technique):** Applied to the training set to synthetically generate new samples for the minority class (Negative cases), creating a balanced dataset for model training.

3.  **Feature Selection:**
    * Several techniques were evaluated to identify the most impactful features and reduce dimensionality:
        * Principal Component Analysis (PCA)
        * Linear Discriminant Analysis (LDA)
        * SelectKBest (ANOVA F-test)
        * Recursive Feature Elimination (RFE)
    * **PCA** was found to yield the best-performing models.

4.  **Model Selection and Training:**
    * Two powerful classifiers were chosen for this task:
        * **Support Vector Machine (SVM):** Effective for high-dimensional spaces and non-linear problems.
        * **K-Nearest Neighbors (KNN):** A simple yet powerful instance-based learning algorithm.
    * Hyperparameter tuning was performed using `GridSearchCV` to find the optimal parameters for each model.

5.  **Evaluation:**
    * Models were evaluated on the unseen test set using a 10-fold cross-validation strategy.
    * The primary evaluation metrics were **Accuracy, Precision, Recall, and F1-Score**, with a focus on the F1-Score due to its robustness in imbalanced scenarios.

## 📈 Results

After comprehensive training and tuning, the models achieved high predictive accuracy. The **K-Nearest Neighbors (KNN)** classifier, when combined with **PCA** for feature selection, emerged as the top-performing model.

### Best Model Performance: KNN (with PCA)
| Metric    | Score   |
| :-------- | :------ |
| **Accuracy** | **98.25%** |
| **Precision** | **98.36%** |
| **Recall** | **98.25%** |
| **F1-Score** | **98.28%** |

These results indicate that the model is highly effective at correctly identifying both positive and negative COVID-19 cases from symptom data alone.

## 💡 Key Insights

- Symptom-based prediction of COVID-19 using machine learning is a highly viable and effective approach for early screening.
- Proper data preprocessing, especially handling class imbalance with techniques like **SMOTE**, is crucial for building a reliable and unbiased model.
- Feature selection using **PCA** proved to be the most effective method for this dataset, improving model performance and efficiency.
- Both **SVM** and **KNN** are strong classifiers for this type of medical diagnostic task, with KNN showing a slight edge after tuning.

## 🚀 How to Run

To replicate this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zx784/COVID-19-Outcome-Prediction.git](https://github.com/zx784/COVID-19-Outcome-Prediction.git)
    cd COVID-19-Outcome-Prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is included for easy setup.
    ```bash
    pip install -r requirements.txt
    ```
    The key dependencies are: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `imbalanced-learn`.

4.  **Run the analysis:**
    Open and run the Jupyter Notebook (`.ipynb`) file to see the complete analysis, from data loading to model evaluation.
    ```bash
    jupyter notebook COVID-19_Prediction.ipynb
    ```

## ⚠️ Limitations and Future Work

- **Dataset:** The dataset is synthetic and may not perfectly reflect the complexities of real-world clinical data.
- **Future Work:**
    - Validate the model on a real-world, non-synthetic patient dataset.
    - Explore deep learning models (e.g., neural networks) to potentially capture more complex patterns.
    - Develop and deploy the final model as a web or mobile application for accessible public use.

## ✍️ Author

- **Amro Khaled Mohammed Hasan Shiek**
- Student ID: AIU22102346
- Albukhary International University

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
