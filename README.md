# Why Linear Models Fail: A Study of Non-Linearity in Heart Disease Prediction

## Project Objective

The objective of this project is to investigate the limitations of linear machine learning models for heart disease prediction and evaluate how non-linear methods can improve classification performance.

Using the UCI Heart Disease Dataset, several machine learning algorithms were implemented and compared to answer the following questions:

* Why do linear models fail on complex medical datasets?
* How does Support Vector Machine improve decision boundaries?
* Why does boosting achieve better predictive performance?
* How do non-linear methods compare to traditional linear approaches?

The project was developed as part of the **Non-Linear Machine Learning Systems** roadmap, focusing on:

* Support Vector Machines (SVM)
* Kernel Methods
* Gradient Boosting
* XGBoost

---

## Implemented Models

### 1. Logistic Regression (From Scratch)

A linear classification model trained using gradient descent and Binary Cross Entropy loss.

**Purpose:** Establish a baseline linear classifier.

---

### 2. k-Nearest Neighbors (KNN) (From Scratch)

A distance-based classifier that predicts labels using the majority class among the nearest neighbors.

**Purpose:** Introduce local non-linear decision making.

---

### 3. Polynomial Kernel SVM

A Support Vector Machine using a polynomial kernel to transform the feature space and learn non-linear decision boundaries.

**Purpose:** Study the effect of feature transformation and margin maximization.

---

### 4. XGBoost (From Scratch Inspired Implementation)

A simplified implementation of XGBoost based on gradient boosting principles and Binary Cross Entropy optimization.

**Purpose:** Investigate how ensemble learning and boosting improve predictive performance.

---

## Dataset

**Dataset:** UCI Heart Disease Dataset

The dataset contains clinical attributes related to cardiovascular health, including:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression
* Other diagnostic indicators

The target variable represents the presence or absence of heart disease.

---

## Project Structure

```text
project/
│
├── data/
│   └── disease_prediction.csv
│
├── notebooks/
│   ├── exploration.ipynb
│   └── experiments.ipynb
│
├── src/
│   ├── classicalmodels.py
│   ├── metrics.py
│   ├── preprocessing.py
│   └── utils.py
    ├──  report.pdf

│
├── figures/
│   ├── target_distribution.png
│   ├── feature_analysis.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├
│
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### 3. Run Dataset Exploration

```bash
jupyter notebook exploration.ipynb
```

This notebook performs:

* Data loading
* Data visualization
* Correlation analysis
* Feature exploration

### 4. Run Experiments

```bash
jupyter notebook experiments.ipynb
```

This notebook:

* Trains all models
* Evaluates model performance
* Generates comparison plots
* Produces confusion matrices and performance metrics

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Results Summary

The experimental results show that Logistic Regression provides a strong baseline and achieves competitive performance on the Heart Disease dataset. This suggests that a substantial portion of the relationship between the clinical features and the target variable can be captured using a linear decision boundary.

However, non-linear models such as Polynomial SVM and XGBoost achieved slightly higher performance. These improvements indicate the presence of additional non-linear feature interactions that are not fully captured by a purely linear model.

Key observations include:

* Logistic Regression provides a strong baseline but assumes a linear decision boundary.
* KNN captures local patterns and improves performance in regions where classes overlap.
* Polynomial Kernel SVM successfully models non-linear relationships through feature transformation.
* XGBoost achieves the strongest performance by combining multiple decision trees through gradient boosting.

These findings support the hypothesis that heart disease prediction involves complex non-linear relationships that cannot be fully captured by a single linear model.

---

## Main Conclusion

This study experimentally demonstrates that:

* Linear models provide a strong and interpretable baseline for heart disease
prediction. Nevertheless, non-linear approaches such as Polynomial SVM and
XGBoost are capable of capturing additional feature interactions and may
achieve improved predictive performance.
* Kernel methods improve classification by transforming the feature space.
* Ensemble learning and boosting significantly enhance predictive performance.
* Non-linear machine learning methods provide a more flexible framework for modeling cardiovascular risk factors.

---

## Authors

Heart Disease Prediction Project – Machine Learning Systems Study

Developed for the study of non-linear machine learning methods and their application to medical diagnosis.
