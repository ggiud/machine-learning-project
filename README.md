# 🪐 Saturn Dataset Classification – Machine Learning Project

## **Overview**

This project focuses on a binary classification problem using the **Saturn Dataset**, developed for the Machine Learning course by Professor Antonio Candelieri.

The objective is to compare different machine learning algorithms and identify the most effective model for classifying unseen observations.

The analysis includes:

* Exploratory Data Analysis (EDA)
* K-Nearest Neighbours (KNN)
* Random Forest (RF)
* Support Vector Machines (SVM)
* Bayesian Optimization for hyperparameter tuning
* Cross-validation and performance evaluation

---

## **Dataset**

The dataset contains:

* **540 observations**
* **8 numerical features**
* **1 binary class label** with values in {-1, 1}

### **Exploratory Findings**

Descriptive analysis highlighted that:

* feature 8 shows the highest variability
* the dataset is approximately balanced:

  * 275 observations with label -1
  * 265 observations with label +1
* correlations between features are generally low

The strongest correlations were observed between:

* feature 1 and feature 7
* feature 1 and feature 3

These characteristics suggest a relatively clean and well-structured classification problem.

---

## **Methods**

Three different machine learning approaches were tested and compared.

### **K-Nearest Neighbours (KNN)**

KNN is a non-parametric classification method that assigns a new observation to the class most represented among its k nearest neighbours.

Advantages:

* simplicity
* interpretability
* flexibility
* no strong distributional assumptions

The main challenge is selecting:

* the number of neighbours (k)
* the distance metric

---

### **Random Forest (RF)**

Random Forest is an ensemble learning method based on multiple decision trees built on bootstrapped samples.

The algorithm:

* reduces variance through randomization
* improves generalization
* handles noisy and high-dimensional data effectively

Main hyperparameters:

* number of trees
* number of variables considered at each split

---

### **Support Vector Machine (SVM)**

Soft Margin SVM aims to separate the data through a hyperplane maximizing the margin between classes while allowing controlled classification errors.

Since the Saturn dataset is not linearly separable, a **Radial Basis Function (RBF) kernel** was adopted.

The model requires tuning:

* regularization parameter **C**
* kernel parameter **γ (gamma)**

Hyperparameter estimation was performed using **Bayesian Optimization**.

---

## **Model Training & Validation**

The dataset was divided into:

* **90% training set**
* **10% test set**

A **10-Fold Cross Validation** procedure was applied on the training portion for model selection and hyperparameter tuning.

---

## **Results**

### **KNN Results**

Different values of k between 10 and 40 were evaluated using cross-validation.

### **Best configuration**

* **k = 20**

### **Performance**

* Test Accuracy ≈ **98.15%**

The selected value achieved:

* low empirical error
* low generalization error
* stable performance across folds

---

### **Random Forest Results**

The Random Forest model was trained using:

* **500 trees**
* **2 variables per split**

### **Key Findings**

* Out-of-Bag (OOB) error converged after approximately 100 trees
* classification error was close to zero
* the model showed excellent stability and generalization ability

---

### **SVM Results**

A linear SVM initially failed to separate the data effectively.

Consequently, an **RBF kernel** was introduced.

### **Bayesian Optimization**

Bayesian Optimization was used to estimate the optimal hyperparameters more efficiently than Grid Search.

### **Selected hyperparameters**

* **C = 31.45**
* **γ = 0.042**

### **Performance**

* Test Accuracy = **100%**

Bayesian Optimization reached the maximum accuracy significantly faster than Grid Search while exploring the same hyperparameter space.

---

## **Support Vectors Analysis**

The final SVM model used:

* **12.14%** of the observations as Support Vectors

This indicates:

* a clear decision boundary
* low risk of overfitting
* strong generalization capability

The resulting model achieved an effective balance between flexibility and robustness.

---

## **Model Comparison**

### **KNN**

* highly accurate
* simple and interpretable
* more sensitive to local data structure
* computationally less efficient

### **Random Forest**

* excellent predictive performance
* robust to complex structures
* unnecessary complexity for this relatively low-dimensional dataset

### **SVM**

* best overall performance
* perfect classification accuracy
* strong generalization properties
* flexible nonlinear decision boundaries

---

## **Conclusion**

All tested methods achieved excellent classification performance on the Saturn dataset.

However, the **SVM with RBF kernel** emerged as the most effective approach due to:

* perfect predictive accuracy
* robustness to nonlinear patterns
* strong generalization ability
* computational efficiency after Bayesian Optimization

The results suggest that the Saturn dataset contains well-separated nonlinear classes that can be effectively modeled through kernel-based methods.

---

## **Methods & Techniques**

* Exploratory Data Analysis (EDA)
* Cross Validation
* K-Nearest Neighbours (KNN)
* Random Forest
* Support Vector Machines (SVM)
* Radial Basis Function (RBF) Kernel
* Bayesian Optimization
* Hyperparameter Tuning

---

## **Tech Stack**

* **R**
* **caret**
* **e1071**
* **randomForest**
* **ggplot2**
* **rBayesianOptimization**

