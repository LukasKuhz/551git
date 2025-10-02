# ECSE 551 - Assignment 1: Machine Learning for Engineers

**Course:** ECSE 551 – Machine Learning for Engineers – Fall 2025  
**Due Date:** October 10, 2025 at 11:59pm (EST, Montreal Time)  
**Group:** Lukas Kuhzarani  
**Kernel:** ecse551-a1  
**Random State:** 42

## Overview

This repository contains Assignment 1 for ECSE 551, which focuses on implementing machine learning algorithms from scratch and comparing them with scikit-learn implementations. The assignment covers both classification and regression tasks using two datasets: the Iris dataset for classification and the California Housing dataset for regression.

## Assignment Requirements

### Task 1: Data Analysis and Preprocessing
- **Classification Dataset:** Iris dataset (150 flowers, 4 features: sepal length/width, petal length/width, 3 species)
- **Regression Dataset:** California Housing dataset (housing prices with various features)

**Required Analysis:**
- Load datasets into NumPy/Pandas objects
- Clean data (check for missing values, outliers, erroneous values)
- Statistical analysis (means, min/max, distributions)
- Data visualization (box plots, histograms, scatter plots)
- Feature correlation analysis
- Determine if scaling is beneficial

### Task 2: Model Implementation (From Scratch)

**Regression Models (using only NumPy):**
1. **Ridge Regression** - with custom class implementation
2. **Lasso Regression** - with soft thresholding
3. Compare with sklearn's LinearRegression

**Classification Models (using only NumPy):**
1. **Multiclass SVM** - one-vs-rest approach
2. **Multilayer Perceptron (MLP)** - with ReLU activation and softmax output
3. Compare with sklearn's RandomForestClassifier

**Requirements:**
- Train/test split: 70%/30% with random_state=42
- Custom Python classes following specified structure
- Parameter analysis and performance evaluation
- Decision boundary plotting for classification models
- MSE for regression, accuracy for classification

## Files

- `A1.ipynb` - Main assignment notebook (currently contains exploratory data analysis)
- `ECSE_551_Assignment_1.pdf` - Original assignment instructions
- `assignment_instructions.txt` - Extracted text from PDF
- `extract_pdf.py` - Python script used to extract PDF content

## Current Progress

The notebook currently contains:
- ✅ Data loading and preprocessing for Iris dataset
- ✅ Exploratory data analysis (statistics, correlations)
- ✅ Data visualization (histograms, box plots, scatter plots)
- ✅ Train-test splitting and standardization
- ❌ **Still needed:** Complete implementation of all 6 models
- ❌ **Still needed:** Model comparison and analysis
- ❌ **Still needed:** California Housing dataset integration

## Setup

### Required Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Additional Libraries Used
- NumPy (for custom model implementations)
- Pandas (for data manipulation)
- Matplotlib (for visualization)
- Scikit-learn (for data preprocessing and comparison models)

## Usage

1. **Run the current analysis:**
   ```bash
   jupyter notebook A1.ipynb
   ```

2. **Complete the remaining tasks:**
   - Implement Ridge and Lasso regression from scratch
   - Implement Multiclass SVM from scratch
   - Implement MLP from scratch
   - Add California Housing dataset analysis
   - Compare all models and analyze results

## Model Implementation Structure

### Ridge Regression
```python
class RidgeRegression():
    def __init__(self, learning_rate, iterations, penalty):
    def fit(self, X, Y):
    def update_weights(self):
    def predict(self, X):
```

### Lasso Regression
```python
class LassoRegression():
    def __init__(self, learning_rate, iterations, penalty):
    def fit(self, X, Y):
    def update_weights(self):
    def predict(self, X):
```

### Multiclass SVM
```python
class MulticlassSVM():
    def __init__(self, learning_rate, lambda_param, n_iters):
    def fit(self, X, y):
    def predict(self, X):
```

### Multilayer Perceptron
```python
class MLP():
    def __init__(self, input_size, hidden_size, output_size, lr):
    def relu(self, Z):
    def relu_derivative(self, Z):
    def softmax(self, Z):
    def fit(self, X, y, epochs):
    def predict(self, X):
```

## Deliverables

1. **Jupyter Notebook** - Complete implementation with all results
2. **Report (PDF)** - Maximum 5 pages using IEEE template including:
   - Abstract (100-150 words)
   - Introduction
   - Methodology
   - Datasets description
   - Results (longest section)
   - Discussion and Conclusion
   - Statement of contributions

## Evaluation

- **Report and Code:** 60%
  - Completeness: 15/60
  - Correctness: 25/60
  - Writing Quality: 20/60
- **Demo:** 40% (5 minutes per team member)

## Important Notes

- Late submissions: 5% penalty per day
- Group assignment (3 members)
- In-person demo required
- All team members must understand entire solution
- Python only, custom implementations required for 4 models
- Use sklearn for preprocessing and 2 comparison models
