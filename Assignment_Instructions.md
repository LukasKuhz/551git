# Group Assignment 1

**ECSE 551 – Machine Learning for Engineers – Fall 2025**  
**September 16, 2025**

## 1 General instructions

• The assignment is due on Oct. 10 at 11:59pm (EST, Montreal Time).  
• Late submissions will be penalized 5 percent per day.  
• This assignment is to be completed in groups of three. All members of a group will receive the same grade for the submitted portion of the work except when a group member is not responding or contributing to the assignment. If this is the case and there are major conflicts, please reach out to the contact TA or instructor for help (well before the deadline) and flag this in the submitted report.  
• There will be an in-person demo. For this, students will receive both a group mark and an individual mark that depends on how well they answer the questions posed to them.  
• Each team member should familiarize themselves thoroughly with all aspects of the submitted solution. The demo questions will NOT be tailored to the specific contributions of each team member.  
• Submit your assignment on MyCourses as a group. Any group member can submit, but there is only one submission per group. Submit your assignment as a single Jupyter Note- book (.ipynb) named with the last names of all group members in alphabetical order (e.g., RoySmith Tremblay Assignment1.ipynb).  
• Python should be used for this and all assignments. You are free to use libraries with general utilities, such as matplotlib, numpy and scipy for Python, unless stated otherwise in the description of the task. In this assignment, you should implement 4 models yourself from scratch (no scikitlearn or other package implementations); the other 2 models you can call from scikitlearn.

## 2 Overview

In this assignment, you will implement (from scratch) two classification techniques and two regression techniques. You will also experiment with an additional classification method and an additional regression method using scikitlearn implementations.

## 3 Task 1

The first step is to acquire the data and conduct an analysis to try to achieve a greater understanding. If necessary you should also clean the data, checking for (and possibly removing) outliers, and handling missing values.

The two datasets we will use are:

• **Classification:** Use the Iris dataset. The dataset contains information about 150 Iris flowers, including the following features: sepal length, sepal width, petal length, and petal width. The target variable is the species of the flower (Iris setosa, Iris versicolor, or Iris virginica). You can load the dataset from sklearn.datasets using `from sklearn.datasets import load_iris`.

• **Regression:** Use the California Housing dataset. The dataset contains information about California districts, including median house value (target variable), median income, housing age, number of rooms, and geographical coordinates.

We fetch the dataset using the following Python script:

```python
import os
import tarfile
import urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
```

You should:

**a.** Load the datasets into NumPy or Pandas objects.

**b.** Determine whether the data needs to be cleaned, and if so, clean it. Check for missing features. Check for erroneous values and missing values. If there are missing values, you can adopt the simplest approach and delete them. Alternatively you can explore other approaches such as imputation (e.g., using the average of the observed values).

**c.** Conduct a statistical analysis of the data. Explore the means of the features and the max and min values. Consider whether this might imply that performing some scaling could be beneficial. Use box plots, histograms, and scatter plots to explore the distributions of the data. For example, is one of the features concentrated around one mode? Two modes? Does the empirical distribution exhibit heavy tails or is it closer to Gaussian? Are some features discrete/categorical? Are some of the features correlated?

## 4 Task 2

In this task, you are required to implement models from scratch for both classification and regression using only numpy. You may use other libraries for data processing and preprocessing, but the model implementations themselves must rely solely on numpy. For the data, use a train/test split of 70% / 30% and a random state of 42.

• **For the regression dataset,** implement **Ridge Regression** and **Lasso Regression with soft thresholding** from scratch using only numpy. You should define a Python class (`RidgeRegression()` or `LassoRegression()`) following the structure below:

```python
class RidgeRegression():
    def __init__(self, learning_rate, iterations, penalty):
        # initialize parameters
    def fit(self, X, Y):
        # training loop
    def update_weights(self):
        # weight update rule
    def predict(self, X):
        # prediction rule
```

Analyse how the choice of parameters affects the model's performance. Compute the mean squared error (MSE) to quantify prediction accuracy and examine the learned feature coefficients. Finally, compare the results of your Ridge and Lasso implementations with the performance of `LinearRegression` from `sklearn.linear_model`.

• **a.** For the classification dataset, implement a **Multiclass SVM** from scratch using only numpy. You should define a Python class (`MulticlassSVM()`) following the structure below:

```python
class MulticlassSVM():
    def __init__(self, learning_rate, lambda_param, n_iters):
        # initialize parameters for gradient descent and regularization
        # prepare storage for one-vs-rest classifiers
    def fit(self, X, y):
        # training loop for each class (one-vs-rest)
        # update weights and bias using hinge loss + regularization
    def predict(self, X):
        # compute decision scores for each class
        # assign each sample to the class with highest score
```

Plot the decision boundaries and analyse the results. Analyse how the choice of parameters affects the model's performance.

**b.** Implement a **Multilayer Perceptron (MLP)** from scratch using only numpy. You should define a Python class (`MLP()`) following the structure below:

```python
class MLP():
    def __init__(self, input_size, hidden_size, output_size, lr):
        # initialize weights and biases for input-hidden and hidden-output layers
    def relu(self, Z):
        # ReLU activation function
    def relu_derivative(self, Z):
        # derivative of ReLU for backpropagation
    def softmax(self, Z):
        # softmax function for output layer
    def fit(self, X, y, epochs):
        # forward pass: compute hidden and output layer activations
        # compute loss and backpropagate errors
        # update weights and biases
    def predict(self, X):
        # compute predictions for input X
        # return class labels
```

You can use sklearn to standardize features and one-hot encode the target labels. Evaluate your model on a test set and report the accuracy. Discuss the influence of hyperparameters (hidden layer size, learning rate, number of epochs) on the model performance.

**c.** Finally, evaluate and compare your models' performance with that of a `RandomForestClassifier` from `sklearn`.

## 5 Deliverables

Submit two separate files to myCourses. Please use consistent naming conventions (surnames in alphabetical order).

– A single Jupyter Notebook (.ipynb) named with the last names of all group members in alphabetical order (e.g., Roy Smith Tremblay Assignment1.ipynb). Your notebook should reproduce all of the results included in your report. Please ensure that all original training and output results are saved in your notebook. These should be the same results provided in the report.

– A (max. 5 page) report submitted as a pdf.

### 5.1 Report

Please prepare your document using the templates provided here:  
https://www.ieee.org/conferences/publishing/templates

The report should include the following sections:

– **Abstract:** a very brief (100-150 words) summary of the content of the report.  
– **Introduction:** summarize the assignment tasks, the datasets, and the most important findings.  
– **Methodology:** Describe the algorithmic concepts at a high-level (there is no need for code or pseudo-code)  
– **Datasets:** Very briefly describe the datasets and any preprocessing you performed. Include any statistical analysis of the data that you deem most important.  
– **Results:** This should be the longest section of the report. Describe the results of the experiments and any other interesting results you obtained through more in-depth analysis. You may include tables and/or figures.  
– **Discussion and Conclusion:** Summarize the key findings of the work and suggest any future directions to improve performance.  
– **Statement of contributions:** State the breakdown of work among the team.

### 5.2 Evaluation

An breakdown of the grading of the assignment is as follows: Report and code: 60 percent; demo: 40 percent.

**Report:**

**a.** **Completeness (15/60)** – did you submit all required components? Did you execute all required experiments? Did you follow the guidelines for the report?

**b.** **Correctness (25/60)** – are the models implemented correctly? Are the reported results similar to ours? Do you observe the correct trends in your analysis? Is your interpretation of the results correct and do you provide adequate justification?

**c.** **Writing quality (20/60)** – is the report free of grammatical errors and typos? Is the English usage satisfactory? Is the writing clear and easy to understand? If you include them, are figures and tables correctly presented? Do you appropriately cite referenced work? Are your citations correctly formatted?

**Demo:**

Each member of the team will be asked questions for approximately 5 minutes about the entire submission (not just their own contribution). The demo grade for each student will be assigned based on the understanding of the code, the algorithms, and the results.
