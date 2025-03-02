# Recursive Feature Elimination with Linear Regression

## Overview
This project implements Recursive Feature Elimination (RFE) using Linear Regression on the Diabetes dataset from scikit-learn. The goal is to identify the most important features that contribute to disease progression prediction.

## Requirements
Ensure you have the following dependencies installed before running the code:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to Use
### 1. Clone the Repository
```bash
git clone https://github.com/ShivamPawar2799/MATH-CSCI485_Spring25_-Shivam-_-Pawar-.git
cd MATH-CSCI485_Spring25_-Shivam-_-Pawar-/Assignment_1
```

### 2. Run the Code
Execute the Python script or Jupyter Notebook:
```bash
python rfe_linear_regression.py
```
OR, if using Jupyter Notebook:
```bash
jupyter notebook rfe_linear_regression.ipynb
```

### 3. Expected Outputs
The script performs the following tasks:
- Loads and explores the Diabetes dataset.
- Splits the data into training and testing sets.
- Trains a linear regression model and evaluates it using R² score.
- Implements RFE to iteratively remove the least important features.
- Identifies the optimal number of features using R² score improvement threshold (0.01).
- Generates visualizations showing R² score vs. number of features.
- Displays the most important features along with their coefficients.

### 4. Output Files
- **visualization.png**: Plot of R² score vs. number of retained features.
- **feature_ranking.csv**: A table ranking features based on importance.
- **final_selected_features.txt**: List of selected features after RFE.

## Example Output
```
10 features retained, R² Score: 0.4523
9 features retained, R² Score: 0.4538
...
Optimal number of features: 5
Selected Features: ['bmi', 'bp', 's1', 's5', 's6']
```

## Contact
For any issues, please contact `your_email@domain.com` or open an issue in the repository.

## README: Diabetes Dataset Linear Regression Assignment ##

# Overview

This assignment implements a linear regression model on the Diabetes dataset using Python and Scikit-learn. The goal is to explore feature selection and evaluate model performance using metrics such as R-squared.


 # Prerequisites #

Ensure you have Python installed along with the following required libraries:

`pip install numpy pandas matplotlib scikit-learn`

## Running the Code

**1.** Open the Jupyter Notebook

Use the following command to start Jupyter Notebook:

`jupyter notebook`

- Open `Assignment_1.ipynb` from the Jupyter interface.

**2.** Execute the Cells in Order

- Run each cell sequentially to load the dataset, preprocess data, train the model, and evaluate performance.

**3.** Understanding the Steps

- Load the Diabetes dataset: Extracts features and target variables.

- Split the dataset: Divides data into training (80%) and testing (20%).

- Train a Linear Regression model: Uses Scikit-learn's LinearRegression.

- Evaluate performance: Computes R-squared score and plots relevant metrics.

## Expected Output

- Feature names and dataset shape.

- Training and testing dataset sizes.

- Model performance evaluation (R-squared score and visualizations).

## Notes

- Modify the random_state parameter in train_test_split for different data splits.

- Experiment with different feature selection techniques for improved performance.

## Author

[Shivam Pawar]
