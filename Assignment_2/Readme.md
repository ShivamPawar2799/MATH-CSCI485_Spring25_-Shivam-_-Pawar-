# Wine Quality Analysis: PCA and t-SNE #

This repository contains code for analyzing the Wine Quality dataset using Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.
# Requirements

* Python 3.7+
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

Install required packages using

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
# Dataset
The code automatically downloads the Wine Quality dataset from the UCI Machine Learning Repository:

* Red wine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
* White wine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

# How to Use

1. Clone this repository:
```bash
git clone https://github.com/yourusername/MATH_CSCI485_Spring25_Firstname_Lastname.git
cd MATH_CSCI485_Spring25_Firstname_Lastname/Assignment_2
```
2. Run the Jupyter notebook or Python script:
```bash 
jupyter notebook Wine_Quality_Analysis.ipynb
```
or
```bash
python wine_quality_analysis.py
```
3. The code will:
* Load and preprocess the wine quality datasets
* Apply PCA and visualize the results in 2D and 3D
* Apply t-SNE and visualize the results
* Generate various plots and statistics

# Code Structure
The code is organized into three main tasks:

# Task 1: Data Preprocessing

* Loading the red and white wine datasets
* Combining datasets and adding wine type identifier
* Checking for missing values
* Normalizing features using StandardScaler

# Task 2: PCA Implementation

* Applying PCA to reduce dimensionality
* Calculating explained variance for each component
* Visualizing data in 2D and 3D projections
* Analyzing feature contributions to principal components

# Task 3: t-SNE Implementation and Comparison

* Applying t-SNE to obtain 2D representation
* Comparing PCA and t-SNE visualizations
* Analyzing differences in approaches and results

# Output 
The code generates several visualizations:

* explained_variance.png: Bar chart of variance explained by each PC
* pca_2d.png: 2D PCA projection colored by wine quality
* pca_2d_wine_type.png: 2D PCA projection colored by wine type
* pca_3d.png: 3D PCA projection
* pca_feature_contributions.png: Heatmap of feature contributions
* tsne_quality.png: t-SNE projection colored by quality
* tsne_wine_type.png: t-SNE projection colored by wine type

It also saves the processed data and results in:

* pca_tsne_results.npz: Contains all the transformed data

# Results
The main findings from the analysis are:

* The first 2 PCs explain approximately 50.2% of variance
* The first 3 PCs explain approximately 64.4% of variance
* PCA shows clear separation between red and white wines
* t-SNE reveals more distinct clusters than PCA
* Both methods provide complementary insights

For a detailed analysis, please refer to the report.pdf file.

# Troubleshooting
If you encounter issues with loading the dataset, ensure you have internet connectivity or download the files manually and update the file paths in the code.
# License
This project is provided for educational purposes only. The Wine Quality dataset is from the UCI Machine Learning Repository.
