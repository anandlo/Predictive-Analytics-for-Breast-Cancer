# Predictive Analytics for Breast Cancer Diagnosis

This project focuses on developing a predictive model for breast cancer diagnosis using machine learning techniques. We utilize the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository to build and evaluate our models.

## Project Overview

The main objectives of this project are:

1. Analyze the Breast Cancer Wisconsin (Diagnostic) dataset
2. Preprocess the data for machine learning
3. Develop and train predictive models
4. Evaluate model performance
5. Provide insights for breast cancer diagnosis

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

**Key Details:**
- **Source**: UCI Machine Learning Repository
- **Instances**: 569
- **Features**: 30
- **Target Variable**: Diagnosis (Malignant or Benign)

## Requirements

- Python 3.x
- Required Python libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Jupyter Notebook (for running the analysis)

## Setup

1. Clone this repository
2. Install the required Python libraries:
3. Run Jupyter Notebook:


## Project Structure

- `Predictive_Analytics_for_Breast_Cancer_Diagnosis.ipynb`: Main collab notebook containing the analysis and model development

## Key Features

- **Data Preprocessing**: Handling missing values, feature scaling, and encoding
- **Exploratory Data Analysis**: Visualizing feature distributions and correlations
- **Model Development**: Implementing various machine learning algorithms (e.g., Logistic Regression, Random Forest, SVM)
- **Model Evaluation**: Using metrics such as accuracy, precision, recall, and F1-score
- **Feature Importance**: Analyzing which features are most predictive of breast cancer diagnosis

# Results

## Overview
In this project, we implemented a machine learning pipeline using scikit-learn to automate the process of data preparation, model training, and evaluation. The focus was on mitigating common issues such as data leakage and establishing a clear workflow through the use of Pipelines.

## Data Preparation and Modeling Pipeline
The dataset was loaded, and a validation set was established. The data was split into training and testing sets, and the class labels were encoded. The pipeline allowed for the chaining of data transformations and modeling processes, leading to a more efficient workflow.

### Model Evaluation
We evaluated six different algorithms using 10-fold cross-validation to assess their accuracy on the training dataset:

| Model | Mean Accuracy | Standard Deviation |
|-------|---------------|---------------------|
| Logistic Regression (LR) | 0.944808 | 0.029070 |
| Linear Discriminant Analysis (LDA) | 0.957372 | 0.029611 |
| K-Nearest Neighbors (KNN) | 0.929872 | 0.036675 |
| Decision Tree Classifier (CART) | 0.922051 | 0.045614 |
| Naive Bayes (NB) | 0.934679 | 0.044038 |
| Support Vector Machine (SVM) | 0.605769 | 0.073092 |

### Observations
- **Best Performers**: Logistic Regression and LDA demonstrated promising results and were identified for further investigation.
- **SVM Performance**: The SVM model showed surprisingly low accuracy, potentially influenced by the data's varied distribution.

## Comparison of Algorithms
A boxplot comparison revealed that all classifiers, except SVM, displayed a similar distribution of accuracies, indicating low variance across the models. 

### Standardization of Data
After standardizing the dataset, we observed an improvement in the SVM model's performance:

| Model | Mean Accuracy | Standard Deviation |
|-------|---------------|---------------------|
| Scaled Logistic Regression (ScaledLR) | 0.943 | 0.032 |
| Scaled LDA (ScaledLDA) | 0.958 | 0.028 |
| Scaled KNN (ScaledKNN) | 0.930 | 0.037 |
| Scaled Decision Tree (ScaledCART) | 0.925 | 0.045 |
| Scaled Naive Bayes (ScaledNB) | 0.935 | 0.043 |
| Scaled SVM (ScaledSVM) | 0.903 | 0.045 |

### Algorithm Tuning
We tuned the hyperparameters for the best-performing algorithms (LR, LDA, and SVM). After applying hyperparameter optimization for the SVM classifier, we achieved the following results:

- **Final Model Training Accuracy**: 0.944 ± 0.032
- **Final Accuracy on Test Set**: 0.94737

### Classification Report
The final model performance on the test set was evaluated using the following metrics:

- **Accuracy**: 0.947368421053

- **Confusion Matrix**:  
  |                | Predicted B | Predicted M |
  |----------------|-------------|-------------|
  | **Actual B**   | 113         | 3           |
  | **Actual M**   | 6           | 49          |

- **Classification Report**:  

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| B     | 0.95      | 0.97   | 0.96     | 116     |
| M     | 0.94      | 0.89   | 0.92     | 55      |
| **Avg / Total** | **0.95** | **0.95** | **0.95** | **171** |



### Conclusion
The implementation of the ML pipeline effectively automated the workflow and provided a structured approach to model evaluation and selection. The results highlighted the importance of standardization and hyperparameter tuning in improving model accuracy.

## Future Improvements

- Implement advanced feature engineering techniques
- Explore deep learning models for improved accuracy
- Develop a web application for real-time breast cancer risk assessment

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- UCI Machine Learning Repository for providing the dataset
- The creators of the Breast Cancer Wisconsin (Diagnostic) dataset: William Wolberg, Olvi Mangasarian, Nick Street, and W. Street

