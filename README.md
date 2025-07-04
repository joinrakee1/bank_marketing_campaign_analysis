# Bank Term Deposit Subscription Prediction

This project applies machine learning techniques to predict whether a client will subscribe to a term deposit based on data from a Portuguese bank's marketing campaign. It uses Logistic Regression and Random Forest classifiers to model and evaluate performance on the UCI Bank Marketing Dataset.

This project was completed as part of a **Supervised Machine Learning course assignment**.

## Files

**Included in this repository:**
- `bank_term_deposit_prediction.ipynb`: Jupyter notebook with data preprocessing, exploratory analysis, model building, training, evaluation, and comparison of models.
- `bank_term_deposit_prediction.pdf`: PDF version of the complete notebook for easy viewing and submission.

**Dataset Not Included**  
To run the notebook, download the dataset from the [UCI Bank Marketing Dataset page](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

## Models

Three classification models were developed and compared:
1. **Model 1 – Baseline Logistic Regression**  
   With standard preprocessing and feature scaling.

2. **Model 2 – Tuned Logistic Regression**  
   With improved preprocessing (handling 'unknown' values) and hyperparameter tuning.

3. **Model 3 – Random Forest Classifier**  
   Tree-based ensemble model using original one-hot encoded features.

## Results

All models achieved high overall accuracy (~89–90%), but performance varied in precision and recall due to class imbalance. The tuned logistic regression model offered the best balance of metrics, while the random forest achieved the highest recall. Full performance metrics, confusion matrices, ROC curves, and bar charts are provided in the notebook.

## Requirements

- Python 3.10
- scikit-learn
- pandas
- seaborn
- matplotlib
- numpy

Install dependencies with:

```bash
pip install pandas seaborn matplotlib scikit-learn
