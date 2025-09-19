# Fraud Detection in the Banking Sector using Machine Learning  

## Project Overview  
This project focuses on detecting fraudulent transactions in the banking sector using machine learning techniques. Fraud detection is a critical challenge because fraudulent transactions are rare compared to legitimate ones, leading to **imbalanced datasets**.  

To address this, I applied multiple ML models and used SMOTE (Synthetic Minority Oversampling Technique) to handle the imbalance.  

## Models Used  
- 🌳 **Random Forest Classifier**  
- ⚡ **XGBoost Classifier**  
- 📈 **Support Vector Machine (SVM with RBF kernel)**  


## Results Summary  

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|-------|-----------|-------------------|----------------|------------------|
| Random Forest + SMOTE | 99.95% | 84.3% | **87.7%** | **86.0%** |
| XGBoost + SMOTE       | 99.94% | 83.0% | 84.7% | 83.8% |
| SVM + RobustScaler + SMOTE | 99.75% | 39.1% | 85.7% | 53.7% |

 **Best Model:** Random Forest (highest balance of recall and F1-score).  
Key Insights  
- **Accuracy alone is not enough** for imbalanced problems like fraud detection.  
- **Recall and F1-score** are more important, since missing fraud is riskier than false alarms.  
- **Random Forest** performed best, while SVM produced more false positives.
  
 ## Tech Stack  
- Python 🐍  
- Jupyter Notebook  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`  
  
##  Project Structure  
fraud-detection-ml
working_with_XGBOOST_RF_SVM.ipynb # Jupyter Notebook with full code
README.md # Project documentation
results # Screenshots of confusion matrices, reports, charts

## Dataset  
- **Credit Card Fraud Detection Dataset** from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
- Contains **284,807 transactions** with **492 frauds (0.17%)**.  

##  Sample Output  
- Confusion Matrices for each model  
- Classification Reports  
- Comparison Chart (Recall vs F1-Score)  


## Acknowledgments  
- Dataset provided by Kaggle.  
- Libraries: scikit-learn, imbalanced-learn, XGBoost.  


*Author: Daniel Udeme*  
*Email: xavidanito35@gmail.com 
