# 🩺 Breast Cancer Classification

📥 The dataset is loaded directly from `sklearn.datasets` using the built-in **Breast Cancer Wisconsin (Diagnostic)** dataset.

This project focuses on building a machine learning model to classify breast cancer tumors as malignant or benign. We apply the powerful XGBoost classifier with hyperparameter tuning to achieve high classification accuracy.

---

## 📂 Dataset  
The dataset consists of 30 numerical features computed from digitized images of breast mass nuclei, including:

• Mean radius, texture, perimeter, area, smoothness, and more  
• Target variable: **diagnosis** (0 = malignant, 1 = benign)

The data is loaded and converted into a pandas DataFrame:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
```

---

## 🧹 Data Preprocessing

• Features (**X**) and target (**y**) are separated.

• Train/test split: 80% training and 20% testing sets, fixed random seed for reproducibility.

---

## 🧐 Model Building & Evaluation

We use XGBoost Classifier, an efficient gradient boosting algorithm widely used in classification tasks for its speed and accuracy.

To find the best model parameters, GridSearchCV with 5-fold cross-validation is applied, tuning:

**• n_estimators:** number of boosting rounds

**• max_depth:** tree depth

**• learning_rate:** step size shrinkage

**• subsample:** fraction of samples for each tree

**• colsample_bytree:** fraction of features per tree

**• min_child_weight:** minimum sum of instance weight needed in a child

```python
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)
```
The best parameters and cross-validation score are printed.

The model is then tested on the unseen test set, and metrics reported include accuracy and a detailed classification report showing precision, recall, and F1-score.

---

## 📈 Final Result

✅ **Accuracy:** 0.9737
🎯 **F1-Score:** 0.98

The tuned XGBoost model demonstrates excellent classification performance, achieving over 97% accuracy and an F1-score of 0.98. This indicates high precision and recall in detecting both malignant and benign tumors.

---

## 🛠 Technologies Used

• Python

• Pandas for data handling

• Scikit-learn (sklearn) for dataset loading, data splitting, and evaluation

• XGBoost for powerful gradient boosting classification

• GridSearchCV for hyperparameter tuning

---

## 📬 Author

**Telegram:** @lesin_official

**Email:** dmitrylesin_official@gmail.com

**© 2025 Dmitry Lesin. All rights reserved.**
