# 🌐 Child Mind Institute Problematic Internet Use Challenge

## 📋 Overview
This project was developed as part of the Child Mind Institute's Kaggle competition to predict and analyze problematic internet use (PIU) in children and adolescents. The goal was to create a robust machine learning pipeline to handle missing data, work with imbalanced and inconsistently labeled datasets, and identify key factors contributing to severe internet involvement. The project involved:

- 🔍 **Data Exploration**: Addressing missing labels, analyzing imbalanced classes, and tackling multi-format data (classification & time-series, stored in **Parquet** and **CSV** formats).
- 🧐 **Data Preparation**: Normalizing numerical features, transforming categorical variables, and ensuring consistency.
- 🤖 **Model Training**: Testing three different strategies for handling missing target values.
- 🎨 **Visualization**: Gaining insights through exploratory plots, heatmaps, and correlation analysis.
- 🚀 **Submission**: Generating and refining predictions for the Kaggle challenge.

Additionally, **PySpark** was used for processing large-scale parquet files efficiently, and extra feature testing was conducted on parquet-based datasets.

---

## 💻 Technologies Used

- **Programming Language**: Python
- **Data Processing & Wrangling**: Pandas, NumPy, PySpark
- **Machine Learning Frameworks**: Scikit-learn, XGBoost, LightGBM
- **Visualization Libraries**: Matplotlib, Seaborn
- **Model Evaluation & Validation**: Cross-validation, Stratified K-Fold, Confusion Matrices

---

## 🎯 Key Results

- **Model 1:**
  - **Accuracy:** 59.43%
  - **ROC-AUC:** 0.71

- **Model 2:**
  - **Accuracy:** 68.99%
  - **ROC-AUC:** 0.75

- **Model 3:**
  - **Accuracy:** 71.11%
  - **ROC-AUC:** 0.78

---

### 🧰 What Was Done

### Data Exploration
- **Previewed Dataset:**
  - Analyzed data types, missing values, and inconsistencies.

  ```python
  print(train_data.info())
  print(train_data.head())
  ```

- **Handling Missing Data:**
  - Created a binary indicator column for missing target values.
  
  ```python
  train_data['target_missing'] = train_data['target'].isnull().astype(int)
  ```

### Data Preparation
- **Normalized Numerical Features:**
  
  ```python
  train_data['BMI_norm'] = train_data['Physical-BMI'] / train_data['Physical-BMI'].max()
  ```

- **One-Hot Encoding for Categorical Variables:**
  
  ```python
  preprocessor = ColumnTransformer([
      ('num', StandardScaler(), numeric_features),
      ('cat', OneHotEncoder(drop='first'), categorical_features)
  ])
  ```

### Model Training
- **Model 1: Excluded Missing Target Values**
  
  ```python
  pipeline_model1 = Pipeline([
      ('preprocessor', preprocessor),
      ('classifier', RandomForestClassifier(random_state=42))
  ])
  pipeline_model1.fit(X_model1, y_model1)
  ```

- **Model 2: Treated Missing Values as a Separate Class**
  
  ```python
  train_data['target'] = train_data['target'].fillna(4).astype(int)
  pipeline_model2.fit(X_model2, y_model2)
  ```

- **Model 3: Imputed Missing Values Using Predictions**
  
  ```python
  train_data.loc[train_data['target'] == 4, 'target'] = pipeline_model1.predict(
      train_data.loc[train_data['target'] == 4, features]
  )
  pipeline_model3.fit(X_model3, y_model3)
  ```

- **Cross-Validation:**
  
  ```python
  cv_scores = cross_val_score(pipeline_model1, X_model1, y_model1, cv=5, scoring='accuracy')
  ```
---

## 📊 Key Visualizations


- **Distribution of Missing Data:** Visualizes the gender distribution of missing values in targets.

![image](https://github.com/user-attachments/assets/b03c89fe-0e31-4e7e-88ba-f458ee6e4a02)


- **Class Distribution in ‘sii’:** Demonstrates the frequency of target classes, including missing data.

![image](https://github.com/user-attachments/assets/f28d4b60-2d47-4230-9ee7-bd9a8204dc0a)

  
- **Improved Correlation Matrix:** Showcases the strongest relationships (threshold ±0.6) for feature engineering.

![image](https://github.com/user-attachments/assets/1b747fc4-1ae6-48bf-9ce8-ea422b14f48a)

  
- **Top 20 Feature Importance:** Provides insights into the most impactful predictors of targets.

  ![image](https://github.com/user-attachments/assets/c2eebc04-6718-4564-9ee7-d8bf0c849e00)


---

## 🗂️ Repository Structure

```plaintext
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset
├── data_dictionary.csv        # Data dictionary
├── Child Mind Institute.ipynb # Jupyter notebook containing the analysis
├── sample_submission.csv      # Sample format for Kaggle submission
├── README.md                  # Project documentation
```

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/SSJ0406/Child-Mind-Institute-AI.git
   cd Child-Mind-Institute-AI
   ```

2. Create the `requirements.txt` file if it’s missing:

   ```bash
   pip freeze > requirements.txt
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Open and run the Jupyter notebook:

   ```bash
   jupyter notebook "Child Mind Institute.ipynb"
   ```

---

## 💡 Reflections and Future Work

- **Business Relevance:** Insights from this analysis can inform policymakers and educators on identifying vulnerable groups and crafting tailored interventions to combat PIU.
- **Technical Learnings:** Iterative imputation for missing data significantly improved model performance. Feature engineering based on domain knowledge played a pivotal role in enhancing predictive accuracy.
- **Future Directions:**
  - Extending this analysis with deep learning models.
  - Exploring time-series data for a longitudinal perspective on PIU progression.

---

## 🤝 Contributions

If you have ideas or suggestions for improving this project, feel free to open an issue or submit a pull request. Contributions are welcome!

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🙏 Acknowledgments

- Child Mind Institute for hosting the Kaggle competition.
- The Kaggle community for fostering a collaborative learning environment.
