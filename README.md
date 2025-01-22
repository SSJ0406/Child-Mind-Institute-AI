# 🌐 Child Mind Institute Problematic Internet Use Challenge

## 📋 Overview
This project was developed as part of the Child Mind Institute's Kaggle competition to predict and analyze problematic internet use (PIU) in children and adolescents. The goal was to create a robust machine learning pipeline to handle missing data and identify key factors contributing to severe internet involvement.

---

## 🎯 Key Results

- **Model 1:**
  - **Accuracy:** 59.43%
  - **First 10 Predictions:** [2, 0, 0, 1, 2, 1, 0, 0, 0, 2]
- **Model 2:**
  - **Accuracy:** 68.99%
  - **First 10 Predictions:** [2, 0, 0, 1, 4, 1, 0, 4, 4, 4]
- **Model 3:**
  - **Accuracy:** 71.11%
  - **First 10 Predictions:** [2, 0, 0, 1, 2, 1, 0, 0, 0, 2]
 
    ![image](https://github.com/user-attachments/assets/ddc0a5dc-c05f-4a76-9295-535f3ff4ca8a)


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

---

## 📊 Key Visualizations

- **Correlation with Missing Values in ‘sii’:** Highlights variables most correlated with missingness in ‘sii’, aiding feature selection.

![image](https://github.com/user-attachments/assets/f2aa12ab-0bbe-4817-8472-0fe827da64b1)

  
- **Distribution of Missing Data:** Visualizes the gender distribution of missing values in ‘sii’.

![image](https://github.com/user-attachments/assets/b03c89fe-0e31-4e7e-88ba-f458ee6e4a02)


- **Class Distribution in ‘sii’:** Demonstrates the frequency of ‘sii’ classes, including missing data.

![image](https://github.com/user-attachments/assets/f28d4b60-2d47-4230-9ee7-bd9a8204dc0a)

  
- **Improved Correlation Matrix:** Showcases the strongest relationships (threshold ±0.6) for feature engineering.

![image](https://github.com/user-attachments/assets/1b747fc4-1ae6-48bf-9ce8-ea422b14f48a)

  
- **Top 20 Feature Importance:** Provides insights into the most impactful predictors of ‘sii’.

  ![image](https://github.com/user-attachments/assets/c2eebc04-6718-4564-9ee7-d8bf0c849e00)


---

## 🛠️ What Was Done

### Models Used
- **Random Forest Classifier (Model 1):** A baseline model trained only on non-missing values.
- **XGBoost Classifier (Model 2):** Handled missing values by treating them as a separate class.
- **LightGBM Classifier (Model 3):** Enhanced the dataset by imputing missing values with predictions from Model 1.

### Data Splitting
- Training data was split using a **5-fold cross-validation** strategy to ensure robustness.

### Feature Engineering
- Normalized features such as BMI, total internet usage, and depression scores.
- Added a binary indicator variable for missing data to capture dependencies.

### Handling Missing Data
- Imputed missing values in the ‘sii’ column using predictions from the baseline Random Forest model.
- Treated missing values as a distinct class for comparison.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/child-mind-piu.git
   cd child-mind-piu
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
