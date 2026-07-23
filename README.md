# 🧠 Stroke Prediction & Risk Analysis

## 📌 Project Overview
This project builds a machine learning model to predict the likelihood of a stroke based on patient health metrics. The primary goal was not just high accuracy, but **high Recall**—ensuring the model catches as many actual stroke cases as possible, reflecting real-world healthcare priorities where false negatives are dangerous.

## 📊 The Data
- **Source:** Kaggle Stroke Prediction Dataset
- **Size:** 5,110 patients, 12 initial features
- **Key Challenge:** Severe class imbalance (only ~4.8% of patients had a stroke).

## 🛠️ Methodology & Pipeline
1. **Data Cleaning:** Handled missing `bmi` values using median imputation and resolved 'Unknown' categories in `smoking_status`.
2. **Preprocessing:** Applied One-Hot Encoding to categorical variables and `StandardScaler` to normalize numerical features, preventing data leakage by fitting only on training data.
3. **Handling Imbalance:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set to give the model balanced examples to learn from.
4. **Modeling:** Trained a Random Forest Classifier.
5. **Optimization:** Adjusted the classification threshold from the default 0.5 to **0.3** to prioritize catching true positives.

## 📈 Results & The "Accuracy Trap"
Initially, a baseline model achieved 94% accuracy but a dismal **6% Recall** for strokes (catching only 3 out of 50 cases). By applying SMOTE and optimizing the decision threshold, the final model achieved:

| Metric | Baseline Model | Final Optimized Model |
| :--- | :---: | :---: |
| **Accuracy** | 94.3% | 84.0% |
| **Recall (Stroke)** | **6.0%** | **44.0%** |
| **Precision (Stroke)**| 21.0% | 14.0% |

### 💡 Clinical Takeaway
While precision dropped, the trade-off is clinically justified. A lower threshold means the model flags more patients for review. In a hospital setting, a false positive simply triggers further diagnostic tests (like an MRI), whereas a false negative could result in a missed, life-threatening stroke. 

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn`
3. Run the Jupyter Notebook: `jupyter notebook 01_stroke_eda.ipynb`

## 👤 Author
Karolyne Muniz - https://www.linkedin.com/in/karolyne-muniz
