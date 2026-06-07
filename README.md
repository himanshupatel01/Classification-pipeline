# End-to-End Loan Default Machine Learning Classification Pipeline

## 📌 Project Overview
This project builds a complete, machine learning pipeline to classify credit applications as either **"Good"** or **"Bad"** standing. Using a dataset of 800 distinct loan applicants across 15 features, the pipeline handles everything from exploratory data analysis (EDA) to feature engineering, model tuning, and ensemble deployment. 

The primary objective was maximising risk detection accuracy while maintaining strong model generalisation on unseen validation data.

---

## 🛠️ Technical Workflow & Frameworks
* **Data Processing & EDA:** Pandas, NumPy, Seaborn, Matplotlib. Managed a 70:30 class imbalance and applied permanent label encoding to categorical strings.
* **Feature Engineering:** Built predictive interaction terms including a *Duration-to-Balance Ratio* and an *Age-Duration Interaction* variable.
* **Data Splitting & Scaling:** Implemented a stratified 80/20 train/test split to protect class proportions, followed by data leakage-free `StandardScaler` transformations.
* **Hyperparameter Tuning:** Utilised `GridSearchCV` with 5-fold cross-validation to optimise decision paths and eliminate model overfitting.

---

## 📊 Model Evaluation & Benchmarks (5-Fold CV)
The pipeline trained and benchmarked 5 distinct model architectures:
1. **Logistic Regression:** Baseline linear classifier using ridge regularisation and L-BFGS optimization (**71.88% Accuracy**).
2. **Linear Discriminant Analysis (LDA):** Separated distributions by maximizing between-class variance (**71.72% Accuracy**).
3. **Decision Tree:** Recursive binary splitter optimised to a max depth of 5 (**73.44% Accuracy**).
4. **Random Forest Classifier:** Bootstrap aggregated ensemble using 100 unrestricted trees (**76.41% Accuracy**).
5. **Multi-Layer Perceptron (Neural Network):** A 4-layer backpropagation network (`14 ➔ 50 ➔ 25 ➔ 2`) with ReLU activation and ADAM optimizer (**70.63% Accuracy**).

---

## 🏆 Final Production Deployment & Insights

### 1. Soft-Voting Ensemble Model
To capture both linear and non-linear patterns, a **VotingClassifier** was deployed to pool predictions from the Random Forest, Decision Tree, and Logistic Regression models. 
* **Ensemble Validation Accuracy:** Achieved a top performance of **77.50%**.
* **Discrimination Power:** Reached an **ROC AUC score of 0.7989**, confirming a strong ability to separate high-risk defaults from safe profiles.

### 2. Feature Importance Validation
Evaluating the Random Forest split architecture proved that financial capacity and custom-engineered features dominated risk discovery:
* **Top Predictor:** Custom `duration_balance_ratio` (**13.72% importance**).
* **Financial Capacity:** Combined `credit_balance` and `checking_balance` drove **25.55%** of model decisions.
* **Interaction Validation:** Custom engineered terms accounted for **23.75%** of total feature importance, validating the preprocessing strategy.
