# Megaline Plan Recommendation System 📱

## 📜 Project Overview
This project involves developing a machine learning model for **Megaline**, a mobile carrier, to recommend the most suitable plan for its subscribers: **Smart** or **Ultra**. Using behavioral data from subscribers who have already switched to the new plans, the objective is to classify users into the correct plan with an accuracy threshold of at least **0.75**.

---

## 📂 Dataset Description
The dataset (`users_behavior.csv`) contains the following columns:
- **calls**: Number of calls made in a month.
- **minutes**: Total call duration in minutes.
- **messages**: Number of text messages sent in a month.
- **mb_used**: Internet traffic used in MB.
- **is_ultra**: Current plan for the month:
  - `1` for **Ultra**
  - `0` for **Smart**

---

## 🚀 Project Objectives
1. **Data Exploration**:
   - Review the dataset for completeness and accuracy.

2. **Data Splitting**:
   - Divide the dataset into:
     - Training set
     - Validation set
     - Test set

3. **Model Development**:
   - Train multiple classification models with varying hyperparameters.
   - Evaluate models based on their performance metrics, primarily **accuracy**.

4. **Model Validation**:
   - Compare model performance on the validation set.
   - Select the best-performing model.

5. **Model Testing**:
   - Evaluate the model's performance on the test set to ensure accuracy meets or exceeds **0.75**.

6. **Sanity Check**:
   - Perform additional analysis to verify the robustness of the model.

---

## 📊 Workflow

### Step 1: Data Exploration
- Review the dataset and understand the distribution of features.
- Identify potential data anomalies or outliers.

### Step 2: Data Splitting
- Split the dataset into:
  - **Training Set**: Used for model training.
  - **Validation Set**: Used to fine-tune hyperparameters and evaluate model performance.
  - **Test Set**: Used to evaluate the final model's accuracy.

### Step 3: Model Development
- Train multiple machine learning classification models:
  - Decision Tree
  - Random Forest
  - Logistic Regression
- Experiment with hyperparameters for each model to optimize performance.

### Step 4: Model Validation
- Compare model performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Select the best model based on validation set results.

### Step 5: Model Testing
- Evaluate the selected model on the test set.
- Ensure accuracy meets or exceeds the 0.75 threshold.

### Step 6: Sanity Check
- Perform additional testing to validate model robustness and interpretability.

---

## 🛠 Tools and Libraries Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`: For data exploration and manipulation.
  - `scikit-learn`: For machine learning model development and evaluation.
  - `matplotlib` & `seaborn`: For data visualization.

---

## 📈 Key Insights
- Identified patterns in user behavior that correlate with plan selection.
- Experimented with various models to achieve optimal performance.
- Validated the model's accuracy using a test dataset to ensure it met business requirements.

---

## 📄 File Structure
- **`users_behavior.csv`**: Dataset containing monthly user behavior data.
- **`plan_recommendation.ipynb`**: Jupyter Notebook containing the analysis, model training, and evaluation code.

---

## 🔍 How to Run
1. Clone the repository:
