# Oil Well Region Selection Project

## Overview
This project was developed for OilyGiant, a mining company, to determine the optimal location for a new oil well. The goal is to use data-driven methods to identify the region with the highest potential profit and the lowest risk of losses. This involves training machine learning models to predict oil well reserves and performing financial and risk analysis to guide business decisions.

## Objectives
- **Predict Oil Reserves**: Build a machine learning model using linear regression to predict the volume of reserves for new oil wells in three regions.
- **Maximize Profit**: Select the region with the highest total profit based on the predicted reserves.
- **Minimize Risk**: Assess the risk of loss using the bootstrapping technique and ensure that selected regions maintain a risk of loss below 2.5%.

## Steps
1. **Data Preparation**: 
   - Load and preprocess the oil well data from three regions.
   - Split the data into training and validation sets.

2. **Model Training**:
   - Train linear regression models on the training data to predict oil well reserves.
   - Evaluate the model's performance using Root Mean Square Error (RMSE).

3. **Profit Calculation**:
   - Calculate the expected profit for each region based on the predictions.
   - Determine the minimum volume of reserves needed to avoid losses.

4. **Risk Analysis**:
   - Use bootstrapping with 1000 samples to estimate the distribution of profit for each region.
   - Calculate the average profit, confidence intervals, and the probability of losses.

5. **Decision Making**:
   - Recommend the region with the highest potential profit and lowest risk of loss for new well development.

## Technologies Used
- **Python**: Data manipulation, machine learning, and analysis
- **Pandas, NumPy**: Data processing and analysis
- **Scikit-learn**: Linear regression modeling
- **Matplotlib, Seaborn**: Data visualization
- **Bootstrapping**: Statistical risk assessment

## Results
- Predicted oil well reserves for three regions.
- Evaluated the profitability of each region while minimizing financial risk.
- Recommended the region with the highest potential return on investment.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
