# Mini Project Report Outline

## Abstract
Social media platforms generate large amounts of post-performance data. This project applies statistical analysis and machine learning to study how post metadata relates to user engagement and to predict the expected number of likes. The workflow includes data cleaning, exploratory analysis, feature engineering, model training, and evaluation using MAE, MSE, RMSE, and R².

## Problem statement
It is difficult for content creators to estimate how a post may perform before publishing it. This project builds a data-driven model that estimates likes from available post metadata.

## Objectives
1. Inspect and clean social media post data.
2. Compare post counts and average likes across platforms.
3. Study the relationship between caption length and likes.
4. Train Linear Regression and Random Forest models.
5. Compare model performance using standard regression metrics.

## Methodology
Data is loaded from a CSV file and checked for missing values and duplicates. Caption length, day of week, and month are derived from existing fields. Categorical features are one-hot encoded and numeric features are scaled. The data is split into training and test sets. Linear Regression and Random Forest Regressor are trained and compared.

## Statistical and evaluation measures
- Mean Absolute Error (MAE): average absolute prediction error.
- Mean Squared Error (MSE): average squared prediction error.
- Root Mean Squared Error (RMSE): square root of MSE, in the target's units.
- R² score: proportion of variation explained by the model, with caveats for poor models.

## Limitations
Results depend on data quality and sample size. Social media engagement also depends on audience size, time, trends, and platform algorithms, which may not be available in the dataset. Predictions are estimates, not guarantees.

## Conclusion
The project demonstrates a complete statistical machine-learning workflow for social media post analysis. Actual conclusions and the best-performing model should be reported only after running the code on the final dataset.
