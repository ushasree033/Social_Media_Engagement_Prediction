# Social Media Engagement Prediction Using Statistical Machine Learning

## Project objective
Analyze social media post metadata and predict the expected number of likes using statistical machine-learning models. The project includes data cleaning, exploratory data analysis, statistical summaries, feature engineering, model training, and evaluation.

## Folder structure
- `code/social_media_prediction.py` — complete project code
- `dataset/sample_train.csv` — sample data so the project runs immediately
- `dataset/train.csv` — put your downloaded Kaggle training CSV here (optional; preferred automatically)
- `graphs/` — generated charts
- `results/model_comparison.csv` — model evaluation table

## Run in VS Code
1. Extract the ZIP.
2. Open the extracted `Social_Media_Engagement_Prediction_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Install packages:
   `python -m pip install -r requirements.txt`
5. Run:
   `python code/social_media_prediction.py`

The script uses `dataset/train.csv` if present; otherwise it uses `dataset/sample_train.csv`.
If using your Kaggle file, copy it into the `dataset` folder and name it exactly `train.csv`.

## Models
- Linear Regression
- Random Forest Regressor

## Evaluation
The script reports MAE, MSE, RMSE, and R² on a held-out test set. Lower MAE/RMSE indicate smaller prediction errors; R² closer to 1 generally indicates a better fit.

## Important note for your report
The included `sample_train.csv` is synthetic sample data for demonstrating the workflow. Replace it with your actual dataset for your final results. The real dataset may have different distributions, so reported scores will change. This model predicts likes using platform, content type, caption length, and posting date features; it deliberately does not use likes or comments as input features.
