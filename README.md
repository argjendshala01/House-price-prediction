# House-price-prediction
This project builds a machine learning pipeline to predict house prices using the House Prices dataset from Kaggle.
It combines data preprocessing, feature engineering, hyperparameter optimization (Optuna), model training (XGBoost), and model interpretability (SHAP).

We loaded the dataset and handled missing values by filling numeric columns with zeros and categorical ones with "None". The LotFrontage column was imputed using regression within each neighborhood. Quality-related features were ordinal‑encoded, and new features like HouseAge, YearsSinceRemod, and WasRemodeled were created.

We log‑transformed the target SalePrice to stabilize variance, then selected the top predictive features with SelectKBest. The data was split into training and test sets and standardized.

Using Optuna, we tuned XGBoost hyperparameters and trained the final model. Performance was evaluated with RMSE on both log and original scales, and results were visualized with actual vs. predicted plots. Finally, SHAP values were used to interpret the model and highlight the most important features driving house prices.
