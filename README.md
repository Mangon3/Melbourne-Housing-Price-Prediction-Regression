# Melbourne Housing Price Predictor

This project develops and evaluates regression models to predict housing prices in Melbourne using the Melbourne_housing_FULL.csv dataset (sourced from Kaggle). It contains multiple Machine Learning features, including data cleaning, feature engineering, model training, evaluation, and deployment via Streamlit. (also the code comments are kind of messy and some parts weren't included so I will be making changes in the future :( )

## Data Preparation & EDA

- Parsed sale dates
- Created derived features
  - Property age
  - Log-transformed land size
- Handled missing values through median/most frequent imputation
- Visualied price distribution

## Preprocessing

Implemented using ColumnTransformer:
1. **Numerical features** standardized after median imputation
2. **Categorical features** encoded with one-hot encoding
3. Data was split into training (80%) and testing (20%) sets

## Model Development

### Trained Regression Models

- Random Forest
- Gradient Boosting
- Linear Regression
- RidgeCV

### Metrics

- MAE
- RMSE
- R²

## Deployment

A Streamlit demo app was used for users to input key features such as rooms, bathrooms, car spaces, and suburb, and receive a predicted house price.

## Tools
- Python: pandas, scikit-learn, numpy, matplotlib
- Streamlit for deployment
- Kaggle dataset: Melbourne_housing_FULL.csv
  - Source: https://www.kaggle.com/datasets/saadmehar/melbourne-housing-fullcsv
