
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# --- Load dataset and prepare model ---
@st.cache_resource
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("Melbourne_housing_FULL.csv")

    # Drop rows with missing target
    df = df.dropna(subset=['Price'])

    # Target
    y = df['Price']

    # Features for simplified model
    deploy_features = ['Rooms', 'Bathroom', 'Car', 'Suburb']
    X = df[deploy_features]

    # Preprocessing pipeline
    preprocess_deploy = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Rooms','Bathroom','Car']),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Suburb'])
    ])

    # Final pipeline
    final_pipe = Pipeline([
        ('preprocess', preprocess_deploy),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]).fit(X, y)

    return final_pipe

# Train once and cache
model = load_and_train_model()

# --- Streamlit UI ---
st.title("Melbourne Housing Price Predictor (simplified demo)")

rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
bathroom = st.number_input("Bathrooms", min_value=1, max_value=5, value=1)
car = st.number_input("Car spaces", min_value=0, max_value=5, value=1)
suburb = st.text_input("Suburb", "Richmond")

row = pd.DataFrame([[rooms, bathroom, car, suburb]], 
                   columns=['Rooms','Bathroom','Car','Suburb'])

if st.button("Predict Price"):
    pred = model.predict(row)[0]
    st.success(f"Predicted Price: AUD {pred:,.0f}")

