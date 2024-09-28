import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Title and Description
st.title('Spotify Song Popularity Prediction')
st.write('Menganalisis faktor-faktor yang mempengaruhi popularitas lagu di Spotify dan memprediksi jumlah streaming lagu menggunakan model Machine Learning.')

# Load Data
st.header('Upload Dataset')
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data Understanding
    st.write('Data Understanding')
    st.write('Data Preview:')
    st.dataframe(df.head())
    
    st.write('Ringkasan Statistik')
    st.write(df.describe())

    st.write('Data Distribution Plots')
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f'Distribution of {column}')
        st.pyplot(plt)

    # Data Preparation
    st.header('Data Preparation')
    st.write('Memeriksa missing values:')
    st.write(df.isnull().sum())
    
    if st.button('Menangani Missing Values'):
        # Example: Fill missing values with mean (you can adjust this)
        df.fillna(df.mean(), inplace=True)
        st.success('Missing values ditangani.')

    st.write('Preview Data yang Sudah Dibersihkan:')
    st.dataframe(df.head())

    # Splitting data into train and test sets
    st.header('Latih Model Machine Learning')
    st.write('Pilih fitur dan target untuk pelatihan model:')
    features = df.columns.tolist()
    X_columns = st.multiselect('Select Features (X)', features)
    y_column = st.selectbox('Select Target (y)', features)

    if len(X_columns) > 0 and y_column:
        X = df[X_columns]
        y = df[y_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        model_choice = st.selectbox('Pilih Model', ['Linear Regression', 'Polynomial Regression', 'Random Forest'])

        # Model training
        if st.button('Latih Model'):
            if model_choice == 'Linear Regression':
                model = LinearRegression()
                model.fit(X_train, y_train)
            elif model_choice == 'Polynomial Regression':
                degree = st.slider('Polynomial Degree', 2, 5)
                poly = PolynomialFeatures(degree=degree)
                X_poly_train = poly.fit_transform(X_train)
                model = LinearRegression()
                model.fit(X_poly_train, y_train)
                X_poly_test = poly.transform(X_test)
            elif model_choice == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

            # Predictions and performance metrics
            if model_choice == 'Polynomial Regression':
                y_pred = model.predict(X_poly_test)
            else:
                y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f'Mean Squared Error: {mse}')
            st.write(f'R-squared: {r2}')

            # Actual vs Predicted Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            st.pyplot(plt)
