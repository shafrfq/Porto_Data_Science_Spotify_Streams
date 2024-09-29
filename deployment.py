import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title('Spotify Song Popularity Prediction')
st.write('This app analyzes factors influencing song popularity on Spotify and predicts song streams using machine learning models.')

# Upload Data
st.header('Upload Dataset')
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data Understanding
    st.write('## Data Understanding')
    st.dataframe(df.head())

    st.write('## Summary Statistics')
    st.write(df.describe())

    # Data Distribution Plots
    st.write('## Data Distribution Plots')
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[column], bins=30, kde=True)
        plt.title(f'Distribution of {column}')
        st.pyplot(plt)

    # Outlier Detection
    st.write('### Outlier Detection with IQR')
    # Select only numeric columns from df
    numeric_cols = df.select_dtypes(include=[float, int])

    # Calculate the IQR (Interquartile Range)
    q1 = numeric_cols.quantile(0.25)
    q3 = numeric_cols.quantile(0.75)
    iqr = q3 - q1

    # Apply the outlier filter to numeric columns only
    outlier_filter = (numeric_cols < (q1 - 1.5 * iqr)) | (numeric_cols > (q3 + 1.5 * iqr))

    # Optionally, remove outliers from the dataframe
    df_filtered = df[~outlier_filter.any(axis=1)]

    # Boxplot for Outliers
    st.write('### Boxplot for Outliers')
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(10, 2))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        st.pyplot(plt)

    # Data Preparation: Handling Missing Values
    st.header('Data Preparation')
    st.write('Checking for missing values:')
    st.write(df.isnull().sum())

    if st.button('Handle Missing Values'):
        df.fillna(df.mean(), inplace=True)
        st.success('Missing values handled.')
        st.dataframe(df.head())

    # Feature and Target Selection for Model Training
    st.header('Train Machine Learning Models')
    features = df.columns.tolist()
    X_columns = st.multiselect('Select Features (X)', features)
    y_column = st.selectbox('Select Target (y)', features)

    if len(X_columns) > 0 and y_column:
        X = df[X_columns]
        y = df[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        model_choice = st.selectbox('Choose a Model', ['Linear Regression', 'Polynomial Regression', 'Random Forest'])

        # Model evaluation function
        def evaluate_model(model, X_train, X_test, y_train, y_test):
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_train = np.sqrt(mse_train)
            rmse_test = np.sqrt(mse_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            return mse_train, rmse_train, r2_train, mse_test, rmse_test, r2_test

        # Model training and evaluation
        if st.button('Train Model'):
            if model_choice == 'Linear Regression':
                model = LinearRegression()
                model.fit(X_train, y_train)
            elif model_choice == 'Polynomial Regression':
                degree = st.slider('Choose Polynomial Degree', 2, 5)
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_pred_test = model.predict(X_test_poly)
            elif model_choice == 'Random Forest':
                n_estimators = st.slider('n_estimators', 10, 100, step=10)
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)

            # Predictions and evaluation
            if model_choice == 'Polynomial Regression':
                y_pred_test = model.predict(X_test_poly)
            else:
                y_pred_test = model.predict(X_test)

            mse_test = mean_squared_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)

            st.write(f'Mean Squared Error: {mse_test}')
            st.write(f'R-squared: {r2_test}')

            # Actual vs Predicted plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred_test, alpha=0.5)
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            st.pyplot(plt)

    # Correlation Matrix
    st.write('### Correlation Matrix:')
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Top Tracks and Artists Visualization
    st.write('### Top Streaming Tracks and Artists:')
    if 'streams' in df.columns and 'track_name' in df.columns:
        top_tracks = df.groupby('track_name')['streams'].sum().sort_values(ascending=False).head(10)
        st.write(top_tracks)

        top_tracks.plot(kind='bar', figsize=(10, 6))
        plt.title('Top 10 Tracks by Streams')
        plt.ylabel('Total Streams')
        st.pyplot(plt)

    # Button for Pie Chart Visualization
    if st.button('Show Pie Chart Visualizations'):
        # Grouping data by released_day
        delay_per_day = df.groupby('released_day')[['in_spotify_playlists', 'in_apple_playlists']].sum()

        # Pie Chart Labels
        pieChartLabels = ['In Spotify Playlists', 'In Apple Playlists']

        # Colors palette
        myColors = sns.color_palette('pastel')

        # Pie Chart per day visualization
        for i in range(1, 8):
            b = delay_per_day.iloc[i-1, :]  # Data for that day
            plt.figure(figsize=(6, 6))
            plt.pie(b, labels=pieChartLabels, colors=myColors, autopct='%.0f%%')
            plt.title('Released Day ' + str(i))
            st.pyplot(plt)
