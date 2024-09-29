import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

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
    numeric_cols = df.select_dtypes(include=[float, int])
    q1 = numeric_cols.quantile(0.25)
    q3 = numeric_cols.quantile(0.75)
    iqr = q3 - q1
    outlier_filter = (numeric_cols < (q1 - 1.5 * iqr)) | (numeric_cols > (q3 + 1.5 * iqr))
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
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        st.success('Missing values handled.')
        st.dataframe(df.head())

    # Scaling
    st.header('Data Scaling')
    numerical_col = ['streams', 'bpm', 'danceability_%']  # Add more columns as necessary
    scaler = MinMaxScaler()
    minmax_df = scaler.fit_transform(df[numerical_col])
    minmax_df = pd.DataFrame(minmax_df, columns=numerical_col)

    # Plot Before and After Scaling
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].set_title('Before Scaling')
    sns.kdeplot(df[numerical_col[0]], ax=axes[0], color='r', label=f"Before Scaling: {numerical_col[0]}")
    sns.kdeplot(df[numerical_col[1]], ax=axes[0], color='b', label=f"Before Scaling: {numerical_col[1]}")
    axes[0].legend()

    axes[1].set_title('After Min-Max Scaling')
    sns.kdeplot(minmax_df[numerical_col[0]], ax=axes[1], color='black', label=f"After Scaling: {numerical_col[0]}")
    sns.kdeplot(minmax_df[numerical_col[1]], ax=axes[1], color='blue', label=f"After Scaling: {numerical_col[1]}")
    axes[1].legend()
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    plt.tight_layout()
    st.pyplot(fig)

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
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])

    if not numeric_cols.empty:
        correlation_matrix = numeric_cols.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        st.pyplot(plt)
    else:
        st.write('No numeric columns available for correlation calculation.')

    # Top Tracks Data
    top_tracks = df.groupby('track_name')['streams'].sum().sort_values(ascending=False).head(10)
    st.write("Top Tracks DataFrame:", top_tracks)

    # Visualization of Top Tracks
    if not top_tracks.empty and top_tracks.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
        fig, ax = plt.subplots(figsize=(10, 6))
        top_tracks.plot(kind='bar', ax=ax)
        ax.set_title('Top 10 Tracks by Streams')
        ax.set_ylabel('Total Streams')
        ax.set_xlabel('Track Names')
        st.pyplot(fig)
    else:
        st.write('No valid numeric data to display in the plot.')

    # Pie Chart Visualization
    if st.button('Show Pie Chart Visualizations'):
        delay_per_day = df.groupby('released_day')[['in_spotify_playlists', 'in_apple_playlists']].sum()
        pieChartLabels = ['In Spotify Playlists', 'In Apple Playlists']
        myColors = sns.color_palette('pastel')

        for i in range(1, 8):
            b = delay_per_day.iloc[i-1, :]  # Data for that day
            plt.figure(figsize=(6, 6))
            plt.pie(b, labels=pieChartLabels, colors=myColors, autopct='%.0f%%')
            plt.title('Released Day ' + str(i))
            st.pyplot(plt)

    # Monthly Streams Visualization
    st.write('### Jumlah Streaming per Bulan Rilis')
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_streams = df.groupby('released_month')['streams'].sum().reset_index()
    monthly_streams.set_index('released_month', inplace=True)
    monthly_streams.plot(ax=ax)
    ticks = range(0, len(monthly_streams), 1)
    plt.xticks(ticks, monthly_streams.index, rotation=45)
    plt.title('Monthly Streams')
    plt.xlabel('Released Month')
    plt.ylabel('Total Streams')
    st.pyplot(fig)
