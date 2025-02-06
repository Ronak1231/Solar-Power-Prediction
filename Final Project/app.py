import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import sqlite3

# Database setup
def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    finally:
        conn.close()

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('C:/Users/91982/Documents/AICTE/AICTE SHELL_Solar Power Prediction/full code/solar_power_generation.csv')
    if data.isnull().sum().sum() > 0:
        data = data.fillna(data.median())
    return data

# Feature selection
def select_features(data):
    X = data.drop(columns=['generated_power_kw'])
    y = data['generated_power_kw']
    selector = SelectKBest(score_func=mutual_info_regression, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X[selected_features], y, selected_features

# Model training and evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    
    # Decision Tree Regression
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_train, y_train)
    dtr_preds = dtr.predict(X_test)
    
    # Random Forest Regression
    rfr = RandomForestRegressor(random_state=42, n_estimators=100)
    rfr.fit(X_train, y_train)
    rfr_preds = rfr.predict(X_test)
    
    return lr, dtr, rfr, lr_preds, dtr_preds, rfr_preds

# Updated evaluate_model function with MAE
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "R^2": r2}

# Streamlit app
def main():
    st.title("Solar Power Generation Prediction")
    
    # User authentication
    st.sidebar.title("User  Authentication")
    choice = st.sidebar.selectbox("Login/Register", ["Login", "Register"])
    
    if choice == "Register":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Register"):
            create_users_table()
            add_user(username, password)
            st.sidebar.success("Registration successful! Please login.")
    
    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if check_user(username, password):
                st.sidebar.success("Logged in as {}".format(username))
                st.session_state.logged_in = True
                st.session_state.username = username
            else:
                st.sidebar.error("Invalid username or password")
    
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        data = load_data()
        X, y, selected_features = select_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and fit the scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models and store them in session state
        if 'models' not in st.session_state:
            st.session_state.models = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        lr, dtr, rfr, lr_preds, dtr_preds, rfr_preds = st.session_state.models
        
        # Navigation sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Choose a page", ["Visualization", "Model Performance", "Prediction"])
        
        if page == "Visualization":
            st.header("Data Visualization")
            st.subheader("Select Visualizations to Display")
            show_correlation_heatmap = st.checkbox("Show Correlation Heatmap")
            show_distribution_power = st.checkbox("Show Distribution of Generated Power")
            show_feature_vs_power = st.checkbox("Show Feature vs Generated Power")
            show_pairplot = st.checkbox("Show Pairplot of Selected Features")
            show_distribution_plots = st.checkbox("Show Distribution Plots for Selected Features")

            if show_correlation_heatmap:
                plt.figure(figsize=(10, 6))
                sns.heatmap(X.join(y).corr(), annot=True, cmap="coolwarm")
                st.pyplot(plt)

            if show_distribution_power:
                plt.figure(figsize=(10, 6))
                sns.histplot(y, bins=30, kde=True)
                st.pyplot(plt)

            if show_feature_vs_power:
                for feature in selected_features:
                    if feature in X.columns:
                        plt.figure(figsize=(10, 6))
                        sampled_data = X.join(y).sample(frac=0.1, random_state=42)
                        sns.scatterplot(x=sampled_data[feature], y=sampled_data['generated_power_kw'], hue=sampled_data[selected_features[0]], palette='cool')
                        plt.title(f"{feature} vs Generated Power by {selected_features[0]}")
                        st.pyplot(plt)

            if show_pairplot:
                st.subheader("Pairplot of Selected Features")
                sns.pairplot(X.join(y).sample(frac=0.1, random_state=42), diag_kind='kde')
                st.pyplot(plt)

            if show_distribution_plots:
                # Distplot for all selected features
                for feature in selected_features:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(X[feature], color='skyblue', kde=True)
                    plt.xlabel(feature)
                    plt.ylabel('Density')
                    plt.title(f'Distribution of {feature}')
                    st.pyplot(plt)

        elif page == "Model Performance":
            st.header("Model Performance Metrics")
            metrics = []
            metrics.append(evaluate_model(y_test, lr_preds, "Linear Regression"))
            metrics.append(evaluate_model(y_test, dtr_preds, "Decision Tree Regression"))
            metrics.append(evaluate_model(y_test, rfr_preds, "Random Forest Regression"))
            metrics_df = pd.DataFrame(metrics)
            st.table(metrics_df)

        elif page == "Prediction":
            st.header("Make a Prediction")
            input_data = {}
            for feature in selected_features:
                input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
            
            # Model selection dropdown
            model_choice = st.selectbox("Select Model for Prediction", ["Linear Regression", "Decision Tree", "Random Forest"])

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])  # Convert to DataFrame
                input_scaled = scaler.transform(input_df)  # Scale the input data
                if model_choice == "Linear Regression":
                    prediction = lr.predict(input_scaled)
                elif model_choice == "Decision Tree":
                    prediction = dtr.predict(input_scaled)
                elif model_choice == "Random Forest":
                    prediction = rfr.predict(input_scaled)

                st.success(f"Predicted Power: {prediction[0]:.2f} kW")

if __name__ == "__main__":
    main()