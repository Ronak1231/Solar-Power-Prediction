# Solar Power Prediction System

## Overview
The **Solar Power Prediction System** is a **machine learning-based web application** built with **Streamlit**. It predicts **solar power generation** using **Linear Regression, Decision Tree, and Random Forest models**.

## Project Structure
```
├── solar_power_generation.csv                 # Dataset file
├── Week 1
│   ├── Solar_Power_Prediction.ipynb           # Jupyter Notebook for week 1
├── Week 2
│   ├── Solar_Power_Prediction_Week 2.ipynb    # Jupyter Notebook for week 2
├── requirements.txt                           # List of dependencies
├── README.md                                  # Project documentation
```

## Features
- **User Authentication**: Secure login/register with SQLite database.
- **Data Preprocessing**: Handles missing values and selects key features.
- **Machine Learning Models**: Trains and evaluates three regression models.
- **Model Performance Evaluation**: Uses **MAE** and **R² Score**.
- **Data Visualization**: Heatmaps, scatter plots, and feature distributions.
- **Real-time Predictions**: Users enter feature values to predict solar power output.

## Installation
1. Clone the repository:  
   ```sh
   git clone https://github.com/your-repo/solar-power-prediction.git
   cd solar-power-prediction
   ```
2. Install dependencies:  
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:  
   ```sh
   streamlit run app.py
   ```

## Technologies Used
- **Python**
- **Streamlit** (Web App Interface)
- **SQLite** (User Authentication)
- **Pandas, NumPy** (Data Processing)
- **Scikit-Learn** (Machine Learning)
- **Matplotlib, Seaborn** (Data Visualization)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
Developed by **[Your Name]**.