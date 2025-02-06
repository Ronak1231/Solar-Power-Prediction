# ☀️🔋 Solar Power Prediction System - End-to-End ML Application

This project implements an **end-to-end solar power prediction system** using **Streamlit** for the user interface and **Scikit-Learn** for machine learning. It predicts **solar power generation** based on historical meteorological data.

---

## ✅ Features

- **User Authentication**: Secure login and registration using SQLite.
- **Data Preprocessing**: Cleans missing values and selects key features.
- **Machine Learning Models**: Implements **Linear Regression, Decision Tree, and Random Forest**.
- **Performance Evaluation**: Uses **Mean Absolute Error (MAE)** and **R² Score**.
- **Data Visualization**: Heatmaps, scatter plots, and feature distributions.
- **Real-time Predictions**: Users can input values to get power predictions.

---

## 📜 Prerequisites

Ensure the following are installed:

1. **Python 3.9 or above**  
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Required Libraries** (Install from `requirements.txt`):
   ```sh
   pip install -r requirements.txt
   ```
3. **Dataset**: Ensure `solar_power_generation.csv` is in the project directory.

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```sh
git clone <repository-url>
cd solar-power-prediction
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```sh
streamlit run app.py
```

---

## 🗃️ File Structure

```
solar-power-prediction/
│
├── solar_power_generation.csv                 # Dataset file
├── Week 1
│   ├── Solar_Power_Prediction.ipynb           # Jupyter Notebook for week 1
├── Week 2
│   ├── Solar_Power_Prediction_Week 2.ipynb    # Jupyter Notebook for week 2
├── Final Submission
│   ├── app.py                                  # Main code 
│   ├── requirements.txt                        # install this requirements
│   ├── solar_power_generation.csv              # Dataset file
│   ├── users.db                                # Stores user login data


├── requirements.txt                           # List of dependencies
├── README.md                                  # Project documentation
```

---

## 🤷 How It Works

### Backend Flow

1. **Data Loading**: Reads and processes dataset, handling missing values.
2. **Feature Selection**: Identifies the most relevant features.
3. **Model Training**: Trains **Linear Regression, Decision Tree, and Random Forest** models.
4. **Model Evaluation**: Compares model performance using MAE and R² score.
5. **Prediction**: Generates power output predictions based on user input.

### Frontend Flow

- Users **log in** and interact via **Streamlit UI**.
- The system processes inputs and **visualizes data trends**.
- Users input feature values and receive **real-time power predictions**.

---

## 🤖 Technologies Used

- **Python**: Programming language for data processing and ML.
- **Streamlit**: Web-based interface for interaction.
- **SQLite**: Database for user authentication.
- **Scikit-Learn**: ML models and data preprocessing.
- **Matplotlib & Seaborn**: Data visualization tools.

---

## 🚚 Deployment

This project can be deployed on **AWS, Google Cloud, or Heroku**. Ensure API keys and environment variables are configured properly.

---

## 🔜 Future Improvements

1. Add real-time weather data integration.
2. Implement deep learning models for better accuracy.
3. Develop a mobile-friendly version.

---

## 🤝 Acknowledgments

- [Scikit-Learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Matplotlib](https://matplotlib.org/)
- [SQLite](https://www.sqlite.org/)

---

## ✍️ Author  
[Ronak Bansal]

---

## 🙌 Contributing  
Feel free to fork this repository, improve it, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🐛 Troubleshooting  
If you encounter issues, create an issue in this repository.

---

## 📧 Contact  
For inquiries or support, contact [ronakbansal12345@gmail.com].
