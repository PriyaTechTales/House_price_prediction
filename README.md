# House_price_prediction
"This project focuses on building a machine learning-based web application to predict house prices in different cities, specifically Pune, Bangalore, and Delhi. The application uses Linear, Lasso and Ridge regression models tailored for each city's unique location parameters to provide accurate predictions. This system is designed to assist users in making informed decisions about real estate investments".

# Features
Dataset: Separate datasets were curated for each city from kaggle (Pune, Bangalore, and Delhi), ensuring localized insights for enhanced prediction accuracy.

Machine Learning Model: Lasso and Ridge regression models are employed for feature selection and regularization, minimizing overfitting and improving prediction accuracy for each city.

User-Friendly Interface
Frontend: Built with HTML, CSS, and JavaScript to deliver an interactive and responsive user experience.
Backend: Developed using Django for robust server-side logic and secure data handling.
SQLite Database: Efficiently stores user inputs and model outputs, seamlessly integrating with the web application.

Multi-City Support: Allows users to select a specific city, ensuring that the predictions are tailored to the selected location's housing market dynamics.

Data Visualization: Key insights, such as trends and distribution of house prices, are presented through interactive graphs and charts.

# How It Works
Users provide property details through an intuitive form on the web application.

The application processes the data and passes it to the corresponding trained Lasso or Ridge regression model for the selected city.

The model returns the predicted house price, displayed in a user-friendly format.

Visual aids help users understand the factors influencing the prediction.

# Technology Stack
Frontend: HTML, CSS, JavaScript

Backend: Django

Database: SQLite

Machine Learning: Linear, Lasso and Ridge Regression


# Requirements for the Project
Django Framework Version: 4.x or higher

Reason: Leverages modern Django features like class-based views and enhanced ORM capabilities.

Python Version: 3.10 or higher

Reason: Ensures compatibility with advanced libraries and features.

# Libraries
NumPy: >=1.21.0

For numerical operations in regression models.

pandas: >=1.3.0

For data manipulation and preprocessing.

scikit-learn: >=1.0.0

For implementing Lasso and Ridge regression models.

joblib: >=1.2.0

For saving and loading serialized models.

Deployment Requirements

Gunicorn: >=20.0.0

WhiteNoise: >=5.3.0 (for serving static files in production).

# Steps or Commands to Run the Project
1.Download the project zip file and extract it.

2.Open the extracted folder in VSCode.

3.Open the terminal in VSCode.

4.Navigate to the project directory by running:bash

5.Copy code

6.cd house_price_prediction

7.Start the Django development server by running:bash

8.Copy code

9.python manage.py runserver  

10.Open the provided URL in a browser to access the application.
