# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title
st.title("🎓 Student Marks Performance Predictor")

# Input
st.subheader("Enter your study hours")
hours = st.number_input("Hours Studied", min_value=0.0, max_value=10.0, step=0.5)

# Sample Dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Scores": [10, 20, 30, 35, 45, 55, 65, 70, 85, 95]
}
df = pd.DataFrame(data)

# Model Training
X = df[["Hours"]]
y = df["Scores"]
model = LinearRegression()
model.fit(X, y)

# Prediction
if st.button("Predict Marks"):
    predicted_score = model.predict([[hours]])[0]
    st.success(f"📚 Predicted Score: {predicted_score:.2f}")

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(df["Hours"], df["Scores"], color='blue', label="Data Points")
    ax.plot(df["Hours"], model.predict(X), color='red', label="Regression Line")
    ax.scatter(hours, predicted_score, color='green', label="Your Prediction", s=100)
    ax.set_xlabel("Hours Studied")
    ax.set_ylabel("Marks Scored")
    ax.legend()
    st.pyplot(fig)
