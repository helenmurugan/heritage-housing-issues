from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import pandas as pd

# code taken from "Modelling_and_Evaluation_v2" notebook

def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    st.write("**Train Set**")
    regression_evaluation(X_train, y_train, pipeline)
    st.write("**Test Set**")
    regression_evaluation(X_test, y_test, pipeline)


def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    st.write(f"R2 Score: {r2_score(y, prediction).round(3)}")
    st.write(
        f"Mean Absolute Error: {mean_absolute_error(y, prediction).round(3)}")
    st.write(
        f"Mean Squared Error: {mean_squared_error(y, prediction).round(3)}")
    st.write(
        f"Root Mean Squared Error: "
        f" {np.sqrt(mean_squared_error(y, prediction)).round(3)}")
    st.write("\n")


def regression_evaluation_plots(X_train, y_train, X_test, y_test,
                                pipeline, alpha_scatter):
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_title("Train Set")

    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].set_title("Test Set")

    st.pyplot(fig)