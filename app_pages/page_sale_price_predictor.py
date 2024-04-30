import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from src.data_management import (
    load_house_prices_data,
    load_pkl_file,
    load_inherited_house_data)
from src.machine_learning.predictive_analysis_ui import predict_sale_price


def page_sale_price_predictor_body():

    # load predict sale price files
    vsn = 'v2'
    sale_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/regression_pipeline.pkl"
    )
    sale_price_features = (
        pd.read_csv(
            f"outputs/ml_pipeline/predict_sale_price/{vsn}/X_train.csv")
        .columns
        .to_list()
    )

    st.write("### House Sale Price Predictor")
    st.success(
        f"The client is interested in predicting"
        f" the potential sale prices"
        f" for properties in Ames, Iowa, and specifically to"
        f" determine a potential value for the sum of the sales "
        f" prices of four inherited properties (business requirement 2).\n"
    )

    st.info(
        f"The price prediction will be based on six "
        f" features of the property, which can be input"
        f" using the widgets below. These features were identified by"
        f" the machine learning model as the most important features "
        f"to predict sale  price."
    )

    st.write("---")

    st.write(
        f"* Information on categorical feature: "
        f"Ex: Excellent, Gd: Good, TA: Typical/Average, Fa: Fair\n\n"
    )
    
    st.write("---")


#     # Generate Live Data
    X_live = DrawInputsWidgets()

#     # predict on live data
    if st.button("Run Predictive Analysis"):
        predict_sale_price(X_live, sale_price_features, sale_price_pipe)

    st.write("---")

    st.write("### Price prediction for the clients inherited properties:")
    in_df = load_inherited_house_data()
    in_df = in_df.filter(sale_price_features)

    st.write("* Features of Inherited Homes")
    st.write(in_df)

    if st.button("Run Prediction on Inherited Homes"):
        inherited_price_prediction = predict_sale_price(
            in_df, sale_price_features, sale_price_pipe)
        total_value = inherited_price_prediction.sum()
        total_value = float(total_value.round(1))
        total_value = '${:,.2f}'.format(total_value)

        st.write(f"* The total value of the inherited homes is estimated"
                 f" to be:")
        st.write(f"**{total_value}**")


def DrawInputsWidgets():

    # load dataset
    df = load_house_prices_data()
    percentageMin, percentageMax = 0.4, 2.0

    # we create input widgets for the 4 best features
    col1, col2, col3, col4 = st.beta_columns(4)
    col5, col6, col7, col8 = st.beta_columns(4)

    # We are using these features to feed the ML pipeline -
    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    with col1:
        feature = "1stFlrSF"
        st_widget = st.number_input(
            label='1st Floor Area SQFT',
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with col2:
        feature = "GarageArea"
        st_widget = st.number_input(
            label='GarageArea SQFT',
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=20
        )
    X_live[feature] = st_widget

    with col3:
        feature = "GrLivArea"
        st_widget = st.number_input(
            label='Above Grade Living Area SQFT',
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=20
        )
    X_live[feature] = st_widget

    with col4:
        feature = "KitchenQual"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col5:
        feature = "YearBuilt"
        st_widget = st.number_input(
            label="Year Built",
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with col6:
        feature = "YearRemodAdd"
        st_widget = st.number_input(
            label="Year Redmodelling Added",
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    return X_live
