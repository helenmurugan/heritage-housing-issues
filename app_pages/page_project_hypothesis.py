import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    st.success(
        f"**Property Size Hypothesis**\n\n"
        f"* We hypothesise that features relating to the size of a property "
        f"are positively correlated to sale price.\n\n"
        f"**Accept hypothesis:** Features related to property size are "
        f"positively correlated to "
        f"sale price as demonstrated by the heatmap analyses and data "
        f"visualisations. \n\n"
        f"* '1stFlrSF', 'GarageArea', 'GrLivArea' and "
        f"'TotalBsmtSF' have a "
        f"strong positive correlation with SalePrice.\n\n"
        f"* '2ndFlrSF' and 'BedroomAbvGr' have a "
        f"weak positive correlation with SalePrice."
    )

    st.success(
        f"**Year Built Hypothesis**\n\n"
        f"* We hypothesise that the year of build has a positive"
        f" correlation with sale price.\n\n"
        f"**Accept hypothesis:** The year of build has a moderate positive"
        f" correlation to "
        f"sale price. This is demonstrated by data visualisations including"
        f" correlation "
        f"heatmaps and a scatter plot.  \n\n"
    )

    st.info(
        f"**Lot Size Hypothesis**\n\n"
        f"* We hypothesise that the lot size has a strong positive"
        f" correlation with sale price.\n\n"
        f"**Reject hypothesis:** 'LotFrontage' and 'LotArea' are only weakly "
        f" correlated with sale price. \n\n"
        f"This may be an interesting insight, as it challenges "
        f"preconceptions about which features most strongly affect "
        f"sale price.\n\n"
    )

    st.success(
        f"**Sale Price Prediction Hypothesis**\n\n"
        f"* We hypothesise that we are able to predict sale prices with an R2"
        f" value of at least 0.75, based on important features that have been"
        f" identified through machine learning. \n\n"
        f"**Accept hypothesis:** R2 scores of 0.943 and 0.825 have been"
        f" achieved for the train set and test set respectively"
        f" through ML modelling and evaluation."
    )
