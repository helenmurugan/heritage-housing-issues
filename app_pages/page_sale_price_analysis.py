import plotly.express as px
import numpy as np
import streamlit as st
from src.data_management import load_clean_house_prices_data
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
from sklearn.preprocessing import LabelEncoder
import pandas as pd
sns.set_style("whitegrid")


def page_sale_price_analysis_body():

    # load data
    df = load_clean_house_prices_data()

    # define variables most strongly correlated with Sale Price
    vars_to_study = [
        '1stFlrSF', 'GarageArea', 'GrLivArea', 'OverallQual',
        'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd'
    ]

    st.write("### Property Sale Price Analysis")
    st.success(
        f"The client is interested in "
        f"understanding the correlations "
        f" between property attributes and sale price."
        f" Therefore, the client expects data visualisations which demonstrate"
        f" the relationships between relevant attributes and sale price."
        f" (Business Requirement 1). \n"
    )

# inspect data
    if st.checkbox("Inspect Dataset"):
        st.write(
            f"* The dataset used for this correlation study has "
            f"{df.shape[0]} rows and {df.shape[1]} columns.\n\n"
            f"* The dataframe below shows the first 10 observations"
            f" in the dataset.\n\n"
            f"* Data cleaning has been perfomed on this dataset to improve"
            f" data quality, leading to more useful data visualisations.")
        st.write(df.head(10))
        st.write(
            f"**Definition of Property Variables**\n\n"
            f"1stFlrSF - First Floor square feet \n\n"
            f"2ndFlrSF - Second-floor square feet \n\n"
            f"BedroomAbvGr - Bedrooms above grade \n\n"
            f"BsmtExposure - Refers to walkout or garden level"
            f" walls"
            f" (Gd: Good Exposure; Av: Average Exposure; Mn: Minimum"
            f" Exposure; No: No Exposure; None: No Basement) \n\n"
            f"BsmtFinType1 - Rating of basement finished area"
            f"(GLQ: Good Living Quarters; ALQ: Average Living Quarters; "
            f"BLQ: Below Average Living Quarters; Rec: Average Rec Room; "
            f"LwQ: Low Quality; Unf: Unfinshed; None: No Basement)\n\n"
            f"BsmtFinSF1 - Type 1 finished square feet \n\n"
            f"BsmtUnfSF - Unfinished square feet of basement area \n\n"
            f"TotalBsmtSF - Total square feet of basement area \n\n"
            f"GarageArea - Size of garage in square feet \n\n"
            f"GarageFinish - Interior finish of the garage (Fin: "
            f"Finished; RFn: "
            f"Rough Finished; Unf: Unfinished; None: No Garage) \n\n"
            f"GrLivArea - Above grade (ground) living area square feet \n\n"
            f"KitchenQual - Kitchen quality (Ex: Excellent; Gd: Good; TA: "
            f"Typical/Average; Fa: Fair; Po: Poor) \n\n"
            f"LotArea - Lot size in square feet \n\n"
            f"LotFrontage - Linear feet of street connected to property \n\n"
            f"MasVnrArea - Masonry veneer area in square feet \n\n"
            f"OpenPorchSF - Open porch area in square feet \n\n"
            f"OverallCond - Rates the overall condition of the house "
            f"(10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; "
            f"6: Above Average; 5: Average; 4: Below Average; 3: Fair; "
            f"2: Poor; 1: Very Poor) \n\n"
            f"OverallQual - Rates the overall material and finish of "
            f"the house (10: Very Excellent; 9: Excellent; 8: Very Good; "
            f"7: Good; 6: Above Average; 5: Average; 4: Below Average; "
            f"3: Fair; 2: Poor; 1: Very Poor) \n\n"
            f"YearBuilt - Original construction date \n\n"
            f"YearRemodAdd - Remodel date (same as construction date"
            f" if no remodelling or additions)\n\n"
        )

    st.write("---")

    st.write("### Correlation Study")

    # Correlation Study Summary
    st.write(
        f"The correlation study showed that the following features"
        f" are most strongly"
        f" correlated with sale price: "
        f"\n\n"
        f"**{vars_to_study}**\n\n"
        f"\n\n"
        f"These attributes have correlation coefficients greater"
        f" than 0.5, indicating strong positive linear relationships.\n"
        f"\n\n"
        f"Data visualisations have been generated to demonstrate the "
        f"relationships between attributes and provide insights on which"
        f"house attributes are most important for predicting sale price."
        f" Visualisations generated in this study include correlation"
        f"heatmaps, bar plots, pie charts and scatter graphs."
    )

    st.info(
        f"***Pearson Correlation Study*** \n\n"
        f" The Pearson correlation is a measure of the linear "
        f"relationship between two continuous variables. It "
        f"quantifies the strength and direction of the linear "
        f"association. "
        f"A value close to 1 or -1 suggests"
        f" a strong linear relationship, while a value close "
        f"to 0 suggests weak or no linear relationship."
        )

    if st.checkbox("Pearson Correlation"):

        # calculate correlations
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)

        # display heatmap
        heatmap_corr(
            df=df_corr_pearson, threshold=0.3,
            figsize=(12, 10), font_annot=10
            )

        st.info(
            f"The heatmap shows that OverallQual has the highest"
            f" Pearson correlation with sale price. A threshold of 0.3 has"
            f" been set, so that only moderate and strong relationships"
            f" are displayed."
        )
        st.write("")

        # display bar plot
        display_pearson_corr_bar(df)

        st.info(
            f"The bar plot shows the order of importance for the variables"
            f" with the strongest Pearson correlation to sale price."
            f" The focus should be on these attributes when estimating"
            f" property sale prices."
            )
        st.write("")

        # display pie chart
        display_pearson_corr_pie(df)
        st.info(
            f"The pie chart shows the relative importance of all variables"
            f" for predicting sale price. Certain features may be"
            f" disregarded if they have very little importance for"
            f" predicting sale prices."
            )

    st.info(
        f"***Spearman Correlation Study*** \n\n"
        f"The Spearman correlation is a measure of the monotonic "
        f"relationship between two continuous variables, "
        f"that is a relationship where the variables behave similarly"
        f" but not necessarily linearly. "
        f"It quantifies the strength and direction of the "
        f"association. "
        f"A value close to 1 or -1 suggests a strong monotonic "
        f"relationship, while a value close to 0 suggests weak "
        f"or no monotonic relationship."
        )

    if st.checkbox("Spearman Correlation"):

        # calculate correlations
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)

        # display heatmap
        heatmap_corr(
            df=df_corr_spearman, threshold=0.3,
            figsize=(12, 10), font_annot=10
            )

        st.info(
            f"The heatmap shows that OverallQual has the highest"
            f" Spearman correlation with sale price. A threshold of 0.3 has "
            f"been set, so that only moderate and strong relationships are"
            f" displayed."
            )
        st.write("")

        # display bar plot
        display_spearman_corr_bar(df)

        st.info(
            f"The bar plot shows the order of importance for the variables"
            f" with the strongest Spearman correlation to sale price."
            f" The focus should beon these attributes when estimating property"
            f" sale prices. The most important variables are the same for both"
            f" Pearson and Spearman methods, although the order of importance"
            f" is different."
            )
        st.write("")

        # display pie chart
        display_spearman_corr_pie(df)
        st.info(
            f"The pie chart shows the relative importance of all variables for"
            f" predicting sale price. Certain features may be disregarded"
            f" if they have very little importance for predicting sale prices."
            f" Comparison of the pie charts in the Pearson and Spearman"
            f" studies show mostly similarites, however, the order of some"
            f" attributes are different."
            )

    st.info(
        f"*** PPS Correlation Heatmap*** \n\n"
        f"The Predictive Power Score (PPS) heatmap visualises"
        f" the relationship between two variables, capturing"
        f" both linear and non-linear associations. Unlike "
        f"Pearson or Spearman correlations, PPS detects any "
        f"type of predictive relationship."
        )

    if st.checkbox("Predictive Power Score (PPS) Correlation"):

        # calculate correlations
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)

        # display heatmap
        heatmap_pps(
            df=pps_matrix, threshold=0.3,
            figsize=(12, 10), font_annot=10
            )

        st.info(
            f"The heatmap shows that OverallQual has the highest predictive"
            f" power for sale price. A threshold of 0.3 has been set,"
            f" so that only moderate and strong relationships are displayed."
            )
        st.write("")

    st.info(
        f"Scatter plots showing the house attributes that are most strongly"
        f" correlated to sale price are displayed below. These plots"
        f" visually demonstrate the variance in data for each attribute."
        f" The following variables all have a strong positive "
        f"correlation with sale price:\n\n"
        f"* First floor area in square feet\n\n"
        f"* Garage area in square feet\n\n"
        f"* Garage year built\n\n"
        f"* Above grade (ground) living area in square feet\n\n"
        f"* Overall quality of materials and finishes\n\n"
        f"* Total basement area in square feet\n\n"
        f"* Original construction date\n\n"
        f"* Year of remodelling (or build if it has not been remodelled)\n\n"
    )

    # Correlation plots adapted from the Data Visualisations Notebook
    if st.checkbox("Scatter Plots of Important Features vs Sale Price"):

        # create new df with variables of interest
        df_eda = df.filter(vars_to_study + ['SalePrice'])

        # display scatter plots
        plot_numerical(df_eda, vars_to_study)


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig, axes = plt.subplots(figsize=figsize)
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis', annot_kws={"size": font_annot},
                    ax=axes, linewidth=0.5
                    )
        axes.set_yticklabels(df.columns, rotation=0)
        plt.title(
            "Correlation Heatmap",
            fontsize=15, y=1.05
        )
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[abs(df) < threshold] = True
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                         mask=mask, cmap='rocket_r',
                         annot_kws={"size": font_annot}, linewidth=0.05,
                         linecolor='grey'
                         )
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.title(
            "Correlation Heatmap: PPS",
            fontsize=15, y=1.05
        )
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def CalculateCorrAndPPS(df):
    df_corr_spearman = df.corr(method="spearman")
    df_corr_pearson = df.corr(method="pearson")

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(
        ['x', 'y', 'ppscore']).pivot(columns='x', index='y', values='ppscore')
    pps_score_stats = pps_matrix_raw.query("ppscore < 1").filter(['ppscore']).describe().T  # noqa

    return df_corr_pearson, df_corr_spearman, pps_matrix


def display_pearson_corr_bar(df):
    """ Calcuate and display Pearson Correlation """
    corr_pearson = df.corr(method='pearson')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    axes = plt.bar(x=corr_pearson[:7].index, height=corr_pearson[:7])
    plt.title(
        "Top Attributes Correlated with Sale Price",
        fontsize=8, y=1.05
        )
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Pearson Coefficient", fontsize=8)
    plt.yticks(fontsize=8)
    st.pyplot(fig)


def display_spearman_corr_bar(df):
    """ Calcuate and display Spearman Correlation """
    corr_spearman = df.corr(method='spearman')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    axes = plt.bar(x=corr_spearman[:7].index, height=corr_spearman[:7])
    plt.title(
        "Top Attributes Correlated with Sale Price",
        fontsize=8, y=1.05
        )
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Spearman Coefficient", fontsize=8)
    plt.yticks(fontsize=8)
    st.pyplot(fig)


def display_spearman_corr_pie(df):
    """
    Calculate and display Pie Chart based on
    Spearman correlation coefficients
    """
    corr_spearman_for_pie = (
        df.corr(method='spearman')['SalePrice']
        .sort_values(key=abs, ascending=False)[1:]
    )

    spearman_data = corr_spearman_for_pie.reset_index()
    spearman_data.columns = ['Feature', 'Correlation']
    df_corr_spearman = pd.DataFrame(spearman_data)
    df_corr_spearman['Correlation'] = df_corr_spearman['Correlation'].abs()
    df_corr_spearman['Normalized_Correlation'] = (
        df_corr_spearman['Correlation'] / df_corr_spearman['Correlation'].sum()
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        df_corr_spearman['Normalized_Correlation'],
        labels=df_corr_spearman['Feature'],
        autopct='%1.1f%%'
    )
    ax.set_title(
        'Importance of House Attributes for Predicting Sale Price',
        fontsize=11,
        y=1.05
    )

    st.pyplot(fig)


def display_pearson_corr_pie(df):
    """
    Calculate and display Pie Chart based on
    Pearson correlation coefficients
    """
    corr_pearson_for_pie = (
        df.corr(method='pearson')['SalePrice']
        .sort_values(key=abs, ascending=False)[1:]
    )

    pearson_data = corr_pearson_for_pie.reset_index()
    pearson_data.columns = ['Feature', 'Correlation']
    df_corr_pearson = pd.DataFrame(pearson_data)
    df_corr_pearson['Correlation'] = df_corr_pearson['Correlation'].abs()
    df_corr_pearson['Normalized_Correlation'] = (
        df_corr_pearson['Correlation'] / df_corr_pearson['Correlation'].sum()
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        df_corr_pearson['Normalized_Correlation'],
        labels=df_corr_pearson['Feature'],
        autopct='%1.1f%%'
    )
    ax.set_title(
        'Importance of House Attributes for Predicting Sale Price',
        fontsize=11,
        y=1.05
    )

    st.pyplot(fig)


def plot_numerical(df_eda, vars_to_study):
    """scatterplots of variables vs SalePrice """
    target_var = 'SalePrice'
    for col in vars_to_study:
        fig, axes = plt.subplots(figsize=(8, 5))
        axes = sns.scatterplot(data=df_eda, x=col, y=target_var)
        plt.title(f"{col}", fontsize=20, y=1.05)
        st.pyplot(fig)
