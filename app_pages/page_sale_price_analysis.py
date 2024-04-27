import plotly.express as px
import numpy as np
import streamlit as st
from src.data_management import load_house_prices_data
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
from sklearn.preprocessing import LabelEncoder
import pandas as pd
sns.set_style("whitegrid")


def page_sale_price_analysis_body():

    # load data
    df = load_house_prices_data()
    # The variables most strongly correlated with Sale Price
    vars_to_study = ['1stFlrSF',
                    'GarageArea',
                    'GarageYrBlt',
                    'GrLivArea',
                    'KitchenQual_encoded',
                    'OverallQual',
                    'TotalBsmtSF',
                    'YearBuilt',
                    'YearRemodAdd'] 

    st.write("### Property Sale Price Analysis")
    st.success(
        f"Business requirement 1 - The client is interested in "
        f"understanding the correlation "
        f" between property attributes and sale price."
        f" Therefore, the client expects data visualization of the correlated"
        f" variables against the sale price. "
        f" \n"
    )

#     # inspect data
    if st.checkbox("Inspect Sale Price Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns\n\n"
            f"* The dataframe below shows the first 10 observations"
            f" in the dataset")
        st.write(df.head(10))
        st.write(
            f"**Definition of Property Variables**\n\n"
            f"1stFlrSF - First Floor square feet \n\n"
            f"2ndFlrSF - Second-floor square feet \n\n"
            f"BedroomAbvGr - Bedrooms above grade (does NOT include basement bedrooms) \n\n"
            f"BsmtExposure - Refers to walkout or garden level walls"
            f" (Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; "
            f"No: No Exposure; None: No Basement) \n\n"
            f"BsmtFinType1 - Rating of basement finished area"
            f"(GLQ: Good Living Quarters; ALQ: Average Living Quarters; "
            f"BLQ: Below Average Living Quarters; Rec: Average Rec Room; "
            f"LwQ: Low Quality; Unf: Unfinshed; None: No Basement)\n\n"
            f"BsmtFinSF1 - Type 1 finished square feet \n\n"
            f"BsmtUnfSF - Unfinished square feet of basement area \n\n"
            f"TotalBsmtSF - Total square feet of basement area \n\n"
            f"GarageArea - Size of garage in square feet \n\n"
            f"GarageFinish - Interior finish of the garage (Fin: Finished; RFn: "
            f"Rough Finished; Unf: Unfinished; None: No Garage) \n\n"
            f"GarageYrBlt - Year garage was built \n\n"
            f"GrLivArea - Above grade (ground) living area square feet \n\n"
            f"KitchenQual - Kitchen quality (Ex: Excellent; Gd: Good; TA: "
            f"Typical/Average; Fa: Fair; Po: Poor) \n\n"
            f"LotArea - Lot size in square feet \n\n"
            f"LotFrontage - Linear feet of street connected to property \n\n"
            f"MasVnrArea - Masonry veneer area in square feet \n\n"
            f"EnclosedPorch - Enclosed porch area in square feet \n\n"
            f"OpenPorchSF - Open porch area in square feet \n\n"
            f"OverallCond - Rates the overall condition of the house "
            f"(10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; "
            f"6: Above Average; 5: Average; 4: Below Average; 3: Fair; "
            f"2: Poor; 1: Very Poor) \n\n"
            f"OverallQual - Rates the overall material and finish of "
            f"the house (10: Very Excellent; 9: Excellent; 8: Very Good; "
            f"7: Good; 6: Above Average; 5: Average; 4: Below Average; "
            f"3: Fair; 2: Poor; 1: Very Poor) \n\n"
            f"WoodDeckSF - Wood deck area in square feet \n\n"
            f"YearBuilt - Original construction date \n\n"
            f"YearRemodAdd - Remodel date (same as construction date"
            f" if no remodelling or additions)\n\n"
          
        )

    st.write("---")

    st.write("### Correlation Study")
    # Correlation Study Summary
    st.write(
        f"A correlation study was performed to gain insights on how "
        f"the house attributes are correlated to Sale Price. \n\n"
        f"The correlation study showed that thefollowing features are most strongly"
        f" correlated with the Sale Price: "
        f"**{vars_to_study}**\n\n"
        f"The relationships between attributes are displayed visually using"
        f" Pearson and Spearman heatmaps, predictive power score (PPS) heatmap,"
        f" bar plots and scatter plots. "
    )

    st.info(
        f"*** Heatmap and Barplot: Pearson Correlation *** \n\n"
        f" The Pearson correlation is a measure of the linear "
        f"relationship between two continuous variables. It "
        f"quantifies the strength and direction of the linear "
        f"association. "
        f"A value close to 1 or -1 suggests"
        f" a strong linear relationship, while a value close "
        f"to 0 suggests weak or no linear relationship."
        f"The five attributes that are most strongly correlated with Sale Price "
        f"are displayed in a bar plot.")

    if st.checkbox("Pearson Correlation"):

        # encode categorical variables
        label_encoder = LabelEncoder()
        encoded_df = pd.DataFrame()

        for col in df.columns:
            if df[col].dtype == 'object':
                encoded_col = label_encoder.fit_transform(df[col])
                encoded_df[col+'_encoded'] = encoded_col
            else:
                encoded_df[col] = df[col]
        df = encoded_df

        # calculate correlations
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)

        #display heatmap
        heatmap_corr(df=df_corr_pearson, 
                    threshold=0.4, 
                    figsize=(12,10), 
                    font_annot=10)

        #display bar plot
        display_pearson_corr_bar(df)

    st.info(
        f"*** Heatmap and Barplot: Spearman Correlation *** \n\n"
        f"The Spearman correlation is a measure of the monotonic "
        f"relationship between two continuous or variables, "
        f"that is a relationship where the variables behave similarly" 
        f"but not necessarily linearly. "
        f"It quantifies the strength and direction of the "
        f"association. "
        f"A value close to 1 or -1 suggests a strong monotonic "
        f"relationship, while a value close to 0 suggests weak "
        f"or no monotonic relationship. "
        f"The five attributes that are most strongly correlated "
        f"with Sale Price are displayed in a bar plot."
        )

    if st.checkbox("Spearman Correlation"):

        # encode categorical variables
        label_encoder = LabelEncoder()
        encoded_df = pd.DataFrame()

        for col in df.columns:
            if df[col].dtype == 'object':
                encoded_col = label_encoder.fit_transform(df[col])
                encoded_df[col+'_encoded'] = encoded_col
            else:
                encoded_df[col] = df[col]
        df = encoded_df

        # calculate correlations
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)

        #display heatmap
        heatmap_corr(df=df_corr_spearman, 
                    threshold=0.4, 
                    figsize=(12,10), 
                    font_annot=10)

        #display bar plot
        display_spearman_corr_bar(df)

    st.info(
        f"*** Heatmap: PPS Correlation *** \n\n"
        f"The Power Predictive Score (PPS) heatmap visualizes"
        f" the relationship between two variables, capturing"
        f" both linear and non-linear associations. Unlike "
        f"Pearson or Spearman correlation, PPS detects any "
        f"type of predictive relationship."
        )

    if st.checkbox("Power Predictive Score (PPS) Correlation"):

        # encode categorical variables
        label_encoder = LabelEncoder()
        encoded_df = pd.DataFrame()

        for col in df.columns:
            if df[col].dtype == 'object':
                encoded_col = label_encoder.fit_transform(df[col])
                encoded_df[col+'_encoded'] = encoded_col
            else:
                encoded_df[col] = df[col]
        df = encoded_df

        # calculate correlations
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)

        #display heatmap
        heatmap_pps(df=pps_matrix, 
                    threshold=0.4, 
                    figsize=(12,10), 
                    font_annot=10)
        

    

        # DisplayCorrAndPPS(df_corr_pearson = df_corr_pearson,
        #                 df_corr_spearman = df_corr_spearman, 
        #                 pps_matrix = pps_matrix,
        #                 CorrThreshold = 0.4, PPS_Threshold =0.4,
        #                 figsize=(12,10), font_annot=10)

    # if st.checkbox("Pearson Correlation"):
    #     calc_display_pearson_corr_heat(df)
    #     calc_display_pearson_corr_bar(df)

# #     st.info(
# #         f"*** Heatmap and Barplot: Spearman Correlation ***  \n\n"
# #         f"The Spearman correlation evaluates monotonic relationship, "
# #         f"that is a relationship "
# #         f"where the variables behave similarly but not necessarily linearly.\n"
# #         f" As with the Pearson heatmap, the last line shows the variables"
# #         f" on the x-axis, that have a correlation of 0.6 or more with"
# #         f" the Sale Price. These are then also presented on a barplot"
# #         f" for simplicity.")

# #     if st.checkbox("Spearman Correlation"):
# #         calc_display_spearman_corr_heat(df)
# #         calc_display_spearman_corr_bar(df)

# #     st.info(
# #         f"*** Correlation Histogram- and Scatterplots *** \n\n"
# #         f"The correlation indicators above confirm that "
# #         f" Sale Price correlates most strongly with "
# #         f"the following variables: \n"
# #         f"* Sale Price tends to increase as Overall Quality "
# #         f" (OverallQual) goes up. \n"
# #         f"* Sale Price tends to increase as Groundlevel Living Area "
# #         f" (GrLivArea) increases. \n"
# #         f"* Sale Price tends to increase with increasing Garage Area "
# #         f" (GarageArea). \n"
# #         f"* Sale Price tends to increase with an increase in Total "
# #         f" Basement Area (TotalBsmtSF). \n"
# #         f"* Sale Price tends to increase with an increase in "
# #         f" Year Built (YearBuilt). \n"
# #         f"* Sale Price tends to increase with an increase in "
# #         f" 1st Floor Squarefootage (1stFlrSF). \n\n"
# #         f"The scatterplots below illustrate the trends of the"
# #         f" correlations for each variable."
# #         f" Firstly a two-dimentional histogram plot gives an idea"
# #         f" of the data trend but also where the majority of data"
# #         f" are concentrated. The darker blue areas indicated a high "
# #         f" concentration of data points. "
# #         f" The scatterplots in which the data points are shaded in redish"
# #         f" tones, illustrates the correlation of the variable with"
# #         f" Sale Price, but also shows how the Overall Quality of "
# #         f" the homes always goes up with the Sale Price. "
# #         f" Each data point is colored according"
# #         f" to the Overall Quality of that data point, with the darker"
# #         f" colors indicating higher quality. The"
# #         f" trend that with increasing overall quality the Sale Price"
# #         f" increases can be clearly seen on all plots."
# #     )

# #     # Correlation plots adapted from the Data Cleaning Notebook
# #     if st.checkbox("Correlation Plots of Variables vs Sale Price"):
# #         correlation_to_sale_price_hist_scat(df, vars_to_study)

# #     st.info(
# #         f"*** Heatmap and Barplot: Predictive Power Score (PPS) ***  \n\n"
# #         f"Finally, the PPS detects linear or non-linear relationships "
# #         f"between two variables.\n"
# #         f"The score ranges from 0 (no predictive power) to 1 "
# #         f"(perfect predictive power). \n"
# #         f" To use the plot, find the row on the y-axis labeled 'SalePrice' "
# #         f" then follow along the row and see the variables, labeled on the "
# #         f" x-axis, with a pps of more"
# #         f" than 0.15 expressed on the plot. Overall Quality (OverallQual)"
# #         f" has the highest predictive power for the Sale Price target.")

# #     if st.checkbox("Predictive Power Score"):
# #         calc_display_pps_matrix(df)


# # def correlation_to_sale_price_hist_scat(df, vars_to_study):
# #     """ Display correlation plot between variables and sale price """
# #     target_var = 'SalePrice'
# #     for col in vars_to_study:
# #         fig, axes = plt.subplots(figsize=(8, 5))
# #         axes = sns.histplot(data=df, x=col, y=target_var)
# #         plt.title(f"{col}", fontsize=20, y=1.05)
# #         st.pyplot(fig)
# #         st.write("\n\n")

# #         fig, axes = plt.subplots(figsize=(8, 5))
# #         axes = sns.scatterplot(data=df, x=col, y=target_var, hue='OverallQual')
# #         # plt.xticks(rotation=90)
# #         plt.title(f"{col}", fontsize=20, y=1.05)
# #         st.pyplot(fig)
# #         st.write("\n\n")


# def calc_display_pearson_corr_heat(df):
#     """ Calcuate and display Pearson Correlation """
#     df_corr_pearson = df.corr(method="pearson")
#     heatmap_corr(df=df_corr_pearson, threshold=0.4,
#                  figsize=(12, 10), font_annot=10)


# # def calc_display_spearman_corr_heat(df):
# #     """ Calcuate and display Spearman Correlation """
# #     df_corr_spearman = df.corr(method="spearman")
# #     heatmap_corr(df=df_corr_spearman, threshold=0.6,
# #                  figsize=(12, 10), font_annot=10)


def display_pearson_corr_bar(df):
    """ Calcuate and display Pearson Correlation """
    corr_pearson = df.corr(method='pearson')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    axes = plt.bar(x=corr_pearson[:5].index, height=corr_pearson[:5])
    plt.title("Pearson Correlation of Attributes with Sale Price", fontsize=15, y=1.05)
    plt.xticks(rotation=90)
    plt.ylabel("Pearson Coefficient")
    st.pyplot(fig)


def display_spearman_corr_bar(df):
    """ Calcuate and display Spearman Correlation """
    corr_spearman = df.corr(method='spearman')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    axes = plt.bar(x=corr_spearman[:5].index, height=corr_spearman[:5])
    plt.title("Spearman Correlation of Attributes with Sale Price", fontsize=15, y=1.05)
    plt.xticks(rotation=90)
    plt.ylabel("Spearman Coefficient")
    st.pyplot(fig)


# # def calc_display_pps_matrix(df):
# #     """ Calcuate and display Predictive Power Score """
# #     pps_matrix_raw = pps.matrix(df)
# #     pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
# #         columns='x', index='y', values='ppscore')
# #     heatmap_pps(df=pps_matrix, threshold=0.15, figsize=(12, 10), font_annot=10)

# #     pps_topscores = pps_matrix.iloc[19].sort_values(
# #         key=abs, ascending=False)[1:6]

# #     fig, axes = plt.subplots(figsize=(6, 3))
# #     axes = plt.bar(x=pps_topscores.index, height=pps_topscores)
# #     plt.xticks(rotation=90)
# #     plt.title("Predictive Power Score for Sale Price", fontsize=15, y=1.05)
# #     st.pyplot(fig)


# def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
#     if len(df.columns) > 1:
#         mask = np.zeros_like(df, dtype=np.bool)
#         mask[np.triu_indices_from(mask)] = True
#         mask[abs(df) < threshold] = True

#         fig, axes = plt.subplots(figsize=figsize)
#         sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
#                     mask=mask, cmap='viridis', annot_kws={"size": font_annot}, ax=axes,
#                     linewidth=0.5
#                     )
#         axes.set_yticklabels(df.columns, rotation=0)
#         plt.ylim(len(df.columns), 0)
#         st.pyplot(fig)


# # def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
# #     """ Heatmap for predictive power score from CI template"""
# #     if len(df.columns) > 1:
# #         mask = np.zeros_like(df, dtype=bool)
# #         mask[abs(df) < threshold] = True
# #         fig, axes = plt.subplots(figsize=figsize)
# #         axes = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
# #                            mask=mask, cmap='rocket_r',
# #                            annot_kws={"size": font_annot},
# #                            linewidth=0.05, linecolor='grey')
# #         plt.ylim(len(df.columns), 0)
# #         st.pyplot(fig)


##########new code ########


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig, axes = plt.subplots(figsize=figsize)
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis', annot_kws={"size": font_annot}, ax=axes,
                    linewidth=0.5
                    )
        axes.set_yticklabels(df.columns, rotation=0)
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[abs(df) < threshold] = True
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                         mask=mask, cmap='rocket_r', annot_kws={"size": font_annot},
                         linewidth=0.05, linecolor='grey')
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def CalculateCorrAndPPS(df):
    df_corr_spearman = df.corr(method="spearman")
    df_corr_pearson = df.corr(method="pearson")

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(columns='x', index='y', values='ppscore')

    pps_score_stats = pps_matrix_raw.query("ppscore < 1").filter(['ppscore']).describe().T
    print("PPS threshold - check PPS score IQR to decide threshold for heatmap \n")
    print(pps_score_stats.round(3))

    return df_corr_pearson, df_corr_spearman, pps_matrix


def DisplayCorrAndPPS(df_corr_pearson, df_corr_spearman, pps_matrix, CorrThreshold, PPS_Threshold,
                      figsize=(20, 12), font_annot=8):

    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold, figsize=figsize, font_annot=font_annot)

    

    