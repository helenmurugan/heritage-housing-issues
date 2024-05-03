import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def page_summary_body():

    st.write("### Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Purpose and Motivation**\n\n"
        f"The client is an individual with four inherited properties in "
        f"Ames, Iowa, who seeks to gain insights into the factors "
        f"influencing house sale prices in the area. Additionally, "
        f"the client wishes to predict the sale prices of their "
        f"inherited properties and explore the potential sale prices "
        f"of other houses in Ames, Iowa.\n\n"
        f"The ideal outcome is to develop a predictive ML model that "
        f"accurately estimates house sale prices based on various "
        f"house attributes. The model will enable the client to make "
        f"informed decisions regarding pricing strategies for their "
        f"inherited properties and provide insights into the local "
        f"real estate market in Ames, Iowa.\n\n"
        f"**Project Terms and Jargon**\n\n"
        f"* **Client** refers to an individual who requires data-related"
        f" services and insights to address specific business "
        f"needs.\n"
        f"* **Sale price** refers to the monetary value at which "
        f"a property is sold. The values are represented in US dollars.\n"
        f"* **Property, real estate or house** may be used interchangably, "
        f"and refer to the sales records in the dataset. \n"
        f"* **Features, attributes and variables** refer to the "
        f"characteristics of a property and may be used interchangably. \n\n"
        f"**Project Dataset**\n"
        f"* The dataset can be accessed from [Kaggle]"
        f"(https://www.kaggle.com/datasets/codeinstitute/housing-prices-data)"
        f".\n"
        f"* The original dataset represents a record of 1460 real estate "
        f" sales in Ames, Iowa. Each record contains the sale price as "
        f"well 23 house attributes such as Ground Floor Living Area, "
        f"Basement Area, Garage Area,  Kitchen Quality, Lot Size,"
        f" Porch Size, Wood Deck Size, Year Built and Year Remodelled."
    )

    # text based on README file - "Business Requirements" section
    st.success(
        f"**Business Requirements**\n\n"
        f"A client has received an inheritance from a deceased "
        f"great-grandfather. Included in the inheritance are four "
        f"houses located in Ames, Iowa, USA. Although the client has "
        f"an excellent understanding of property prices in her home "
        f"country of Belgium, she fears that basing her estimates of "
        f"property worth on her current knowledge "
        f"might lead to inaccurate appraisals in the Iowan market.\n\n"
        f"The client would like to maximise the sales "
        f"price for the four houses, and the following business "
        f"requirements will help her achieve this goal. \n\n"
        f"* 1 - The client is interested in discovering how the house "
        f"attributes correlate with the sale price. Therefore, the client "
        f"expects data visualisations of the correlated variables against"
        f" the sale price to show that.\n\n"
        f"* 2 - The client is interested in predicting the house sale price "
        f"from her four inherited houses and any other house in Ames, Iowa."
    )

    # Link to README file, so the users can have access to full
    # project documentation
    st.write(
        f"* For additional information on this project please consult the "
        f"[README.md](https://github.com/helenmurugan/heritage-housing-issues)"
        f" file for the development of this web app, which is hosted on "
        f"GitHub."
    )
