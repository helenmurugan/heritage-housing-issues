# Heritage Housing Issues
Data Correlation and Predictive Modelling Study<br>
Developed by [Helen Murugan](https://github.com/helenmurugan)

[!Image](filepath)

Live Site: [Heritage Housing Sale Price Predictor](link)<br>
Link to [Repository](https://github.com/helenmurugan/heritage-housing-issues)

## Contents
* [Introduction](#introduction)
* [CRISP-DM Workflow](#crisp-dm-workflow)
* [Datset Content](#dataset-content)
* [Business Requirements](#business-requirements)
* [Hypothesis and how to Validate](#hypothesis-and-how-to-validate)
* [Rationale to Map Business Requirements to the Data Visualizations and ML tasks](#rationale-to-map-business-requirements-to-the-data-visualizations-and-ml-tasks)
* [ML Business Case](#ml-business-case)
* [Dashboard Design](#dashboard-design)
    * [Page 1: Project Summary](#page-1-project-summary)
    * [Page 2: Sale Price Correlation Analysis](#page-2-sale-price-correlation-analysis)
    * [Page 3: Sale Price Prediction](#page-3-sale-price-prediction)
    * [Page 4: Hypothesis and Validation](#page-4-hypothesis-and-validation)
    * [Page 5: Machine Learning Model](#page-4-hypothesis-and-validation)
* [PEP8 Compliance Testing](#pep8-compliance-testing)
* [Unfixed Bugs](#unfixed-bugs)
* [User Warnings](#user-warnings)
* [Deployment](#deployment)
    * [Heroku](#heroku)
* [Main Technologies](#main-technologies)
* [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
* [Credits](#credits)
    * [Content](#content)
    * [Media](#media)
    * [Acknowledgements](#acknowledgements)

## Introduction
Heritage Housing Issues includes a comprehensive data correlation study aimed at visualising the relationship between house attributes and sale prices for properties in Ames, Iowa. Additionally, the project features a machine learning model that enables clients and other users to predict property sale prices based on various house attributes. The project utilizes a Kaggle real estate dataset as its input.

## CRISP-DM Workflow
The Cross Industry Standard Process for Data Mining (CRISP-DM) was followed during development:

* <b>EPIC 1 - Business Understanding</b>: This phase entails extensive discussions with the client to understand their expectations and develop acceptance criteria, as outlined in the Business Requirements section below.

* <b>EPIC 2 - Data Understanding</b>: The identification and understanding of data necessary to fulfill the business requirements are crucial. An initial statistical analysis is conducted to determine if the available data are sufficient. This task is performed in the Data Cleaning Notebook.

* <b>EPIC 3 - Data Preparation</b>: Data cleaning, imputation, and feature engineering are carried out in this phase, ensuring the most effective and accurate modeling outcome. This step takes place in the Data Cleaning and Feature Engineering Notebooks.

* <b>EPIC 4 - Modelling</b>: Model algorithms are determined, and the data is split into train and test sets. Various algorithms are validated and tuned using hyperparameter search on the train sets. This phase is executed in the Modelling_and_Evaluation Notebook.

* <b>EPIC 5 - Evaluation</b>: Model performance is evaluated using the test set, matching the results with the business acceptance criteria. This evaluation process occurs in the Modelling_and_Evaluation Notebook.

* <b>EPIC 6 - Deployment</b>: The streamlit app, developed to meet the business requirements in collaboration with the client, is deployed online. The deployment process is described in the Deployment section below.

These steps align neatly with the Agile development process, with the ability to move back and forth between stages/epics as new insights are gained and previous steps are refined. Ultimately, the project aims to deliver a product that satisfies the client's requirements.


## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data).
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Range|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|


## Business Requirements
As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.


## Hypothesis and how to Validate
In alignment with the business requirements and through discussions with the client, the following hypotheses have been formulated:

<b>Correlation between Property Sale Price and Features:</b> 
* We hypothesise that a property's sale price strongly correlates with a subset of features in the dataset. Our aim is to validate this hypothesis through a correlation study of the dataset.
* The extensive correlation study conducted and displayed on the app confirms this hypothesis.

<b>Strong Correlation with Common Features:</b>
* We hypothesize that the correlation is strongest with common features of a home, such as total square footage, overall condition, and overall quality. Our aim is to validate this hypothesis through a correlation study.
* The extensive correlation study confirms that the five features with the strongest correlation to Sale Price are: 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', and '1stFlrSF'. These features are common to the majority of homes.

<b> Predictive Model Performance:</b> 
* We hypothesize that we can predict a sale price with an R2 value of at least 0.8. To validate this, we propose developing a predictive model, optimizing it using data modeling tools, and evaluating it based on the required criteria.

The model evaluation has validated this hypothesis, achieving R2 values of ... for both train and test sets.


## Rationale to Map Business Requirements to the Data Visualizations and ML tasks
<b>Business Requirement 1: Data Visualization and Correlation Study</b>

* Inspect the dataset to understand its relevance to property sale prices in Ames, Iowa.
* Conduct a correlation study (Pearson and Spearman) to analyze the relationships between variables and sale prices.
* Visualise the most important and relevant data against sale prices to gain insights.

<b>Business Requirement 2: Regression and Data Analysis</b>

* Develop a regression model with sale price as the target value to predict home prices in Ames, Iowa.</b>
* Optimize and evaluate the regression model to achieve an R2 value of 0.8 or higher.

<b>Business Requirement 3: Online App and Deployment</b>

* Build a Streamlit app showcasing data analysis, visualization, and a feature for predicting sale prices for properties in Ames, Iowa.
* Deploy the app using Heroku to make it accessible to the client and other users.








## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


## Dashboard Design
The project will be built using a Streamlit dashboard and will contain the following pages:

### Page 1: Project Summary
* Statement of the project purpose.
* Project terms and jargon.
* Brief description of the data set.
* Statement of business requirements.
* Links to further information.

Project Summary Page Screenshots

### Page 2: Sale Price Correlation Analysis
This page will satisfy the first business requirement of discovering how the house attributes correlate with the sale price. It includes checkboxes so the client has the ability to display the following visual guides to the data features:

* A sample of data from the data set.
* Pearson and Spearman correlation plots between the features and the sale price.
* Histogram and scatterplots of the most important predictive features.
* Predictive Power Score Analysis.

Correlation Analysis Screenshots
### Page 3: Sale Price Prediction
This page will satisfy the second business requirement or predicting house price sales. It will include:

* Input feature of property attributes to produce a prediction on the sale price.
* Display of the predicted sale price.
* Feature to predict the sale prices of the clients specific data in relation to her inherited properties.

Sale Price Prediction Screenshots

### Page 4: Hypothesis and Validation

* Description of how project hypotheses were validated.

Hypothesis and Validation Screenshots

### Page 5: Machine Learning Model

* Description of the ML pipeline used to train the model.
* Information on feature importance.
* Evaluation of the pipeline performance.

ML Model: Price Prediction Screenshots

# PEP8 Compliance Testing 
The python code from the app_pages files was passed through the CI Python Linter. Code now passes with no errors.


## Unfixed Bugs
When evaluating missing data in notebook 3, the og df had to be reloaded in order for missing data to be evaluated. The label encoder must encode the variables in a way that they no longer have missing data.

## User Warnings
During correlation and PPS analysis, the following warning was generated. A correlation is a measure of the strength of association between two variables. The warning is just a caution that the data may not be sufficient to show strong correlations In this case, the warning was ignored.
(image of n_splits warning saved on desktop)


## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Technologies
* GitHub - used to create the project repository and for version control.
* Gitpod - integrated development environment used to build this project.
* Jupyter Notebooks - used for data analysis and ML pipeline development and evaluation.
* Kaggle - open source data used in this project.
* Streamlit - used to develop the user dashboard.

## Main Data Analysis and Machine Learning Libraries
* os - module in python that provides a way of using operating system dependent functionality.
* pandas - a library used to generate reports with statistical analysis of data.
* sci-kit learn - modules for pre-processing data eg. LabelEncoder.
* matplotlib - a library for creating static and interactive visualizations in Python.
* seaborn - a python library used for visualizing data from Pandas data frames and arrays. Used to plot heatmaps for correlation and predictive power score analysis.
* ydata_profiling - used to generate the pandas profile reports.
* numpy - a library in Python that provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. 
* feature_engine - a library that provides a set of tools for feature engineering in Python.
* scipy - 
* warnings - provides functions to issue warning alerting users to potential issues.
* sklearn.pipeline - a module that provides a flexible way to chain together multiple data processing steps into a single object.  
* joblib -  a library primarily used for saving and loading Python objects.

## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site

## Acknowledgements
* Precious Ijege
* CI peers
* Arul Murugan


