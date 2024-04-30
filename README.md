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
* [Hypotheses and Validation](#hypothesis-and-validation)
* [Rationale to Map Business Requirements to the Data Visualizations and ML tasks](#rationale-to-map-business-requirements-to-the-data-visualizations-and-ml-tasks)
* [ML Business Case](#ml-business-case)
* [Dashboard Design](#dashboard-design)
    * [Page 1: Project Summary](#page-1-project-summary)
    * [Page 2: Sale Price Correlation Analysis](#page-2-sale-price-correlation-analysis)
    * [Page 3: Sale Price Prediction](#page-3-sale-price-prediction)
    * [Page 4: Hypothesis and Validation](#page-4-hypothesis-and-validation)
    * [Page 5: Machine Learning Model](#page-4-hypothesis-and-validation)
* [Testing](#testing)
    * [PEP8 Compliance Testing](#pep8-compliance-testing)
    * [Manual Testing](#manual-testing)
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
Heritage Housing Issues utilises conventional data analysis techniques and machine learning to answer a client's business requirements, relating to property sales in Ames, Iowa, USA. It includes a comprehensive data correlation study aimed at visualising the relationship between house attributes and property sale price.  The project also features a machine learning model that enables the client and other users to predict property sale prices based on various house attributes, through a dedicated web application. The project utilizes a Kaggle real estate dataset as its input. This dataset provides comprehensive data on house attributes and sale price in order to accurately answer a real-world business requirement. 

Heritage Housing Issues is the fifth and final portfolio project for a Code Institute Diploma in Full Stack Software Development with Predictive Analytics specialisation.

## CRISP-DM Workflow
The Cross Industry Standard Process for Data Mining (CRISP-DM) was followed during development:

* <b>EPIC 1 - Information gathering and data collection:</b> This stage involves understanding the business requirements (this involves extensive discussions with the client), identifying the data source(s), and collecting necessary data to support the project goals.

* <b>EPIC 2 - Data visualization, cleaning, and preparation:</b> 

* <b>EPIC 3 - Model training, optimization and validation:</b>

* <b>EPIC 4 - Dashboard planning, designing, and development:</b>

* <b>EPIC 5 - Dashboard deployment and release.</b>: 
 

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
A client has received an inheritance from a deceased great-grandfather, included in the inheritance are four houses located in Ames, Iowa, USA. Although the client has an excellent understanding of property prices in her home country of Belgium, she fears that basing her estimates for property worth on her current knowledge of the Iowan market might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. The client would like to maximise the sales price for the four houses. A public dataset with house prices for Ames, Iowa, is provided.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypotheses and Validation
Taking into consideration the business requirements, knowledge of the dataset, and existing knowledge/assumptions on house sale prices in alternative locations, the following hypotheses have been formulated:

* <b>1: Property Size Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> Features relating to the size of a property do not affect sale price.
    * <b>Alternative Hypothesis (H1):</b> Features relating to the size of a property are positively correlated to sale price.
    * <b>How to Validate:</b> Investigate how features relating to property size in square feet or number of bedrooms are correlated to sale price. The features expected to correlate positively with sale price are '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotalBsmtSF', 'GarageArea' and 'GrLivArea'. Plot these features against SalePrice to determine the correlation. In addition, check the Spearman and Pearson correlation heatmaps to determine which features have the greatest affect on sale price.
    * <b>Validation:</b> Features related to property size are positively correlated to sale price as demonstrated by the heatmap analyses and data visualisations. '1stFlrSF', 'GarageArea' and 'GrLivArea' are strongly correlated with SalePrice, whilst '2ndFlrSF', 'BedroomAbvGr' and 'TotalBsmtSF' are weakly correlated to SalePrice.
    * <b>Action</b> Accept alternative hypothesis, sale price is positively and strongly correlated with the square footage of the first floor, garage, above garde living area and basement. However, the client should be informed that number of bedrooms, square footage of the second floor and basement are only weakly positively correlated to sale price.

* <b>2: Year Built Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The year of build does not have a positive correlation with sale price.
    * <b>Alternative Hypothesis (H1):</b> The year of build has a positive correlation with sale price.
    * <b>How to Validate:</b> Check Spearman and Pearson correlation heatmaps to assess correlation levels.
    * <b>Validation:</b> The Spearman and Pearson correlation heatmaps showed that 'YearBlt' has a moderate positive correlation to sale price. This is further demonstrated by a data visualisation.
    * <b>Action</b> Accept alternative hypothesis.

* <b>3: Lot Size Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The lot size of a property does not have a strong positive correlation with sale price.
    * <b>Alternative Hypothesis (H1):</b> The lot size of a property has a strong positive correlation with sale price.
    * <b>How to Validate:</b> Check Spearman and Pearson correlation heatmaps to assess correlation levels for 'LotFrontage' and 'LotArea'.
    * <b>Validation:</b> 'LotFrontage' and 'LotArea' are weakly to moderately correlated with sale price.
    * <b>Action</b> Accept null hypothesis. This may be an interesting insight to the client, as it challenges preconceptions about which features affect sale price.

* <b>4: Sale Price Prediction Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> We are not able to predict a sale price with an R2 value of at least 0.75 based on important features that have been identified through machine learning modelling.
    * <b>Alternative Hypothesis (H1):</b> We are able to predict a sale price with an R2 value of at least 0.75, based on important features that have been identified through machine learning modelling.
    * <b>How to Validate: Calculate and optimise R2 scores for train and test set during Modelling and Evaluation stage. Compare to agreed limit of 0.75.</b>
    * <b>Validation: R2 scores of greater than 0.75 have been achieved.</b>
    * <b>Action</b> Accept alternative hypothesis.


## Rationale to Map Business Requirements to the Data Visualisations and ML tasks
<b>Business Requirement 1: Data Visualisation and Correlation Study</b>

* As a client, I want to perform an in-depth study of house records data in Ames, Iowa, in order to discover how the house attributes correlate to the sale price.
* As a client, I want to perform a correlation study using both Pearson and Spearman methods, in order to gain deeper insights into the multicollinearity relationships between variables and sale prices.
* As a client, I want to visualise the main variables plotted against sale price, in order to gain deeper insights on how house attributes correlate to sale price.
* As a client, I want to visualise house record data in a variety of ways such as heatmaps, scatterplots, bar plots etc, in order to maximise my understanding of the complexities of the house sales market in teh area.
* As a client, I want the results of this data study to feed directly into the development of a machine learning model, in order to predict house sale prices in Ames, Iowa.

<b>Business Requirement 2: Sale Price Prediction</b>

* As a client, I want to create a machine learning model that can accurately predict property sale prices based on house attributes, in order to maximise the sale price of my four inherited houses.
* As a client, I want to identify the most important house features to train an ML model on, in order to achieve the most accurate predictions.
* As a client, I want the ML model inputs to be house attributes and the output to be sale price, in order to easily predict sale prices for my four inherited houses.
* As a client, I want a regressor model to be utilised to predict sale price with an R2 score of at least 0.75 on the train set as well as the test set, in order to measure the performance of the model.
* As a client, I want the model to be optimimised based on testing a selection of algorithms and hyperparameters, in order to achieve the most accurate results for sale price prediction. 

<b>Business Requirement 3: Online App Deployment</b>

* As a client, I want an online dashboard which follows the principles of good UX design, in order to easily and intuitively navigate around the app.
* As a client, I want to input only the most important features relating to house attributes into my adshboard, in order to quickly and accurately receive a sale price prediction from the ML model.
* As a client, I want the data visualisations to be displayed in the app, in order to allow myself and other users to quickly gain insights into how house sale prices are affected by house attributes.


## ML Business Case
* Our client is an individual with inherited properties in Ames, Iowa, who seeks to gain insights into the factors influencing house sale prices in the area. Additionally, the client wishes to predict the sale prices of their inherited properties and explore the potential sale prices of other houses in Ames, Iowa.

* Our ideal outcome is to develop a predictive ML model that accurately estimates house sale prices based on various house attributes. The model will enable the client to make informed decisions regarding pricing strategies for their inherited properties and provide insights into the local real estate market in Ames, Iowa.

* The model will use supervised learning techniques, where the label or target is a continuous numeric variable - SalePrice in US dollars. In this case, a regression model is appropriate to use. 

* The training data for the development of the ML model is sourced from Kaggle. The dataset includes information on 24 house attributes, which are the variables for the ML modelling task (columns) and 1460 observations (rows). The training data will be refined through data pre-processing including data cleaning, exploratory data analysis and feature engineering. 

    * Data cleaning will be required to remove missing data and could includes techniques such as dropping variables, MeanMedianImputation and CategoricalImputation. 

    * The aim of the Feature Engineering stage is to to provide quality data that a model can be trained on, by attempting to transform the data towards normal distribution, where possible. Techniques for feature engineering include Numerical Encoding, Categorical Encoding and SmartCorrelated Selection where certain variables can be dropped if they provide similar correlation to another variable. Additionally, techniques to handle outliers in the data will be explored, for example, using the Winsorizer method.

* We will apply established heuristics to select appropriate machine learning algorithms and tune hyperparameters to optimise model performance. Grid search cross validation of known regressor models will be utilised to identify the algorithm that generalises best on unseen data, measured by R2 scores. Hyperparameter optimisation will be used to tune the model(s) to allow for the best possible performance.These heuristics will ensure that our model is built on sound principles and leverages industry best practices to achieve accurate predictions of house sale prices.

* CRISP-DM workflow will be followed during development to maximise model performance. For example, data cleaning, feature engineering and feature selection can be revisited iteratively until we are satisfied with the model performance, taking into account the expectations agreed with the client.

* Our success metric, as agreed with the client, is to achieve R2 scores of at least 0.75 for both train and test sets. Additionally, since this is a prediction model, we would aim for the difference between R2train and R2test to be less than 0.15 to ensure that our model generalises well on unseen data and does not significantly overfit or underfit the data.


## Dashboard Design
The project will be built using a Streamlit dashboard and will contain the following pages:

### Page 1: Project Summary
* Statement of the project purpose.
* Project terms and jargon.
* Brief description of the dataset.
* Statement of business requirements.
* Links to further information.

Project Summary Page Screenshots

### Page 2: Sale Price Correlation Analysis
This page will satisfy the first business requirement of discovering how the house attributes correlate with the sale price. It includes checkboxes so the client has the ability to display the following visual guides to the data features:

* A sample of data from the data set.
* Pearson and Spearman correlation plots between the features and the sale price.
* Scatterplots of the most important predictive features.
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

## Testing
### PEP8 Compliance Testing 
The python code from all .py files was passed through the [CI Python Linter](https://pep8ci.herokuapp.com/). Code passed with no errors in most cases. However, there was one exceptions where the code could not be split across multiple lines whilst maintaining readability.
* In page_sale_price_analysis - line 286:
    * pps_score_stats = pps_matrix_raw.query("ppscore < 1").filter(['ppscore']).describe().T

### Manual Testing

## Unfixed Bugs
Encoding kitchen qual using label encoder gave me some problems - would use ordinal encoder next time. plot is missing from scatter plots.
When evaluating missing data in notebook 3, the og df had to be reloaded in order for missing data to be evaluated. The label encoder must encode the variables in a way that they no longer have missing data.

Smartcorrelatedselection was not performed properly during ML pipeline in NOtebook 5

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
* <b>GitHub</b> - a web-based platform and version control system that was utilised for hosting and managing the project repository.
* <b>Gitpod</b> - a cloud-based integrated development environment used to create this project.
* <b>Jupyter Notebooks</b> - interactive computing environments that allow users to create and share documents containing code, visualisations and text. Jupyter Notebooks were used throughout this project for data analysis, and ML pipeline development and evaluation.
* <b>Kaggle</b> - an online platform with open source data, used as the data source for this project.

## Main Data Analysis and Machine Learning Libraries
* <b>numpy</b> - a library that provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. Numpy was used in the Data Cleaning Notebook to create arrays and masks for filtering data.
* <b>pandas</b> - a library with easy-to-use data structures and functions. Throuughouit this project, pandas was utilised for working with dataframes, selecting and displaying key features of the data, creating reports and providing insights on data.
* <b>matplotlib</b> - a library for creating static and interactive visualizations in Python. Matplotlib was used in the Data Visualisations and Feature Engineering Notebooks for displaying plots eg. scatterplots, bar plots and heatmaps.
* <b>seaborn</b> - a library used for visualising data from pandas dataframes and arrays. Seaborn was utilised in the Feature Engineering Notebook to plot heatmaps for correlation and predictive power score analysis.
* <b>ydata_profiling</b> - a package for data profiling, that automates and standardises the generation of detailed reports, complete with statistics and visualisations. In Data Cleaning and Feature Engineering Notebooks it was utilised to generate pandas profile reports with tremendous insights on the data.
* <b>feature_engine</b> - a library that provides a set of tools for feature engineering. It was used in the Feature Engineering Notebook for data transformation such as SmartCorrelatedSelection, OrdinalEncoder and Winsorizer.
* <b>ppscore</b> - predictive power score is a statistical metric used to quantify the predictive power of one feature with respect to another feature in a dataset. This tool was utilised in the Data Cleaning Notebook to assess correlation levels between house attributes.
* <b>streamlit</b> - a library used for creating interactive web applications for data science and machione learning projects. Streamlit was utilised for creating the user dashboard.
* <b>scikit-learn</b> - a library that provides a range algorithms for machine learning models. Scikit-learn was utilised in the Modelling and Evaluation Notebook for grid search cross validation and hyperparameter optimisation.
* <b>xgboost</b> - eXtreme Gradient Boosting is a library that provides gradient boosting algorithms for machine learning tasks. Xgboost was utilised in the Modelling and Evaluation Notebook for for grid search cross validation.

## Credits 

### Content
* The following documentation and websites helped with hyperparameter tuning in Notebook 5.
    * [GradientBoostingRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
    * [RandomForestRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn-ensemble-randomforestregressor)
    * [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
    * [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
    * [How Extra trees classification and regression algorithm works](https://pro.arcgis.com/en/pro-app/latest/tool-reference/geoai/how-extra-tree-classification-and-regression-works.htm#:~:text=The%20extra%20trees%20algorithm%2C%20like,selected%20randomly%20for%20each%20tree.)
    * [Ensembles: Gradient boosting, random forests, bagging, voting, stacking](https://scikit-learn.org/stable/modules/ensemble.html#forest)
    * [Streamlit documentation](https://docs.streamlit.io/)

### Code
* Code Institute custom code was used for data cleaning, feature engineering and model fitting and is referenced in the notebooks.

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements
* The following projects were used for inspiration during the development of this app.
    * [Heritage Housing Issues project by URiem](https://github.com/URiem/heritage-housing-PP5/blob/main/README.md)
    * [Heritage Housing Issues project by t-ullis](https://github.com/t-hullis/milestone-project-heritage-housing-issues)
* CI peers, in particular the slack community on the #project-portfolio-5-predictive-analytics channel who were always there to support each other.
* Precious Ijege for his expert mentoring.



