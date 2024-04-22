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
Heritage Housing Issues utilises conventional data analysis techniques and machine learning to answer a clients business requirements relating to property sales in Ames, Iowa, USA. It includes a comprehensive data correlation study aimed at visualising the relationship between house attributes and property sale price.  The project also features a machine learning model that enables the client and other users to predict property sale prices based on various house attributes, through a dedicated web application. The project utilizes a Kaggle real estate dataset as its input. This dataset provides comprehensive data on house attributes and sale price in order to accurately answer a real-world business requirement. 

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
As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.


## Hypotheses and Validation
Taking into consideration the business requirements, knowledge of the dataset, nad existing knowledge and/or assumptions on house sale prices in alternative locations, the follwoing hypotheses have been formulated:

* <b>1: Size Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The size of a property does not affect sale price.
    * <b>Alternative Hypothesis (H1):</b> The size of a property is positively correlated to sale price.
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>2: Condition Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The condition of a property does not affect sale price
    * <b>Alternative Hypothesis (H1):</b> The condition of a property is positively correlated to sale price.
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>3: Year Built Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The year of build does not affect sale price.
    * <b>Alternative Hypothesis (H1):</b> The year of build does affect sale price. A subhypothesis is that properties built in recent years would achieve a higher sales price.
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>4: Storey Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The presence of a second storey/floor does not affect sale price.
    * <b>Alternative Hypothesis (H1):</b> Properties with a second storey/floor achieve higher sale price.
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>5: Garage Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The presence of a garage does not affect sale price.
    * <b>Alternative Hypothesis (H1):</b> Properties with garages achieve higher sale prices. 
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>6: Basement Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The presence of a basement does not affect sale price.
    * <b>Alternative Hypothesis (H1):</b> Properties with basements achieve higher sale prices.
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>7: Lot Size Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The lot size of a property does not affect sale price.
    * <b>Alternative Hypothesis (H1):</b> The lot size of a property is positively correlated to sale price.
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>8: Important Features Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> The size of a property, lot size, and overall condition are not within the most important features in predicting sale price.
    * <b>Alternative Hypothesis (H1):</b> The size of a property, lot size and overall condition are expected to be included in the most important features in predicting sale price.
    * <b>How to Validate:</b>
    * <b>Validation:</b>

* <b>9: Sale Price Prediction Hypothesis</b> 
    * <b>Null Hypothesis (H0):</b> We are not able to predict a sale price with an R2 value of at least 0.75 based on important features that have been identified through machine learning modelling.
    * <b>Alternative Hypothesis (H1):</b> We are able to predict a sale price with an R2 value of at least 0.75, based on important features that have been identified through machine learning modelling.
    * <b>How to Validate:</b>
    * <b>Validation:</b>


## Rationale to Map Business Requirements to the Data Visualizations and ML tasks
<b>Business Requirement 1: Data Visualization and Correlation Study</b>

* As a client, I 

<b>Business Requirement 2: Regression and Data Analysis</b>

* As a client, I 
clearly indicate at least one machine learning task present in the project.

<b>Business Requirement 3: Online App and Deployment</b>

* As a client, I 


## ML Business Case
* We want an ML model to predict..
* Our ideal outcome is...
* Our model success metrics are....
* The output is defined..
* Heuristics...
* The training data....

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
* <b>GitHub</b> - a web-based platform and version control system that was utilised for hosting and managing the project repository.
* <b>Gitpod</b> - a cloud-based integrated development environment used to create this project.
* <b>Jupyter Notebooks</b> - interactive computing environments that allow users to create and share documents containing code, visualisations and text. Jupyter Notebooks were used throughout this project for data analysis, and ML pipeline development and evaluation.
* <b>Kaggle</b> - an online platform with open source data, used as the data source for this project.

## Main Data Analysis and Machine Learning Libraries
* <b>numpy</b> - a library that provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. Numpy was used in the Data Cleaning Notebook to create arrays and masks for filtering data.
* <b>pandas</b> - a library with easy-to-use data structures and functions. Throuughouit this project, pandas was utilised for working with dataframes, selecting and displaying key features of the data, creating reports and providing insights on data.
* <b>matplotlib</b> - a library for creating static and interactive visualizations in Python. Matplotlib was used in the Data Visualisations and Feature Engineering Notebooks for displaying plots eg. scatterplots, bar plots and heatmaps.
* <b>seaborn</b> - a library used for visualising data from pandas dataframes and arrays. Seaborn was utilised in the Feature Engineering Notebook to plot heatmaps for correlation and predictive power score analysis.
* <b>ydata_profiling</b> - a package for data profiling, that automates and standardises the generation of detailed reports, complete with statistics and visualizations. In Data Cleaning and Feature Engineering Notebooks it was utilised to generate pandas profile reports with tremendous insights on the data.
* <b>feature_engine</b> - a library that provides a set of tools for feature engineering. It was used in the Feature Engineering Notebook for data transformation such as SmartCorrelatedSelection, OrdinalEncoder and Winsorizer.
* <b>ppscore</b> - predictive power score is a statistical metric used to quantify the predictive power of one feature with respect to another feature in a dataset. This tool was utilised in the Data Cleaning Notebook to assess correlation levels between house attributes.
* <b>streamlit</b> - a library used for creating interactive web applications for data science and machione learning projects. Streamlit was utilised for creating the user dashboard.
* <b>scikit-learn</b> - a library that provides a range algorithms for machine learning models. Scikit-learn was utilised in the Modelling and Evaluation Notebook for grid search cross validation and hyperparameter optimisation.
* <b>xgboost</b> - eXtreme Gradient Boosting is a library that provides gradient boosting algorithms for machine learning tasks. Xgboost was utilised in the Modelling and Evaluation Notebook for for grid search cross validation.

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


