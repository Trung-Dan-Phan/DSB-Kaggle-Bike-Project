# Bike Traffic Prediction Project
MAP536 - Python for Data Science Final Project: Bike traffic counters installed by Paris city - Kaggle Challenge 

## Overview

Our bike traffic prediction project focuses on leveraging cyclist counters installed by the Paris city council, encompassing data from September 1st, 2020, to September 9th, 2021. The objective is to predict hourly bike counts at 30 different sites, addressing the challenge of understanding and forecasting bike traffic patterns in the city. The significance lies in providing insights for urban planning and promoting sustainable transportation.

### Data Source
The dataset originates from cyclist counters deployed by the Paris city council.

### Project Objectives
- Predict hourly bike counts at various sites.
- Uncover patterns and trends in bike traffic.
- Contribute valuable insights for urban planning and transportation optimization.

## Team Members
- Trung Dan Phan
- Antoine Gosset

## Installation and Setup

To set up the project, follow these steps:
1. Clone the repository.
2. Ensure the required Python version and dependencies are installed using `pip install -r requirements.txt`.

## File Structure

The project is organized as follows:
- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and modeling.
- `script/`: Script for Kaggle submission 
- `requirements.txt`: Lists project dependencies.

## Methodology

Our approach involves:
- Exploratory Data Analysis (EDA) to gain insights.
- Feature Engineering to enhance predictive power.
- Predictive model implementations and hyperparameter tuning.
- Evaluation metrics and validation strategies on Kaggle test set to revise our strategy and adapt accordingly.

## Usage

Instructions on running scripts or notebooks:
- Execute the analysis.
- Train the model or run predictions.

Uncommon libraries documentation:
- https://pypi.org/project/ydata-profiling/
- https://pypi.org/project/holidays/
- https://dev.meteostat.net/python/
- https://pypi.org/project/vacances-scolaires-france/#description
- https://pypi.org/project/lockdowndates/
- https://xgboost.readthedocs.io/en/stable/python/python_intro.html
- https://catboost.ai/docs/concepts/python-quickstart
- https://lightgbm.readthedocs.io/en/latest/Python-Intro.html

## Results

In summary, our project focused on optimizing four key models - XGBoost, RandomForest, Catboost, and LGBM - through exhaustive hyperparameter tuning, utilizing grid search cross-validation and Optuna. The meticulous process resulted in significant improvements in predictive performance. LGBM emerged as the top-performing model, showcasing remarkable accuracy and an ability to capture intricate patterns in predicting bike counts.

Equipped with the finely-tuned LGBM model and optimal parameters, we were confident in deploying it in real-world scenarios. Anticipating strong performance on Kaggle's test set, we expected RMSE results to align closely with refined predictions, marking a notable advancement in implementing a reliable and accurate model for practical applications.

However, our initial Kaggle submission, placing us at the bottom of the leaderboard, prompted a crucial reassessment. We shifted from a complex to an iterative approach, emphasizing the importance of starting with a simple, overfit-resistant model and gradually introducing complexity. This strategic change allowed us to thoroughly evaluate each feature's impact, refining the feature engineering process for enhanced model performance.

## Challenges and Learnings

Our participation in the Kaggle challenge underscores the significance of adaptability and strategic thinking in the field of data science. Initially challenged by overfitting resulting from a complex model, we transitioned to a more straightforward, iterative approach, introducing complexity judiciously for maximum benefit. This shift in strategy resulted in a consistent ascent up the leaderboard.

By incorporating nuanced features such as weather patterns and school holidays, we struck a balance between feature richness and model simplicity, ultimately opting for LightGBM for its efficiency. The final model, honed through meticulous feature selection, demonstrated our ability to seamlessly integrate data insights with technical expertise.

This experience reinforces fundamental data science principles: commence with simplicity, assess rigorously, and continually adapt. The noteworthy improvement in both public and private scores serves as a testament to these principles. Moving forward, the insights gained from this challenge will inform our future ventures in data science, armed with a more profound understanding of model construction and feature selection.

## Acknowledgements

Credit to the Paris city council for providing the dataset.

Credit to our professors Mathurin Massias, Badr Moufad, and Vincent Maladiere for their guidance on data science best practices. 
