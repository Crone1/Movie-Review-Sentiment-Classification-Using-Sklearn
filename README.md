# Overview
This repository contains the work I did in using sklearn machine learning models and other natural language techniques for classification of movie review sentiment.
This project was completed as part of a college module *'Natural Language Technologies'*.


# Contents
This work developed on the code given in the notebook *'original_sentiment_naive_bayes_nb.ipynb'*. The code in this notebook was refactored and developed on to create a new better version of this file - *'movie_sentiment_classification.ipynb'* notebook.
This notebook details my modelling process and uses the functions contained in the *'Sentiment_classification_functions.py'* file to create a model with high accuracy at classifying sentiment of movie reviews.

The *'Data'* folder contains the original data used in this modelling process but also contains the evaluation results of each of the models tested in this assignment.
If this *'Data'* folder is empty, the *'chromedriver.exe'* file is needed to scrape the data from the internet.
This driver may be out of date depending on th version of google chrome you have installed.
This issue can be solved by downloading the updated version of chromedriver that matches your version of chrome [here](https://chromedriver.chromium.org/downloads).

# Testing the model on new data
To test the model on new data, it is a matter of switching out the original dataset in the *'Data'* folder or else changing the web link to the online dataset in the *.py* file.
As long as this data file is in a similar format to the original dataset, the code will load it in and split it into a train and test split dataset.
Once this is done, you will need to specify which parameters you want to evaluate the model using.
This can be done by commenting out the parameters in the parameter lists that you don't want to the model to have.
The output evaluation results folder location should be changed to ensure the results are not mixed up with the results of the model results in my initial training and experimentation.
Once this is done, the iteration evaluation cell can be run to iteratively evaluating the model useing the specified parameter configuration.
This will train the model, test the model, and will output the evaluation results to a CSV.
