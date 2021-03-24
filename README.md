# Overview

This repository contains my submission for assignment 2 in the module Natural Language Processing.

# Contents

The main file in this repo where all the work is done it the *movie_sentiment_classification.ipynb* notebook.
This notebook details my modelling process and uses the functions contained in the *Sentiment_classification_functions.py* file to create a model with high accuracy at classifying sentiment of movie reviews.

The Data folder contains the original data used in this modelling process but also contains the evaluation results of each of the models tested in this assignment.

# Testing the model on new data

To test the model on new data, it is a matter of switching out the original dataset in the *Data* folder or else changing the web link to the online dataset in the *.py* file.
As long as this data file is in a similar format to the original dataset, the code will load it in and split it into a train and test split dataset.
Once this is done, you will need to specify which parameters you want to evaluate the model using.
This can be done by commenting out the parameters in the parameter lists that you don't want to the model to have.
The output evaluation results folder location shout be changed to ensure the results are not mixed up with the results of the models in my initial training and experimentation.
Once this is done, the iteration evaluation cell can be run to iteratively evaluating the model useing the specified parameter configuration.
This will train the model, test the model, and will output the evaluation results to a CSV.

** If you have any problems getting this running, feel free to contact me and I will be sure to help with this process.