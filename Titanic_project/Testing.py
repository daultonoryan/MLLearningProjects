import numpy as np
import pandas as pd
from sklearn import svm, model_selection


titanic_df = pd.read_csv("titanic_data.csv")
qualifiers = np.array(titanic_df[["SibSp", "Parch"]])
outcomes = np.array(titanic_df["Survived"])



def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """

    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred):

        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean() * 100)

    else:
        return "Number of predictions does not match number of outcomes!"


def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        # Predict the survival of 'passenger'
        predictions.append(0)

    # Return our predictions
    return pd.Series(predictions)


def predictions_1(data):
    """ Model with one feature:
            - Predict a passenger survived if they are female. """

    predictions = []
    for _, passenger in data.iterrows():

        # Remove the 'pass' statement below
        # and write your prediction conditions here
        if passenger["Sex"] == "female":
            predictions.append(1)
        else:
            predictions.append(0)

    # Return our predictions
    return pd.Series(predictions)


def predictions_2(data):
    """ Model with two features:
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """

    predictions = []
    for _, passenger in data.iterrows():
        # Remove the 'pass' statement below
        # and write your prediction conditions here
        if passenger["Sex"] == "female":
            predictions.append(1)
        elif passenger["Sex"] == "male" and passenger["Age"] < 10:
            predictions.append(1)
        else:
            predictions.append(0)

    # Return our predictions
    return pd.Series(predictions)


def predictions_3(data):
    """ Model with two features:
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """

    predictions = []
    for _, passenger in data.iterrows():
        # Remove the 'pass' statement below
        # and write your prediction conditions here
        if passenger["Sex"] == "female":
            if passenger["SibSp"] > 2 or passenger["Parch"] > 3:
                predictions.append(0)
            else:
                predictions.append(1)
        elif passenger["Sex"] == "male" and passenger["Age"] < 10:
            predictions.append(1)
        else:
            predictions.append(0)

    # Return our predictions
    return pd.Series(predictions)


# Make the predictions
predictions = predictions_3(titanic_df)

print (accuracy_score(outcomes, predictions))



