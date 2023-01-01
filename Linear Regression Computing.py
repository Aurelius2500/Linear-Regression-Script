# -*- coding: utf-8 -*-
"""
Data Analytics Computing Follow Along: Linear Regression With Python
Spyder version 5.1.5
"""

# This is a comment
# Import required packages. We need these pakcages for the script to run
# Think of a package as a piece of software that someone already built

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# If you do not have the required packages, run the following lines

# pip install pandas
# pip install sklearn

print("Packages imported")

# The lines below import the dataset that we will use in this script
# The link in kaggle is in the description

# pandas.read_excel is pretty similar to pandas.read_csv
# This is our first variable, a dataframe

google_df = pd.read_excel('C:/Videos/Google Dataset/Google Dataset.xlsx',
                          sheet_name = "Sheet1")

# We can get the shape of thedataframe with the shape method

google_df.shape

# The print function just shows in the console the text that you give it    
    
print("Dataset imported")

# See the columns and the first rows of the dataset

print(google_df.columns)

print(google_df.head())

# We will be using the Change % as our predictor
# The Average Volume will be the prediction

X = google_df[['Change %']]

y = google_df['Avg. Volume']

# Import the linear regression model from sklearn

linear_model = LinearRegression()

linear_model.fit(X, y)

linear_model.score(X, y)

# The R-Squared value is pretty bad, try with Close instead as a predictor
# We first overwrite X with the new variable

X = google_df[['Close']]

# Use a new linear model to keep the old one

linear_model_2 = LinearRegression()

# Train it and get the score

linear_model_2.fit(X, y)

linear_model_2.score(X, y)

# This score is better than the previous one

# We can calculate the predictions of the model

google_predictions = linear_model_2.predict(X)

# Plot the results of the second regression

plt.ticklabel_format(style='plain')
plt.xlabel("Closing Price")
plt.ylabel("Average Volume")
plt.title("Linear Regression on Monthly Closing Price vs Average Volume")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X, y, color = "Blue", s = 10)
plt.plot(X["Close"], google_predictions, color = "black", linewidth = 2)
plt.show()

# This is the basis of linear regression, the regression line
# Notice how we use a scatterplot to visualize the model
# This is the most basic approach to linear regression
# Notice that there is a point on the far left, this is an outlier

# As this is a linear model, it has an intercept and coefficient
# In some statistics classes, you may be familiar with slope instead

coefficient = linear_model_2.coef_.item()

# The intercept is the point where X is 0

intercept = linear_model_2.intercept_

print(f"The linear regression formula is {intercept} + {coefficient}X")

# Let's look at the predictions once more

plt.ticklabel_format(style='plain')
plt.xlabel("Closing Price")
plt.ylabel("Average Volume")
plt.title("Linear Regression on Monthly Closing Price vs Average Volume")

# The lines above are optional, merely for aesthetic purposes

plt.scatter(X, y, color = "Blue", s = 10)
plt.scatter(X, google_predictions, color = "Red", s = 10)
plt.show()

# Notice how the regression line needs to pass through the predictions

# One advantage of linear regression is that we can predict points outside the original range

new_prediction = intercept + coefficient * 200

print(f"For a 200 dollar Closing Price, the model predicts a volume of {new_prediction}")