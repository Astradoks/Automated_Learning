# Automated_Learning
 Automated learning with decision tree and neural network with python

To start first run the command pip install -r requirements.txt

The purpose of this code is to compare decision tree and neural network in a prediction that tells if an asteroid is hazard for the Earth. The data for this project was extracted from Asteroid Impacts dataset in Kaggel: https://www.kaggle.com/shrushtijoshi/asteroid-impacts. This data is divided in each run in a random way to get 80% of it for the training and 20% for the test. The only thing changed from the original dataset is that the Names column was deleted because that information doesn't change the result of knowing if the asteroid is hazard for the Earth.

At the end, this project shows a report in console with 6 different tests for decision trees and 6 different tests for neural networks with the same data. The best algorithm is selected by the accuracy of its prediction.

As the data is selected randomly each time, every time the results change, but in general, for this dataset, the decision tree is the best algorithm for this prediction.