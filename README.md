# Machine-Learning-Capstone-Project

Here I discuss my machine learning capstone project I applied various algorithms (for both classification and regression) to identify the most reliable algorithm that depicts the highest performance on both training and test data and that can be considered for the future dataset.

Dataset

The data file “GHG_Emission.csv” has been retrieved from AER website; where the locations of the wells have been changed, and some key properties are generated synthetically or are greatly manipulated for confidential reasons.

Regression and Classification


Gathering Data

First, the dataset was imported and read using pandas. The data was shuffled and then random. seed (42) was used to save the state of a random function. The index of the data was reset.

Data Processing

Stratified sampling was performed for even distribution of data. The test and training data were split based on that.
The outliers were removed for instances out of the range of 𝜇 ± 2.5𝜎, imputation (with median) was performed, text handling using one-hot encoding and standardization.


CLASSIFICATION


Model Training for Classification

Binary classification was applied using the following Machine Learning models below.
• Dummy Classifier
• Stochastic Gradient Descent
• Logistic Regression
• Support Vector Machine: Linear
• Support Vector Machine: Polynomial Kernel
• Decision Trees
• Random Forest
• Adaptive Boosting with Linear SVM
• Adaptive Boosting
• Hard and Soft Voting
• Shallow Neural Network (with 3 layers)
• Deep Neural Network ( with 6 layers)

The hyperparameters were fine-tuned using RandomizedSearchCV based on accuracy. The optimized parameters were used to predict accuracy. K-fold cross-validation with 5-folds (cv=5) was applied and then the mean of 5 accuracies for each classifier was calculated.
These optimized hyper-parameters for all the above-mentioned algorithms were used for finding the performance on the test dataset as well.

Model Performance for Classification

Random Forest should be used for future datasets as it gives the best performance on both testing and training data.

REGRESSION


Model Training for Regression

Similar to Binary classification, Regression was applied with the following Machine Learning models below.
• Linear Regression
• Support Vector Machine: Polynomial Kernel
• Decision Trees
• Random Forest
• Gradient Boosting
• Shallow Neural Network (with 3 layers)

The hyperparameters were fine-tuned using RandomizedSearchCV based on RMSE. The optimized parameters were used to predict RMSE. K-fold cross-validation with 5-folds (cv=5) was applied and then the mean of 5 RMSEs for each regressor was calculated.
These optimized hyper-parameters for all the above-mentioned algorithms were used for finding the performance on the test dataset as well.

Model Performance for Regression

Random Forest should be used for future datasets as it gives the best performance on both testing and training data.
 
 
