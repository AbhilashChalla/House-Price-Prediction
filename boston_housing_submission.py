
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project: Predicting Boston Housing Prices
# 
# Welcome to the first project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# - 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
# - The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.
# 
# Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[17]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)


# ## Data Exploration
# In this first section of this project, you will make a cursory investigation about the Boston housing data and provide your observations. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand and justify your results.
# 
# Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MEDV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.

# ### Implementation: Calculate Statistics
# For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, you will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.

# In[18]:

# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)


# ### Question 1 - Feature Observation
# As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.
# 
# 
# ** Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? Justify your answer for each.**
# 
# **Hint:** This problem can phrased using examples like below.  
# * Would you expect a home that has an `'RM'` value(number of rooms) of 6 be worth more or less than a home that has an `'RM'` value of 7?
# * Would you expect a neighborhood that has an `'LSTAT'` value(percent of lower class workers) of 15 have home prices be worth more or less than a neighborhood that has an `'LSTAT'` value of 20?
# * Would you expect a neighborhood that has an `'PTRATIO'` value(ratio of students to teachers) of 10 have home prices be worth more or less than a neighborhood that has an `'PTRATIO'` value of 15?

# **Answer: ** 
# 
# RM value: An increase in the RM value will lead to increase in the MEDV value. RM is the value of average number of rooms among the homes in neighborhood. It is a basic understanding that more the number of rooms more will be the price of the house as the cost of construction of a bigger house is more and also higher the number of rooms, more is the space consumed. 
# 
# LSTAT value: The LSTAT value represents the percentage of lower class workers in that neighborhood. Higher the LSTAT value  lower should be the value of MEDV as if a neighborhood consists of high proportion of lower class workers, the purchasing power of the people or the money they can spend on buying a house automatically decreases.
# 
# PTRATIO value:PTRATIO value represents the ratio of students to teachers in primary and secondary schools in the neighborhood. One teacher for 5 students is better than one teacher for 10 students. In this way, the students can have more interaction with the teacher and have better learning experience. Hence a lower value of PTRATIO is considered to be good as parents care lot about the education of their children and PTRATIO is certainly one of the factors which helps them in deciding the quality of education in the neighborhood. Thus, we can say that lower the value of PTRATIO, higher will be the MEDV value.   

# ----
# 
# ## Developing a Model
# In this second section of the project, you will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in your predictions.

# ### Implementation: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, you will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify your model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 
# 
# The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 is no better than a model that always predicts the *mean* of the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. _A model can be given a negative R<sup>2</sup> as well, which indicates that the model is **arbitrarily worse** than one that always predicts the mean of the target variable._
# 
# For the `performance_metric` function in the code cell below, you will need to implement the following:
# - Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
# - Assign the performance score to the `score` variable.

# In[19]:

# TODO: Import 'r2_score'
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true,y_predict)
    
    # Return the score
    return score


# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
# 
# | True Value | Prediction |
# | :-------------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
# 
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.

# In[20]:

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)


# * Would you consider this model to have successfully captured the variation of the target variable? 
# * Why or why not?
# 

# **Answer:**
# 
# The R2 score determines the proportion of variance in the dependent variable that is predictable from the independent variable. If the R2 score is closer to 1, it means that the model successfully captures the variation of target variable. So this model which has a coefficient of determination R2=0.923 successfully captured the variation of the target variable.

# ### Implementation: Shuffle and Split Data
# Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
# 
# For the code cell below, you will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.

# In[21]:

# TODO: Import 'train_test_split'
from sklearn.cross_validation import train_test_split
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size=0.20,random_state=15)

# Success
print "Training and testing split was successful."


# ### Question 3 - Training and Testing
# 
# * What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?
# 
# 

# **Answer: **
# 
# The benefit of splitting the dataset into some ratio of training and substes for an algorithm is that we can asses the quality and accuracy of the model. By doing this, we can actually how close are our predicitons to our actual outcome and if the predictions are incorrect, we can know how large is the prediction error.
# 
# If the whole dataset is used for training, overfitting takes place. In this case the model tends to memorize instead of learning the characteristcs. For a good fit to take place, the dataset needs to be split into training and testing data.Thus, splitting the data into some ratio of training and testing subjects leaves us with the opportunity of validating the model with new data(testing data) after training the algorithm.

# ----
# 
# ## Analyzing Model Performance
# In this third section of the project, you'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

# ### Learning Curves
# The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  
# 
# Run the code cell below and use these graphs to answer the following question.

# In[22]:

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)


# ### Question 4 - Learning the Data
# * Choose one of the graphs above and state the maximum depth for the model. 
# * What happens to the score of the training curve as more training points are added? What about the testing curve? 
# * Would having more training points benefit the model? 
# 
# 

# **Answer: **
# 
# -> Graph chosen: The second graph with max_depth=3 as the fitting is just right.
# 
# -> The score of the training curve decreases when more training points are added as it becomes difficult to fit the training data. The testing curve score increases and error reduces. As we keep increasing training points more and more, the training score decreases and testing score increases till some point after which they converge and become parallel to each other.
# 
# ->When both curves become parallel or flatten out, there is no use of adding more data points as there wont be any significant improvement.

# ### Complexity Curves
# The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  
# 
# ** Run the code cell below and use this graph to answer the following two questions Q5 and Q6. **

# In[23]:

vs.ModelComplexity(X_train, y_train)


# ### Question 5 - Bias-Variance Tradeoff
# * When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? 
# * How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?
# 
# 

# **Answer: **
# 
# -> When the model is trained with a maximum depth of 1, the model suffers from high bias as this is the case of underfitting.
# 
# ->When the model is trained with a maximum depth of 10, the model suffers from high variance as this is the case of overfitting.
# I can justify my conclusions of the above mentioned points with the help of the visual cues in the graph. As we can see, at the max depth=1, the difference between training score and validation score is very minimal and there is very less separation between two curves. This can be understood as oversimplification of the problem or use of minimal features which can also be termed as High bias. At the max depth=10, the training score curve and validation score curve have a large gap between them indicating high variance. This means that the model at max depth=10 is over complicating the problem and it is memorizing rather than learning from the characteristcs.

# ### Question 6 - Best-Guess Optimal Model
# * Which maximum depth do you think results in a model that best generalizes to unseen data? 
# * What intuition lead you to this answer?
# 
# 

# **Answer: **
# 
# -> In my opinion, the model with max-depth=3 is the optimal model. This model best generalizes to unseen data. I think this model is the best as we can see from the model complexxity graph that both training and validation curves are at smallest level of uncertainity and also the validation score is nearly at its highest. Although at max depth=4, the validation score may be better, the training score keeps on increasing from that point and the gap between two curves becomes visibly large. This intuition of mine lead to select model with max depth=3 over others.

# -----
# 
# ## Evaluating Model Performance
# In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

# ### Question 7 - Grid Search
# * What is the grid search technique?
# * How it can be applied to optimize a learning algorithm?
# 
# 

# **Answer: **
# 
# ->Sometimes we need to tune the hyper parameters of the model to get the optimal performance. For example: In SVM model, the hyper parameters are kernel and Gamma. To get the best combination of these two hyper parameters, we use the Grid search technique.
# 
# -> In the above mentioned example, we need to know what kernel to use along with what value of Gamma to get the most suitable SVM model for solving our problem. We may get the best model using a polynomial kernel with Gamma value of 5. We can obtain this information by looking at the corresponding grids of specific kernels and Gamma values. So the end goal of grid search is to provide us with the best combination of the parameters to work with the model. In this way, the grid search allows for better and faster experimentation with the parameters giving us the combination with which we can optimize a learning algorithm.

# ### Question 8 - Cross-Validation
# 
# * What is the k-fold cross-validation training technique? 
# 
# * What benefit does this technique provide for grid search when optimizing a model?
# 
# 
# 
# 

# **Answer: **
# 
# -> The common approach in machine learning is to divide the dataset into training and testing data. But we may not want to waste some of the data points as testing data. For this, we use K-fold cross validation is used. The 'K' here stands for K number of buckets. The model is then trained K times, each time with different bucket as testing set and remaining points as training set. Then the results obtained are averaged to get the final model.
# 
# -> The main drawback of the Grid search technique is that it only uses a specific subset of data for training or testing. This will result the Grid Search technique to perform tuning on specific subset of the data. This type of tuning will not help the model in prediction in case of some unseen data. K-fold cross validation prevents this in Grid search as all the data points are shuffled and this helps the grid search in tuning the parameters optimally. 

# ### Implementation: Fitting a Model
# Your final implementation requires that you bring everything together and train a model using the **decision tree algorithm**. To ensure that you are producing an optimized model, you will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.
# 
# In addition, you will find your implementation is using `ShuffleSplit()` for an alternative form of cross-validation (see the `'cv_sets'` variable). While it is not the K-Fold cross-validation technique you describe in **Question 8**, this type of cross-validation technique is just as useful!. The `ShuffleSplit()` implementation below will create 10 (`'n_splits'`) shuffled sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as the *validation set*. While you're working on your implementation, think about the contrasts and similarities it has to the K-fold cross-validation technique.
# 
# Please note that ShuffleSplit has different parameters in scikit-learn versions 0.17 and 0.18.
# For the `fit_model` function in the code cell below, you will need to implement the following:
# - Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
# - Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
#   - Assign the `GridSearchCV` object to the `'grid'` variable.

# In[24]:

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':range(1,11)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor,param_grid=params,scoring=scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions
# Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.

# ### Question 9 - Optimal Model
# 
# * What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?  
# 
# Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.

# In[25]:

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


# 
# **Answer: **
# 
# The maximum depth of the optimal depth is 4. In question 6, I answered that, for optimal model the max_depth should be 3. I answered based on my intuition of looking at the graph. This makes me understand that we cannot solely depend upon the visual representation of Complexity of model. In this type of cases techniques like Grid Search come handy.

# ### Question 10 - Predicting Selling Prices
# Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:
# 
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Neighborhood poverty level (as %) | 17% | 32% | 3% |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
# 
# * What price would you recommend each client sell his/her home at? 
# * Do these prices seem reasonable given the values for the respective features? 
# 
# **Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  Of the three clients, client 3 has has the biggest house, in the best public school neighborhood with the lowest poverty level; while client 2 has the smallest house, in a neighborhood with a relatively high poverty rate and not the best public schools.
# 
# Run the code block below to have your optimized model make predictions for each client's home.

# In[26]:

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)


# **Answer: **
# 
# These are the statistics I calculated in Data Exporation section earlier in the project: 
# 
# Minimum price: $105,000.00
# 
# Maximum price: $1,024,800.00
# 
# Mean price: $454,342.94
# 
# Median price $438,900.00
# 
# Standard deviation of prices: $165,171.13
# 
# 
# 
# -> The prices I would recommend each client to sell his/her home are: 
# 
# -> I would recommend the first client to sell the home at $416,550.  I would justify this with the following reasons:
#  The  mean  selling prices is $454,342 and the median price is $438,900. The price we predicted is pretty close to these prices and also one standard deviation from the mean. For a 5 room house in a neighborhood with Student-teacher ratio 15:1 , this is a reasonable price to sell.
# 
# -> I would recommend the second client to sell the home at $220,556. I would justify this with the following reasons:
# Although the price is far away from the mean and median price, we ignore this point as the house has only 4 rooms and it is in a neighborhood with Student-teacher ratio 22:1. The main reason due to which we say that this is the right price for this house is that the neighborhood poverty level is 32%. With this level of poverty in neighborhood, the house needs to priced with lower rate.
# 
# -> I would recommend the third client to sell the home at $919,609. I would justify this with the following reasons:
# This is a very high price for buying a house and its predicted cost is very near to the maximum price we calculated. But  considering the fact that it is an 8 room house in a neighborhood with poverty level of only 3% and with Student-teacher ratio 12:1 the price of the house is justified.

# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. 
# 
# **Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with respect to the data it's trained on.**

# In[27]:

vs.PredictTrials(features, prices, fit_model, client_data)


# ### Question 11 - Applicability
# 
# * In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.  
# 
# **Hint:** Take a look at the range in prices as calculated in the code snippet above. Some questions to answering:
# - How relevant today is data that was collected from 1978? How important is inflation?
# - Are the features present in the data sufficient to describe a home? Do you think factors like quality of apppliances in the home, square feet of the plot area, presence of pool or not etc should factor in?
# - Is the model robust enough to make consistent predictions?
# - Would data collected in an urban city like Boston be applicable in a rural city?
# - Is it fair to judge the price of an individual home based on the characteristics of the entire neighborhood?

# **Answer: **
# 
# -> The model constructed is should not be used in a real-world setting. This is very evident from the range in prices calculated above. 
# 
# -> The data collected from 1978 is not relevant to today's scenario. The factor of inflation plays a major role. The dollar value in 1978 is very different when compared to this time (2018). 
# 
# -> The features present in data are not at all sufficient to describe a home. The factors like square feet of the plot area, quality of appliances etc have a greater role in deciding the price along with current features. We can say that if these features are included in the data, there might some variations in the predicted values.
# 
# -> The present model is not robust enough. It cannot make consistent predictions as it is varying widely when run multiple times (as seen in the range in prices calculated).
# 
# -> Definitely not. The data collected in an urban city like Boston cannot be applicable to a rural city. The features or factors are quite different when we compare an urban city and a rural city. The cost of land, living and education are costly in an urban city than a rual city.
# 
# -> It is fair to judge the price of an individual home based on characteristics of the entireee neighborhood in some cases. This is applicable in cases like education- the whole neighborhood has access to schools present in that neighborhood. One such other characteristic is crime rate in the neighborhood. The house price will be inversely proportional to the crime rate of the neighborhood.

# ## References
# 

# http://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html
# 
# https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
# 
# http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# 
# 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



