"""
AdaBoost Classifier

- AdaBoost or Adaptive Boosting is one of the ensemble boosting classifier proposed by Yoav Freund and Robert Schapire in 1996.

- It combines multiple weak classifiers to increase the accuracy of classifiers.

- AdaBoost is an iterative ensemble method. AdaBoost classifier builds a strong classifier by combining multiple poorly performing classifiers so that you will get high accuracy strong classifier.

- The basic concept behind Adaboost is to set the weights of classifiers and training the data sample in each iteration such that it ensures the accurate predictions of unusual observations.

- Any machine learning algorithm can be used as base classifier if it accepts weights on the training set.

- AdaBoost should meet two conditions:

      - The classifier should be trained interactively on various weighed training examples.

      -In each iteration, it tries to provide an excellent fit for these examples by minimizing training error.

- To build a AdaBoost classifier, imagine that as a first base classifier we train a Decision Tree algorithm to make predictions on our training data.

- Now, following the methodology of AdaBoost, the weight of the misclassified training instances is increased.

- The second classifier is trained and acknowledges the updated weights and it repeats the procedure over and over again.

- At the end of every model prediction we end up boosting the weights of the misclassified instances so that the next model does a better job on them, and so on.

- AdaBoost adds predictors to the ensemble gradually making it better. The great disadvantage of this algorithm is that the model cannot be parallelized since each predictor can only be trained after the previous one has been trained and evaluated.
"""





import numpy as np # linear algebra
import pandas as pd # data processing



df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()

df.info()

# Declare feature vector and target variable
X = df[['x1','x2','x3','x4']]

X.head()

y = df['y']   # target

y.head()


######
######
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

y=le.fit_transform(y)


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#############
#############
# Build the AdaBoost model 

# Import the AdaBoost classifier
from sklearn.ensemble import AdaBoostClassifier


# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)

# Train Adaboost Classifer
model1 = abc.fit(X_train, y_train)


#Predict the response for test dataset
y_pred = model1.predict(X_test)



"""
Create Adaboost Classifier
The most important parameters are base_estimator, n_estimators and learning_rate.

base_estimator is the learning algorithm to use to train the weak models. This will almost always not needed to be changed because by far the most common learner to use with AdaBoost is a decision tree – this parameter’s default argument.

n_estimators is the number of models to iteratively train.

learning_rate is the contribution of each model to the weights and defaults to 1. Reducing the learning rate will mean the weights will be increased or decreased to a small degree, forcing the model train slower (but sometimes resulting in better performance scores).

loss is exclusive to AdaBoostRegressor and sets the loss function to use when updating weights. This defaults to a linear loss function however can be changed to square or exponential.

"""


########
########
# Evaluate Model 
# Let's estimate, how accurately the classifier or model can predict the type of cultivars.


#import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score


# calculate and print model accuracy
print("AdaBoost Classifier Model Accuracy:", accuracy_score(y_test, y_pred))

# In this case, we got an accuracy of XX.XX %, which will be considered as a good accuracy.



###########
###########
# Further evaluation with SVC base estimator 
# For further evaluation, we will use SVC as a base estimator as follows:

# load required classifer
from sklearn.ensemble import AdaBoostClassifier


# import Support Vector Classifier
from sklearn.svm import SVC


# import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score
svc=SVC(probability=True, kernel='linear')


# create adaboost classifer object
abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1, random_state=0)


# train adaboost classifer
model2 = abc.fit(X_train, y_train)


# predict the response for test dataset
y_pred = model2.predict(X_test)


# calculate and print model accuracy
print("Model Accuracy with SVC Base Estimator:",accuracy_score(y_test, y_pred))


# In this case, we have got a classification rate of 91.11%, which is considered as a very good accuracy.
# In this case, SVC Base Estimator is getting better accuracy then Decision tree Base Estimator.

"""
Advantages and disadvantages of AdaBoost 

The advantages are as follows:

- AdaBoost is easy to implement.

- It iteratively corrects the mistakes of the weak classifier and improves accuracy by combining weak learners.

- We can use many base classifiers with AdaBoost.

- AdaBoost is not prone to overfitting.

The disadvantages are as follows:

- AdaBoost is sensitive to noise data.

- It is highly affected by outliers because it tries to fit each point perfectly.

- AdaBoost is slower compared to XGBoost.
"""

