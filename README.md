# NLP_With_Disaster_Tweets
# Problem

Natural-Language-Processing-with-Disaster-Tweets

# Aim

Predict which Tweets are about real disasters and which ones are not

# Requirements  
Titanic datasets, Machine Learning model, Python, Sklearn

# Description  
	The dataset contains two file train.csv and test.csv. 
	Training data is use for traning the model while testing data is use for the submission on the Kaggle. 
	Training data contains 7613 rows and 5 columns. 
    The text column contains the tweets and target column contains the label telling which tweets is about disaster and which is not about disaster. 
    Machine Learning models used for training are Naive Bayes classifier, Random Forest Classifier, Logistic Regression, SVM and KNN. 
    Out of the all classifiers Naive Bayes classifier and Logistic Regression predicts the label with high accuracy as compared with the other classifiers.
    Highest accuracy reached is 81% using Naive Bayes and Logistic Regression Classifiers.
    CounterVectorizer from sklearn is used for data preprocessing. For training the model text column is used which contains the tweets.

# Results 
|Model|Accuracy| Precision| Recall| F1-Score|
|---|---|---|---|---|
|Logistic Regression|81|81|81|81|
|Random Forest|77|78|77|76|
|Support Vector Machine|78|78|78|78|
|Naive Bayes|81|81|81|81|
|KNN|73|76|73|70|
