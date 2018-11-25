#%%
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

#%%
data = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, :48]
y = data[:, -1]

#%%
Xtrain = X[:-100,]
ytrain = y[:-100,]
Xtest = X[-100:,]
ytest = y[-100:,]

#%%
model = MultinomialNB()
model.fit(Xtrain, ytrain)
print("\nAccuracy for NB: ",model.score(Xtest, ytest))

#%%
from sklearn.ensemble import AdaBoostClassifier

#%%
model = AdaBoostClassifier()
model.fit(Xtrain, ytrain)
print("Accuracy for Adaboost is: ", model.score(Xtest, ytest))

