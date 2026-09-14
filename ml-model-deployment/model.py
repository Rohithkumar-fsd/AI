# print("heello")

import pandas as pd
import numpy as np

df=pd.read_csv("BankNote_Authentication.csv")

# print(df.head(5))

X=df.iloc[:,:-1]

Y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier()

classifier.fit(X_train,Y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score

score=accuracy_score(Y_test,y_pred)

print(score)


import pickle

pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)

pickle_out.close()