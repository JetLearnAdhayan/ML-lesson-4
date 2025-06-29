#logistic regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

data = pd.read_csv("titanic.csv")

print(data.isnull().sum())

data["Age"].fillna(data["Age"].median(skipna = True),inplace = True)

data["Embarked"].fillna(data["Embarked"].value_counts().idxmax(),inplace=True)

print(data.isnull().sum())

data["TravelAlone"] = np.where((data["SibSp"]+data["Parch"])>0,0,1)

data.drop("Cabin",axis=1,inplace = True)
data.drop("PassengerId",axis=1,inplace=True)
data.drop("Name",axis=1,inplace=True)
data.drop("Ticket", axis=1,inplace=True)
data.drop("SibSp",axis=1,inplace=True)
data.drop("Parch",axis=1,inplace=True)

print(data.head())

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
data["Sex"] = label_encoder.fit_transform(data["Sex"])
data["Embarked"] = label_encoder.fit_transform(data["Embarked"])
      
print(data.head())

#dataanalysis 

X = data[["Pclass","Sex","Age","Fare","Embarked","TravelAlone"]]
Y = data["Survived"]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

y_predict = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sb

matrix = confusion_matrix(Y_test,y_predict)
#heatmap to plot the map
sb.heatmap(matrix, annot = True, fmt="d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()






