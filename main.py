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

data["TravellAlone"] = np.where((data["SibSp"]+data["Parch"])>0,0,1)

data.drop("Cabin",axis=1,inplace = True)
data.drop("PassengerId",axis=1,inplace=True)
data.drop("Name",axis=1,inplace=True)
data.drop("Ticket", axis=1,inplace=True)
data.drop("SibSp",axis=1,inplace=True)
data.drop("Parch",axis=1,inplace=True)

print(data.head())

      
