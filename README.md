# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and read the csv file.
2. Encoding the data and Import Decision tree classifier.
3. Fit the data in the model.
4. Find the MSE , r2 and the Predicted. 

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VIGNESH KUMARAN N S
RegisterNumber:  212222230171
*/

import pandas as pd
df=pd.read_csv("Salary.csv")
df.head(10)
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head(10)
x=df[['Position','Level']]
y=df['Salary']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
from sklearn import metrics
mse=metrics.mean_squared_error(Ytest,Ypred)
mse
r2=metrics.r2_score(Ytest,Ypred)
r2
dt.predict([[5,6]])
```

## Output:
#### df.head():
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393540/5775e2cb-cee2-4ee3-8ad9-3d641fafe991" width= "200">

#### df.info:
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393540/747a0a6f-3a04-4c96-aa5a-539760acb034" width= "200">

#### df.isnull().sum():
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393540/46f14eb3-7744-4003-bbc9-f7dbbe474d5d" width= "100">

#### Label Encoded Data:
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393540/eb29cb61-411d-4727-bdd7-dfa54a58e4cf" width= "200">

#### MSE:
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393540/d9ec0550-8504-4329-8260-c8127e7539d0" width= "100">

#### r2:
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393540/14d4f148-82ad-4b88-9015-ed187286c138" width= "150">

#### Prediction:
<img src= "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393540/f2330b3a-2009-4732-8782-8b35a0709bf4" width= "100">

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
