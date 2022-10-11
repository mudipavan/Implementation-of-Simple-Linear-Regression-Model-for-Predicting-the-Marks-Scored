# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
 1.Import the needed packages 2.Assigning hours To X and Scores to Y 3.Plot the scatter plot 4.Use mse,rmse,mae formmula to find.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: mudi.pavan
RegisterNumber:  212221230067

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/Placement_Data.csv')
print(dataset.iloc[3])

print(dataset.iloc[0:4])

print(dataset.iloc[:,1:3])

#implement a simple regression model for predicting the marks scored by students
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')

#implement a simple regression model for predicting the marks scored by students
#assigning hours to X& Scores to Y
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title("Traning set (H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,reg.predict(X_test),color="pink")
plt.title("Test set (H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("MES = ",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
 ![image](https://user-images.githubusercontent.com/94828517/194994493-2bbf2858-9e63-4032-a1eb-38182286217d.png)
 ![image](https://user-images.githubusercontent.com/94828517/194994583-be551f2e-ef97-4fbf-b3c4-ca08ce7e04ad.png)
 ![image](https://user-images.githubusercontent.com/94828517/194994661-fc26e56b-2a95-45bf-b046-52f45c163d8a.png)
 ![image](https://user-images.githubusercontent.com/94828517/194994691-84f9fcae-8bf0-4d17-9591-0fc5ce9f1e83.png)
 ![image](https://user-images.githubusercontent.com/94828517/194994737-05147ae0-f74a-40ea-bb3b-2791f72a5160.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
