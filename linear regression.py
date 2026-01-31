import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score



import pandas as pd
data={
    'area':[500,600,700,800,900,1000,1100,1200],
    'price':[25,35,45,55,65,75,90,100]

}
df=pd.DataFrame(data)
print(df)
x=df[['area']]
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("MSE",mean_squared_error(y_test,y_pred))