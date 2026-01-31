from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

data={
    'experience':[1, 2, 3, 5, 7, 10],
    'promotion':[0, 0, 0, 1, 1, 1]
}

df=pd.DataFrame(data)
print(df)

x=df[['experience']]
y=df['promotion']
model=LogisticRegression()
model.fit(x,y)
pred=model.predict(x)
print("accuracy:",accuracy_score(y,pred))