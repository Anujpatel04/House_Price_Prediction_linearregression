import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv(r"C:\Users\a\Desktop\NARESH-IT\MACHINE-LEARNING-NIT\SLR - House price prediction_trial\SLR - House price prediction\House_data.csv")

space=df['sqft_living']
price=df['price']

x=np.array(space).reshape(-1,1)
y=np.array(price)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

reg=LinearRegression()
reg.fit(x_train,y_train)

pickle.dump(reg,open('model.pkl','wb'))


