import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# DATA 
df = pd.read_csv('car data.csv')

######################################################### Step-1 : Data preprocessing .

# Go through the data for information of each column.
# inf = df.info()
# print(df)
# print(inf)

# check if there is any null value .
NA = df.isnull().sum()
# print(NA)

# check if there is any duplicated value.
dup = df.duplicated().sum()
# print(dup)

# remove duplicate values
df = df.drop_duplicates()
# print(df.duplicated().sum())
 

######################################################### Step-2 : Data Visualisation .

import seaborn as sns


sns.pairplot( data=df )
plt.show()


sns.set_style("whitegrid", {
    'axes.facecolor': 'lightgrey',
    'figure.facecolor': 'gray',
     "grid.color": "black",
    "grid.linestyle": "-",
    "axes.edgecolor": "black"
})


sns.scatterplot(x = df['Driven_kms'] , y = df['Selling_Price'] , hue=df['Fuel_Type'])
plt.xlabel('Driven kilometers' , fontsize = 18)
plt.ylabel('Selling price' , fontsize = 18)
plt.title('Driven Kilometer VS Selling price ', fontsize = 22)
plt.show()

sns.scatterplot(x = df['Present_Price'] , y = df['Selling_Price'] , hue=df['Fuel_Type'])
plt.xlabel('Present Price' , fontsize = 18)
plt.ylabel('Selling price' , fontsize = 18)
plt.title('Present Price VS Selling price ', fontsize = 20)
plt.show()

sns.scatterplot(x = df['Selling_type'] , y = df['Selling_Price'] , hue=df['Fuel_Type'])
plt.xlabel('Selling type' , fontsize = 18)
plt.ylabel('Selling price' , fontsize = 18)
plt.title('Selling type VS Selling price ', fontsize = 20)
plt.show()

sns.scatterplot(x = df['Fuel_Type'] , y = df['Selling_Price'] , hue=df['Transmission'])
plt.xlabel('Fuel type' , fontsize = 18)
plt.ylabel('Selling price' , fontsize = 18)
plt.title('Fuel type VS Selling price ', fontsize = 20)
plt.show()

sns.scatterplot(x = df['Transmission'] , y = df['Selling_Price'] , hue=df['Selling_type'])
plt.xlabel('Transmission' , fontsize = 18)
plt.ylabel('Selling price' , fontsize = 18)
plt.title('Transmission VS Selling price ', fontsize = 20)
plt.legend(loc = 'upper center')
plt.show()


################################################################## STEP -3 : MODEL SELECTION.

# on the basis of ubove graphs , the data is linear . so we can put it in Linear Rergressioin Model for predicting the selling price of car.

################################################################## STEP - 4 : DESIGINING MODEL FOR PREDICTION OF CAR PRICES.

# Encode the object data .
# fuel type has 3 types : petrol , diesel , CNG.
# Selling type has 2 types : Dealer , individual
# Transmission has 2 types : Manual , Automatic .

from sklearn.preprocessing import OrdinalEncoder
df['Fuel_Type'] = OrdinalEncoder().fit_transform(df[['Fuel_Type']])
df['Selling_type'] = OrdinalEncoder().fit_transform(df[['Selling_type']])
df['Transmission'] = OrdinalEncoder().fit_transform(df[['Transmission']])
df['Year'] = OrdinalEncoder().fit_transform(df[['Year']])

# As for buying the car , car's facilities and feature are more important then name so drop the name.
df.drop('Car_Name' , axis=1 , inplace=True)

# scale the data .
from sklearn.preprocessing import MinMaxScaler
df['Selling_Price'] = MinMaxScaler().fit_transform(df[['Selling_Price']])
df['Present_Price'] = MinMaxScaler().fit_transform(df[['Present_Price']])
df['Driven_kms'] = MinMaxScaler().fit_transform(df[['Driven_kms']])



# for training and testig the data making feature(x) and target() values .
x = df[['Driven_kms','Fuel_Type','Present_Price','Selling_type','Transmission','Owner' , 'Year']]
y = df['Selling_Price']

# splitting the data into testing data and traing data set.
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y, test_size=0.2 ,random_state=0)


# making the model for predicting the price of car.
# As we have to predict the continous value(price ) , so we will use linear regression.

from sklearn.linear_model import LinearRegression
lg = LinearRegression()

# training the model .
lg.fit(x_train , y_train)

# making predictions.
pred = lg.predict(x_test)




##################################################################### STEP - 5 : CHECKING THE EFFICIENCY OF MODEL.


# for checking the efficiency of our model , we are checking the mean squared error and mean absolute errors.
from sklearn.metrics import mean_squared_error, mean_absolute_error 
mae = mean_absolute_error(pred , y_test)
mse = mean_squared_error(pred , y_test)
print()
print(" MEAN SQUARED ERROR :" , mse)
print(" MEAN ABSOLUTE ERROR : " ,mae)
print()


# Cross checking our model predictions.
from sklearn.model_selection import cross_val_score
me_scores = cross_val_score(lg , x, y, cv=5 , scoring='neg_mean_squared_error')
mse_score = -me_scores
print(" CROSS VALUE SCORES FOR MEAN SQUARED ERROR : " ,mse_score)

ma_scores = cross_val_score(lg , x, y, cv=5 , scoring='neg_mean_absolute_error')
mae_score = -ma_scores
print(" CROSS VALUE SCORES FOR MEAN ABSOLUTE ERROR : " ,mae_score)
print()

