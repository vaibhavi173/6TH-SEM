import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('data.csv')
print("The first 5 values of data is :\n",data.head())

X = data.iloc[:,:-1]
print("\nThe First 5 values of train data is\n",X.head())
y = data.iloc[:,-1]
print("\nThe first 5 values of Train output is\n",y.head())

le_sky = LabelEncoder()
X.sky = le_outlook.fit_transform(X.sky)
le_airtemp = LabelEncoder()
X.airtemp = le_airtemp.fit_transform(X.airtemp)
le_humidity = LabelEncoder()
X.humidity = le_humidity.fit_transform(X.humidity)
le_wind = LabelEncoder()
X.wind = le_wind.fit_transform(X.wind)
le_water = LabelEncoder()
X.water = le_water.fit_transform(X.water)
le_forcast = LabelEncoder()
X.forcast = le_forcast.fit_transform(X.forcast)

print("\nNow the Train data is :\n",X.head())
le_enjoysport = LabelEncoder()
y = le_enjoysport.fit_transform(y)
print("\nNow the Train output is\n",y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(classifier.predict(X_test),y_test))
