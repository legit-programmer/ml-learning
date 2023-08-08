import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

df = pd.read_csv('car.data')
processor = preprocessing.LabelEncoder()

buyingList = list(df['buying'])
buying = processor.fit_transform(buyingList)

maintList = list(df['maint'])
maint = processor.fit_transform(buyingList)

doorList = list(df['door'])
door = processor.fit_transform(buyingList)

personsList = list(df['persons'])
persons = processor.fit_transform(buyingList)

lug_bootList = list(df['lug_boot'])
lug_boot = processor.fit_transform(buyingList)

safetyList = list(df['safety'])
safety = processor.fit_transform(safetyList)

clsList = list(df['class'])
cls = processor.fit_transform(clsList)

predict = "class"
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
predicted = model.predict(X_test)

names = ['unacc', 'acc', 'good', 'vgood']
for i in range(len(X_test)):
    print(f"Predicted: {names[predicted[i]]} Data: {X_test[i]} Actual: {names[y_test[i]]}")
