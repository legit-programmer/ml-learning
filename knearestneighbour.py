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


print(safety)