import tensorflow
import pandas as pd
print("Hello World!")
data = pd.read_csv('student-mat.csv', sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = "G3"
print(data.head())