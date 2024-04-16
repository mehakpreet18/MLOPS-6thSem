
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score, recall_score
import time


# print("Hello world")
df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

"""# Data Cleaning"""

df.drop("Unnamed: 133",inplace=True,axis=1)

df.head(2)

"""# Split data"""

x_train = df.drop('prognosis', axis = 1)
y_train = df['prognosis']

x_test = test_df.drop('prognosis',axis=1)
y_test = test_df['prognosis']

"""# Model train"""

Accuracy={}

# We will be using classification models as the data is discreate

"""### KNN"""

knn = KNeighborsClassifier()

start_time = time.time()
#training

knn.fit(x_train,y_train)

ti = time.time() - start_time

pred = knn.predict(x_test)
acc = knn.score(x_test,y_test)
f_s=f1_score(y_test, pred,average='macro')
p_s=precision_score(y_test, pred,average='macro')
r_s=recall_score(y_test, pred,average='macro')

Accuracy.update({"K-Nearest Neightbour":[acc,ti,f_s,p_s,r_s]})

import joblib

#Saving

joblib.dump(knn, 'knn_model.joblib')

"""# Model Loading"""

model = joblib.load('knn_model.joblib')

#Importing test data
test_df = pd.read_csv("Testing.csv")

x_test = test_df.drop('prognosis',axis=1)
y_test = test_df['prognosis']

fin_acc = model.score(x_test,y_test)
print(f"Accuracy: {fin_acc*100}%")

ran = x_test.head(1)


pre = model.predict(ran)

print(f'The person may be suffering from {pre[0]}')

