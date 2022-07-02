import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 


breast_cancer=sklearn.datasets.load_breast_cancer()
# print(breast_cancer)
x=breast_cancer.data
y=breast_cancer.target
# print(x)
# print(y) 
# print(x.shape,y.shape)
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data['class']=y
# print(data)                 #difference between data and data describe?
# print(data.describe())
# print(data['class'].value_counts())
# print(breast_cancer.target_names)
# print(data.groupby('class').mean())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,stratify=y,random_state=1)
# print(y.shape,y_train.shape,y_test.shape) above arguments is to disturbute in equal mean and also to take less values for testing

classifier=LogisticRegression()
classifier.fit(x_train,y_train)
prediction_on_training_data=classifier.predict(x_train)
accuracy=accuracy_score(y_train,prediction_on_training_data)
print('accuracy on training data:',accuracy)
prediction_on_testing_data=classifier.predict(x_test)
accuracy2=accuracy_score(y_test,prediction_on_testing_data)
print('accuracy2:',accuracy2)

# KNN=KNeighborsClassifier(n_neighbors=3)
# KNN.fit(x_train,y_train)
# predict=KNN.predict(x_test)
# accuracy=accuracy_score(y_test,predict)
# print(accuracy)



#now detecting stage
input=input().split()
input_data=np.asarray(input)
resized_data=input_data.reshape(1,-1)
test_result=classifier.predict(resized_data)
if(test_result[0]==0):
    print('Type of the cancer is malignat')
if(test_result[0]==1):
    print('Type of cancer is Benign')


#using logistic regression we can get more accurate values than KNN process.