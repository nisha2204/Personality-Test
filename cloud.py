import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

def dct(in1,in2,in3,in4,in5):
	data=pd.read_csv('./dataset.csv')
	#data.head()
	x=data.values[:,0:5]
	y=data.values[:,5]
	#y

	#print ("Dataset Length:: ", len(balance_data))
	#print ("Dataset Shape:: ", balance_data.shape)
   

	#print(balance_data.head(5))
	#print(X)
	#print(Y)
	#X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3)

	#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
	#max_depth=10, min_samples_leaf=15)

	#clf_entropy.fit(X_train, y_train)
	X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.2)
	model=DecisionTreeClassifier()
	model.fit(x,y)
	p=model.predict([[int(in1),int(in2),int(in3),int(in4),int(in5)]])
	return str(p[0])

	#x = clf_entropy.predict(int(in1),int(in2),int(in3),int(in4),int(in5))
	#print(x)
	#y_pred_en = clf_entropy.predict(X_test)
	#y_pred_en



if __name__=='__main__':
	dct()
