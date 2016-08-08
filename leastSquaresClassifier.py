def generateClassfier(x,percentage,lmda):
	trainingdata1 ,testSet1= data1[:int(N*percentage)],data1[:int(N*(1-percentage))]
	trainingdata2 ,testSet2= data2[:int(N*percentage)],data2[:int(N*(1-percentage))]
	trainingdata3 ,testSet3= data3[:int(N*percentage)],data3[:int(N*(1-percentage))]
	finalTrainingSet=np.concatenate((trainingdata1,trainingdata2,trainingdata3), axis=0)
	finalTestSet=np.concatenate((testSet1,testSet2,testSet3), axis=0)
	X=finalTrainingSet[:,:5]
	z=np.dot(X.T,X)
	z=z+lmda *(np.identity(len(z)))
	z=np.linalg.inv(z)
	prod2=np.dot(X.T,np.concatenate((y1[:int(N*percentage)],y2[:int(N*percentage)],y3[:int(N*percentage)]), axis=0))
	theta=np.dot(z,prod2)
	testSet=np.dot(finalTestSet[:,:5],theta)
	trainingSet=np.dot(finalTrainingSet[:,:5],theta)
	maxArrayTest=np.argmax(testSet,axis=1)
	maxArrayTraining=np.argmax(trainingSet,axis=1)
	misclassificationErrorTestSet=0
	misclassificationErrorTrainingSet=0
	
	for i in range(0,(len(maxArrayTest))):
		if maxArrayTest[i]!=finalTestSet[[i],[5]]:
			misclassificationErrorTestSet +=1
	totalErrorTestSet=misclassificationErrorTestSet/(len(maxArrayTest))
	
	for j in range(0,(len(maxArrayTraining))):
		if maxArrayTraining[j]!=finalTrainingSet[[j],[5]]:
			misclassificationErrorTrainingSet +=1		
	totalErrorTrainingSet=misclassificationErrorTrainingSet/(len(maxArrayTraining))
	
	print(repr(x)+" \t\t"+repr(round(totalErrorTestSet*100,2))+" \t\t\t\t"+repr(round(totalErrorTrainingSet*100,2)))
	
import csv
import numpy as np
import math
with open('C:\iris.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
data1=[];
data2=[];
data3=[];
y1=[];y2=[];y3=[]
for i in range(0,(len(data))):
	if data[i][4]=='Iris-setosa':
		data[i][4]=1
		data[i].append(0);
		data1.append(data[i][0:6]);
		y1.append([1,0,0]);
	elif data[i][4]=='Iris-versicolor':
		data[i][4]=1
		data[i].append(1);
		data2.append(data[i][0:6]);
		y2.append([0,1,0]);
	elif data[i][4]=='Iris-virginica':
		data[i][4]=1
		data[i].append(2);
		data3.append(data[i][0:6])
		y3.append([0,0,1]);		
z = np.ones((len(data1),1))
data1 = np.array(data1, dtype=float)
data2 = np.array(data2, dtype=float)
data3 = np.array(data3, dtype=float)
N = len(data1)
percentage=[10,30,50]
collection = [0, math.pow(10,-2), math.pow(10,-4),math.pow(10,-8)]
for percent in percentage:
	for lmda in collection:
		print ("The observations for " +repr(percent) +"% of the training set for lambda of "+repr(lmda)+" :") 
		print("Trials\t\tTesting Error(%)\t\tTraining Error(%)")
		for x in list(range(10)):
			np.random.shuffle(data1)
			np.random.shuffle(data2)
			np.random.shuffle(data3)
			generateClassfier(x,percent/100,lmda)
		print("\n")