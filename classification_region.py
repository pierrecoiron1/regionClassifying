#Import the Machine Learning Libraries
from tensorflow.keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import numpy as np

# Create a dataset for the XOR function
trainingSampleInput = np.array([
        [1,5],
        [2,4],
        [7,7],
        [4,6],
        [6,4],
        [6,9],
        [4,2],
        [8,6],
        [5,5],
        [3,8]])

#normalize the input data
trainingSampleInput=trainingSampleInput.astype(float)
trainingSampleInputNorm=np.true_divide(trainingSampleInput, 10)
    
trainingSampleOutput = np.array([0,0,0,0,0,1,1,1,1,1])
trainingSampleOutput = to_categorical(trainingSampleOutput)

trainingSampleOutput=trainingSampleOutput.astype(float)

#Build the Hidden Layer
network=models.Sequential()
models.add(Dense(50,
                input_dim=2,
                activation='sigmoid'))

models.add(Dense(1))
models.compile(
        loss='mean_squared_error',
        optimizer='rmsprop',
        metrics=['accuracy'])

#train it
history=models.fit(trainingSampleInputNorm,
                  trainingSampleOutput,
                  verbose=0,
                  epochs=10000)

print("Modeling Complete")

# Predict
pred = models.predict(trainingSampleInputNorm)

xTestInputQuantity=100 #both the x and the y inputs will have this number of datapoints
yTestInputQuantity=xTestInputQuantity

#define the test input matrix
testCordInputHeight=yTestInputQuantity*xTestInputQuantity
testCordInput=np.zeros([testCordInputHeight,2])

#create linespace
xTestInputLinespace=np.linspace(0,10,xTestInputQuantity) #linespace vector
yTestInputLinespace=np.linspace(0,10,yTestInputQuantity)

#debug\n",
print("Before we even enter the loop, the size is " + str(np.size(testCordInput)))#debug

#fill it
cordIndex=0
for yCord in range(yTestInputQuantity):
    for xCord in range(xTestInputQuantity):
        testCordInput[cordIndex,0]=yTestInputLinespace[yCord]
        testCordInput[cordIndex,1]=xTestInputLinespace[xCord]
        
        cordIndex=cordIndex+1
#debug
print(testCordInput)#debug
#normalize the data
testCordInputNorm=np.true_divide(testCordInput, 10)

#print(testCordInputNorm)
#feed the test coordinates into the predition
testingPredVector=models.predict(testCordInputNorm)
print(testingPredVector)
#round\n",
testingPredVector=np.rint(testingPredVector)
print(testingPredVector)
#transpose the column vector to a row vector so it can be used as a color label
testingPredVector=np.transpose(testingPredVector)
#testInputLinespace=np.linspace(0,10,testInputQuantity) #linespace vector
XtestInputSquare, YtestInputSquare=np.meshgrid(xTestInputLinespace, yTestInputLinespace)


#turning the predition vector into something that makes sense as a color vector for a line plot
colorVector=testingPredVector

#print(XtestInputSquare)
XtestInputSquare=XtestInputSquare.flatten()
YtestInputSquare=YtestInputSquare.flatten()
#print(np.shape(XtestInputSquare))\n",
print(XtestInputSquare)
print(YtestInputSquare)

#plot the graph\n",
import matplotlib.pyplot as plt

for colorVectorIndex in range(testCordInputHeight):
    if colorVector[colorVectorIndex] <0.5:
        colorVector[colorVectorIndex]=0
    else :
        colorVector[colorVectorIndex]=1

colorVector=colorVector.ravel()


#plt.scatter(XtestInputSquare, YtestInputSquare, s=1)\n",
plt.scatter(YtestInputSquare, XtestInputSquare, c=colorVector, s=1)
plt.show()
   
#probing
print(colorVector[900])