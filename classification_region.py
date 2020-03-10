{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#Import the Machine Learning Libraries\n",
    "from tensorflow.keras import models\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras import optimizers\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a dataset for the XOR function\n",
    "trainingSampleInput = np.array([\n",
    "    [1,5],\n",
    "    [2,4],\n",
    "    [7,7],\n",
    "    [4,6],\n",
    "    [6,4],\n",
    "    [6,9],\n",
    "    [4,2],\n",
    "    [8,6],\n",
    "    [5,5],\n",
    "    [3,8]])\n",
    "    \n",
    "#normalize the input data\n",
    "trainingSampleInput=trainingSampleInput.astype(float)\n",
    "trainingSampleInputNorm=np.true_divide(trainingSampleInput, 10)\n",
    "    \n",
    "    \n",
    "trainingSampleOutput = np.array([0,0,0,0,0,1,1,1,1,1])\n",
    "trainingSampleOutput = to_categorical(trainingSampleOutput)\n",
    "\n",
    "trainingSampleOutput=trainingSampleOutput.astype(float)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Build the Hidden Layer\n",
    "network=models.Sequential()\n",
    "model.add(Dense(50,\n",
    "                input_dim=2,\n",
    "                activation='sigmoid'))\n",
    "\n",
    "#build the output layer\n",
    "model.add(Dense(1))\n",
    "model.compile(\n",
    "    loss='mean_squared_error',\n",
    "    optimizer='rmsprop',\n",
    "    metrics=['accuracy'])\n",
    "\n",
    "#train it\n",
    "history=model.fit(trainingSampleInputNorm,\n",
    "                  trainingSampleOutput,\n",
    "                  verbose=0,\n",
    "                  epochs=10000)\n",
    "\n",
    "print(\"Modeling Complete\")\n",
    "\n",
    "# Predict\n",
    "pred = model.predict(trainingSampleInputNorm)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "xTestInputQuantity=100 #both the x and the y inputs will have this number of datapoints\n",
    "yTestInputQuantity=xTestInputQuantity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#define the test input matrix\n",
    "testCordInputHeight=yTestInputQuantity*xTestInputQuantity\n",
    "testCordInput=np.zeros([testCordInputHeight,2])\n",
    "\n",
    "#create linespace\n",
    "xTestInputLinespace=np.linspace(0,10,xTestInputQuantity) #linespace vector\n",
    "yTestInputLinespace=np.linspace(0,10,yTestInputQuantity)\n",
    "\n",
    "#debug\n",
    "print(\"Before we even enter the loop, the size is \" + str(np.size(testCordInput)))#debug\n",
    "\n",
    "#fill it\n",
    "cordIndex=0\n",
    "for yCord in range(yTestInputQuantity):\n",
    "    #debug\n",
    "    #print(\"Y loop size: \" + str(np.size(testCordInput)))\n",
    "    \n",
    "    for xCord in range(xTestInputQuantity):\n",
    "        #debug\n",
    "        #print(\"Beginning of x loop size: \" + str(np.size(testCordInput)))#debug\n",
    "        #print(\"The value of yCord: \"+str(yCord))#debug\n",
    "        #print(\"The value of xCord: \"+str(xCord))#debug\n",
    "        \n",
    "        testCordInput[cordIndex,0]=yTestInputLinespace[yCord]\n",
    "        testCordInput[cordIndex,1]=xTestInputLinespace[xCord]\n",
    "        \n",
    "        cordIndex=cordIndex+1\n",
    "        #debug\n",
    "        #print(\"At completion of subloop size: \" + str(np.size(testCordInput)))#debug\n",
    "    \n",
    "#debug\n",
    "print(testCordInput)#debug\n",
    "    \n",
    "\n",
    "#normalize the data\n",
    "testCordInputNorm=np.true_divide(testCordInput, 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#print(testCordInputNorm)\n",
    "#feed the test coordinates into the predition\n",
    "testingPredVector=model.predict(testCordInputNorm)\n",
    "print(testingPredVector)\n",
    "#round\n",
    "testingPredVector=np.rint(testingPredVector)\n",
    "print(testingPredVector)\n",
    "#transpose the column vector to a row vector so it can be used as a color label\n",
    "#testingPredVector=np.transpose(testingPredVector)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#turning the predition vector into something that makes sense as a color vector for a line plot\n",
    "colorVector=testingPredVector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#testInputLinespace=np.linspace(0,10,testInputQuantity) #linespace vector\n",
    "XtestInputSquare, YtestInputSquare=np.meshgrid(xTestInputLinespace, yTestInputLinespace)\n",
    "#print(XtestInputSquare)\n",
    "XtestInputSquare=XtestInputSquare.flatten()\n",
    "YtestInputSquare=YtestInputSquare.flatten()\n",
    "#print(np.shape(XtestInputSquare))\n",
    "print(XtestInputSquare)\n",
    "print(YtestInputSquare)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "testingPredVector=np.transpose(testingPredVector)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#plot the graph\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "for colorVectorIndex in range(testCordInputHeight):\n",
    "    if colorVector[colorVectorIndex] <0.5:\n",
    "        colorVector[colorVectorIndex]=0\n",
    "    else :\n",
    "        colorVector[colorVectorIndex]=1\n",
    "\n",
    "colorVector=colorVector.ravel()\n",
    "\n",
    "\n",
    "#plt.scatter(XtestInputSquare, YtestInputSquare, s=1)\n",
    "plt.scatter(YtestInputSquare, XtestInputSquare, c=colorVector, s=1)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#probing\n",
    "print(colorVector[900])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.7 (tensorflow)",
   "language": "python",
   "name": "tensorflow"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
