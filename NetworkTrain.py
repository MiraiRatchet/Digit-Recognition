from keras.datasets import mnist
import numpy as np
import random
from tqdm import tqdm
import os
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, denominator):
    for i in range(layerLength[-1]):
        denominator += np.exp(x[i])
    return np.exp(x)/denominator


def dsigma(x):
    return sigmoid(x)*(1-sigmoid(x))


def funcz(W, a, b):
    return np.add(np.dot(W, a), b)


def goForward(x_train, y_train):
    inputLayer = [[] for i in range(layers)]
    inputLayer[0] = np.array(x_train).flatten()
    trueVector = np.array([0 for i in range(layerLength[-1])])
    trueVector[y_train] = 1
    z = [0 for i in range(layers-1)]
    denominator = 0
    for i in range(layers-1):
        z[i] = funcz(weightsMatrix[i], inputLayer[i], biasVector[i])
        inputLayer[i+1] = sigmoid(z[i])
    inputLayer[-1] = softmax(z[-1], denominator)
    return inputLayer, trueVector, z


def epochtrain(x_train, y_train):
    global weightsMatrix, biasVector
    inputLayer, trueVector, z = goForward(x_train, y_train)
    localGradient = [[] for i in range(layers-1)]
    changeWeights = [[] for i in range(layers-1)]
    localGradient[-1] = inputLayer[-1]-trueVector
    localSigma = [[0 for i in range(layerLength[k+1])]
                  for k in range(layers-2)]
    for j in range(layers-2, -1, -1):
        changeWeights[j] = np.array([learningRate*localGradient[j]*inputLayer[j][i]
                                     for i in range(layerLength[j])])
        if (j == 0):
            break
        WmatrTemp = weightsMatrix[j].T
        for k in range(layerLength[j]):
            sum = 0
            for i in range(layerLength[j+1]):
                sum += localGradient[j][i]*WmatrTemp[k][i]
            localSigma[j-1][k] = sum
        localGradient[j-1] = localSigma[j-1]*dsigma(z[j-1])
    
    for i in range(layers-1):
        biasVector[i] -= learningRate*localGradient[i]
        weightsMatrix[i] -= changeWeights[i].T


def test(x_test, y_test):
    inputLayer = goForward(x_test, y_test)[0]
    if (np.argmax(inputLayer[-1]) == y_test):
        return 1
    return 0


def train():
    print('Training in progress...')
    for e in range(epoches):
        epochTest = 0
        print('Epoch', e+1)
        for i in tqdm(range(y_train.size)):
            epochtrain(x_train[i], y_train[i])
        print('Testing...')
        for i in range(y_test.size):
            epochTest += test(x_test[i], y_test[i])
        accuracyPercent = epochTest/y_test.size
        print('Epoch accuracy test: ', f'{accuracyPercent:.0%}')


def writeResults():
    if (not os.path.isdir(dir)):
        try:
            os.mkdir(dir)
            print("Directory '%s' created" % dir)
        except OSError as error:
            print(error)

    if (os.path.isdir(dir)):
        with open(os.path.join(dir, 'Biases.dat'), 'wb') as f:
            pickle.dump(biasVector, f)
        with open(os.path.join(dir, 'Weights.dat'), 'wb') as f:
            pickle.dump(weightsMatrix, f)
        with open(os.path.join(dir, 'NetworkStructure.txt'), 'w') as f:
            f.write(str(learningRate)+'\n')
            f.write(str(layerLength))


def readOptions():
    with open(os.path.join(dir, 'Options.txt'), 'r') as f:
        f.readline()
        hiddenLayers = int(f.readline())
        f.readline()
        layers = hiddenLayers+2
        layerLength = [0 for i in range(layers)]
        for i in range(hiddenLayers):
            layerLength[i+1] = int(f.readline())
        f.readline()
        epoches = int(f.readline())
        f.readline()
        learningRate = float(f.readline())
    layerLength[0] = 784
    layerLength[-1] = 10
    return layers, layerLength, learningRate, epoches


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
dir = os.path.dirname(os.path.abspath(__file__))
layers, layerLength, learningRate, epoches = readOptions()
weightsMatrix = [[] for i in range(layers-1)]
biasVector = [[] for i in range(layers-1)]
for i in range(len(layerLength)-1):
    biasVector[i] = np.array([random.uniform(0, 1)
                             for x in range(layerLength[i+1])])
    weightsMatrix[i] = np.array([[random.uniform(-1, 1) for x in range(layerLength[i])]
                                 for y in range(layerLength[i+1])])
train()
folderName = 'DigitNeuralNetworkData'
dir = os.path.join(dir, folderName)
writeResults()
