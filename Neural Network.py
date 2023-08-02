import numpy as np
import csv
import random



trainData = []
data = []
with open(r'C:\Users\asus\OneDrive\Personal Projects\Machine Learning MNIST\two_variable_train2.csv', 'r') as f:
    reader = csv.reader(f, delimiter = 't')
    for i, line in enumerate(reader):
        if i > 0:
            if len(data) == 100:
                trainData.append(data)
                data = []
            data.append([float(a) for a in ''.join(line).split(',')])

testData = []
data = []
with open(r'C:\Users\asus\OneDrive\Personal Projects\Machine Learning MNIST\two_variable_test2.csv', 'r') as f:
    reader = csv.reader(f, delimiter = 't')
    for i, line in enumerate(reader):
        if i > 0:
            if len(data) == 100:
                testData.append(data)
                data = []
            data.append([float(a) for a in ''.join(line).split(',')])

print(trainData[0])


testData = []
with open(r'C:\Users\asus\OneDrive\Personal Projects\Machine Learning MNIST\two_variable_test2.csv', 'r') as f:
    reader = csv.reader(f, delimiter = 't')
    for i, line in enumerate(reader):
        if i > 0:
            testData.append([float(a) for a in ''.join(line).split(',')])

class Layer:

    def __init__(self, numNodeIn, numNodeOut):
        self.numNodeIn = numNodeIn
        self.numNodeOut = numNodeOut
        self.costGradientW = np.zeros((self.numNodeIn, self.numNodeOut), dtype = float)
        self.costGradientB = np.zeros(self.numNodeOut, dtype = float)
        self.weights = np.zeros((self.numNodeIn, self.numNodeOut), dtype = float)
        self.biases = np.zeros(self.numNodeOut, dtype = float)
        self.activatedInputs = list()
        self.InitialiseRandomWeights()

    def CalculateOutputs(self, inputs):
        self.activatedInputs = list()
        for nodeOut in range(0, self.numNodeOut):
            weightedInput = self.biases[nodeOut]
            for nodeIn in range(0, self.numNodeIn):
                weightedInput += inputs[nodeIn] * self.weights[nodeIn, nodeOut]
            
            self.activatedInputs.append(self.ActivationFunction(weightedInput))
        
        return self.activatedInputs
    
    def ActivationFunction(self, weightedInput):
        return 1 / (1 + np.exp(-weightedInput))
    
    def NodeCost(self, outputActivation, expectedOutput):
        error = outputActivation - expectedOutput
        return error * error

    def ApplyGradients(self, learnRate):
        for nodeOut in range(0, self.numNodeOut):
            self.biases[nodeOut] -= self.costGradientB[nodeOut] * learnRate
            for nodeIn in range(0, self.numNodeIn):
                self.weights[nodeIn, nodeOut] -= self.costGradientW[nodeIn, nodeOut] * learnRate
    
    def InitialiseRandomWeights(self): 
        for nodeIn in range(0, self.numNodeIn):
            for nodeOut in range(0, self.numNodeOut):
                randomValue = random.uniform(-1, 1) * 2 - 1
                self.weights[nodeIn, nodeOut] = randomValue / np.sqrt(self.numNodeIn)

class NeuralNetwork:
    
    def __init__(self, layerSizes):
        self.layerSizes = layerSizes
        self.layers = list()
        for i in range(0, len(layerSizes)-1):
            self.layers.append(Layer(layerSizes[i], layerSizes[i + 1]))
    
    # feeding layers into layers, to get the end output
    def CalculateOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.CalculateOutputs(inputs)
        return inputs
    
    def Classify(self, inputs):
        outputs = self.CalculateOutputs(inputs[1:])
        return outputs.index(max(outputs))
    
    def ClassifyArray(self, inputs):
        outputs = self.CalculateOutputs(inputs[1:])
        return outputs
    
    def Cost(self, dataPoint):
        # dataPoint here is just a line in the csv file
        outputs = self.CalculateOutputs(dataPoint[1:])
        outputLayer = self.layers[len(self.layers) - 1]
        cost = 0
        
        for node in range(0, len(outputs)):
            expectedOutList = [0 for i in range(0, len(outputs))]
            expectedOutList[int(dataPoint[0])] = 1
            
            cost += outputLayer.NodeCost(outputs[node], expectedOutList[node])
            
        
        return cost
    
    def TotalCost(self, data):
        totalCost = 0
        for point in data:
            totalCost += self.Cost(point)
        
        return totalCost / len(data)

    def Learn(self, trainingData, learnRate):
        h = 0.0001
        originalCost = self.TotalCost(trainingData)
        
        for layer in self.layers:
            # calculate cost gradient of current weights
            for nodeIn in range(0, layer.numNodeIn):
                for nodeOut in range(0, layer.numNodeOut):
                    layer.weights[nodeIn, nodeOut] += h
                    deltaCost = self.TotalCost(trainingData) - originalCost
                    layer.weights[nodeIn, nodeOut] -= h
                    layer.costGradientW[nodeIn, nodeOut] = deltaCost / h
            
            #calculate cost gradient of current biases
            for biasIndex in range(0, len(layer.biases)):
                
                layer.biases[biasIndex] += h
                deltaCost = self.TotalCost(trainingData) - originalCost
                layer.biases[biasIndex] -= h
                layer.costGradientB[biasIndex] = deltaCost / h

        for layer in self.layers:
            layer.ApplyGradients(learnRate)
        #print(self.TotalCost(trainingData))



a = NeuralNetwork((2,3,2))

for i, layer in enumerate(a.layers):
    if i == 0:
        layer.weights = np.array([np.array([6.19446192,  -3.18072636, -11.450854  ]),
 np.array([  0.38200956  , 0.0665295 ,   0.58574202])])
        layer.biases = np.array([-18.13506398 , 30.83661461 ,  2.99201654])
    if i == 1:
        layer.weights = np.array([np.array([ 10.34366413, -10.3657501 ]),
 np.array([ 10.26634288, -10.28691879]),
 np.array([  9.65295667 , -9.67435055])])
        layer.biases = np.array([-15.04458894 , 15.07559781])

'''
count = 0
for i in range(0, 50):
    for data in trainData:
        a.Learn(data, 0.25)
        count += 1
        if count % 100 == 0:
            print(count)
            print(a.TotalCost(data))
'''
for layer in a.layers:
    print(layer.weights)
    print(layer.biases)

correct = 0
for data in testData:
    correctAnswer = int(data[0])
    if correctAnswer == a.Classify(data):
        correct += 1
    else:
        pass
print(correct / len(testData))

print(a.TotalCost(testData))








