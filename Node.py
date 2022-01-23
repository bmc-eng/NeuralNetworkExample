import math
import random

class Node:
    def __init__(self, name, inputs):
        self.name = name
        self.weights = {}
        self.weightAddress = []
        self.lastValue = 0.0

        for i in inputs:
            self.weights[i] = 0.0
            self.weightAddress.append(i)

    def getValue(self):
        # return the output of the node
        return self.lastValue

    def run(self, nodeValues):
        # this is the fnction which will run the values
        # throgh the weights and retrn the value for the node
        sum = 0.0
        for nAddress in self.weightAddress:
            sum += nodeValues[nAddress] * self.weights[nAddress]

        self.lastValue = 1 / (1 + math.exp(sum * -1))
        return self.lastValue

    def changeWeight(self, nodeAddress, value):
        # function to change the weight of a particular leg of a node
        self.weights[nodeAddress] = float(value)

    def teachNode(self, error, nodeValues, learningRate):
        # loop through al of the keys in the weights and change the values
        errorForNode = 0.0

        if self.name in error:
            errorForNode = self.lastValue * (1 - self.lastValue) * error['output'] * error[self.name]
        else:
            errorForNode = error['output']

        for key in self.weights:
            newWeight = self.weights[key] + (learningRate * errorForNode * nodeValues[key])
            self.weights[key] = newWeight
            if '.' in key:
                error[key] = newWeight
        
        return error

