import random
import math
import Node

class Network:

    def __init__(self, networkMap, numberOfInputs, initialWeights):
        #This is where the network is chopped up into the various parts. The format is , for new nodes . for layers I for inputs to connect the nodes together
        # 1.1:I1:I2:b,2.1:I1:I2:1.1:b

        print("Initializing")
        self.layers=[]
        self.nodes = {}
        self.nodeValues = {}
        self.inputs = []
        self.learnRate = 0.4

        # chop the network string into componets to configure the network
        nodeInput = networkMap.split(',')

        for node in nodeInput:
            # Each individual node - split by : to get the inputs into the node (this allows other nodes as well)
            section = node.split(':')
            name = section[0]

            # Split the inputs for the node into an array
            nIn = []
            for i in range(1,len(section)):
                nIn.append(section[i])

            #Create the node object and add the node to the node database
            self.nodes[name] = Node.Node(name, nIn)

            #Set the node outputs to 0
            self.nodeValues[name] = 0.0

            # Add the reference to the correct layer
            layerNum = name.split('.')

            if len(self.layers) == layerNum [0]:
                self.layers[len(self.layers)].append(name)
            else:
                newLayer = []
                newLayer.append(name)
                self.layers.append(newLayer)

        #setup the inputs to the node database
        for i in range (1,numberOfInputs + 1):
            self.nodeValues['I' + str(i)] = 0.0

        self.nodeValues['b'] = 0.0

        self.setWeights(initialWeights, networkMap)
        print("Network setup complete!")

    def trainNetwork(self, inputs, answers, numberOfIterations):
        # Function to start to train the networks. Accepts an array of inputs and and array of answers to the training sets. Set numberOfInterations to prevent an infinite loop

        count = 0
        continueToLoop = True
        
        #Test to see if correct
        isCorrect = []
        for i in range(len(answers)):
            isCorrect.append(False)

        print("Looping...")

        #Loop through the network with the inputs, answers and configure the weights of the network
        while continueToLoop:

            for i in range(len(answers)):
                #This is 1 loop through the input array
                networkOut = self.runNetwork(inputs[i])

                #Check to see if the network has learned the correct answer
                if (answers[i] == 1 and networkOut > 0.9) or (answers[i] == 0 and networkOut < 0.1):
                    isCorrect[i] = True
                
                print('Epoch: %s Count: %s : Target %s Actual: %s' % (count, i + 1, answers[i], networkOut))

                error = {}

                # Formula for the error of the output layer
                # (target - actual) * (actual * (1-actual))

                error['output'] = (answers[i] - networkOut) * ((1-networkOut) * networkOut)

                # Backpropogate the network and adjust the weights
                for j in range(len(self.layers),0,-1):
                    for k in range(len(self.layers[j-1]),0,-1):
                        n = self.nodes[self.layers[j-1][k-1]]
                        error = n.teachNode(error,self.nodeValues,2.0)
                
                count += 1

                # Check if the network is now correctly predicting the output based on the inputs
                isEnd = True
                for test in isCorrect:
                    if test == False:
                        isEnd = False
                        break

                if count >= numberOfIterations or isEnd:
                    # End of the training
                    print("Done: Completed learning in %s cycles" % count)
                    continueToLoop = False

    def runNetwork(self, inputs):
        # used to run the network for one run/ instance
        # returns the results of the output node

        # enter the inputes into the node database
        for i in range(1, len(inputs)+1):
            self.nodeValues["I" + str(i)] = inputs[i-1]

        if 'b' in self.nodeValues:
            self.nodeValues['b'] = 1.0

        # calculate the values for each payer of the network
        for layer in self.layers:
            for n in layer:
                self.nodeValues[n] = self.nodes[n].run(self.nodeValues)
        
        #return the value in the last layer of the node
        topLayer = len(self.layers)
        return self.nodes[self.layers[topLayer-1][0]].lastValue

    def setWeights(self, testWeights, networkMap):
        # Set the initial weights of each node
        # split the testWeights staring into the correct weighting
        nodeInput = networkMap.split(',')
        weightInput = testWeights.split(',')

        for i in range(0,len(nodeInput)):
            address = nodeInput[i].split(':')
            weight = weightInput[i].split(':')
            n = self.nodes[address[0]]
            for j in range(1,len(address)):
                n.changeWeight(address[j], weight[j])



        