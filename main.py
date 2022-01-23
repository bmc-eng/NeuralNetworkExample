import Network

def XORTable():
    inputValues = [[0,0],[0,1],[1,0],[1,1]]
    outputValues = [0,1,1,0]
    return inputValues, outputValues


def Test():
    # test the network and the run feature
    address = '1.1:I1:I2:b,2.1:I1:I2:1.1:b'
    weights = '1.1:0.0:0.0:0.0,2.1:0.0:0.0:0.0:0.0'

    net = Network.Network(address,2,weights)
    inputValues, outputValues = XORTable()
    net.trainNetwork(inputValues, outputValues, 10000)

if __name__ == '__main__':
    Test()
