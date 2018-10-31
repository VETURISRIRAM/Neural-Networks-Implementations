import numpy as np
import matplotlib.pyplot as plt
import struct

def read_idx(filename):
    
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def stepActivationFunction(result, emptyArray):
   
    counter = 0
    for x in result:
        if x >= 0:
            emptyArray[counter] = 1.0
        else:
            emptyArray[counter] = 0.0
        counter += 1

    y = np.array(emptyArray)
    return y

def computeError(n, epoch, errors):
    
    for x in range(0, n):
        sample = train_data[x]
        trainLabel = train_labels[x]
        if sample.shape != (784, 1):
            sample.resize(784, 1)

        mulResult = np.matmul(w, sample)
        largestComponent = mulResult.argmax(axis=0)

        if largestComponent != trainLabel:
            errors[epoch] += 1
    finalError = errors[epoch]

    return finalError

def updateWeights(w, n, learningRate):
    
    for x in range(n):
        xi = train_data[x]
        xi.resize(784, 1)
        xit = np.transpose(xi)

        empty = np.empty([10, 1])
        y = stepActivationFunction(np.matmul(w, xi), empty)
        label = np.zeros((1,10)).T
        label[train_labels[x]] = 1

        w += learningRate*(np.matmul((label-y), xit))

def multiclassPTA(n, w, epoch, E, eta, errors):

    flag = True
    weightsVector = []

    while True:

        errors.append(epoch)
        # Compute the error for the epoch
        error = computeError(n, epoch, errors)
        errors[epoch] = error
        updateWeights(w, n, eta)
        epoch += 1
        print(errors[epoch-1]/n)
        if errors[epoch - 1] / n <= E:
            break

if __name__=="__main__":

    train_data = read_idx('train-images.idx3-ubyte')
    train_labels = read_idx('train-labels.idx1-ubyte')
    test_data = read_idx('t10k-images.idx3-ubyte')
    test_labels = read_idx('t10k-labels.idx1-ubyte')

    np.random.seed(10)
    w = np.random.uniform(-1.0, 1.0, (10, 784))
    initialweights = np.random.uniform(-1.0, 1.0, (10, 784))
    n = 50
    epoch = 0
    E = 0.0
    eta = 1.0
    errors = []
    multiclassPTA(n, w, epoch, E, eta, errors)


    print("Debug")
