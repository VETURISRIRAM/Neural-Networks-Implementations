import numpy as np
import matplotlib.pyplot as plt
import random

def weightVectorGenerator():

    # Pick weight vectors randomly and uniformly
    w0 = random.uniform(-0.25, 0.25)
    w1 = random.uniform(-1.0, 1.0)
    w2 = random.uniform(-1.0, 1.0)

    weightVector = [w0, w1, w2]
    weightVector = np.array(weightVector)

    return weightVector

def weightVectorPrimeGenerator():

    # Pick weight vectors randomly and uniformly
    w0 = random.uniform(-1.0, 1.0)
    w1 = random.uniform(-1.0, 1.0)
    w2 = random.uniform(-1.0, 1.0)

    weightVector = [w0, w1, w2]
    weightVector = np.array(weightVector)

    return weightVector

def randomInputGenerator(n):
    S = []
    for x in range(n):
        S.append([random.uniform(-1, 1), random.uniform(-1, 1)])
    #S = np.array(S)

    return S

def differentiateSamples(S, weightVector):
    S1 = []
    S2 = []

    for x in S:
        result = np.matmul(np.array([1.0, x[0], x[1]]), np.transpose(weightVector))
        if result >= 0:
            S1.append(list(x))
        else:
            S2.append(list(x))

    return S1, S2

def misclassifications(S, SPrime, actions, weights, eta):
    omega = weights
    numMis = 0

    for x in S:
        if x in SPrime:
            pass
        else:
            X = [1] + x
            computation = np.array(weights).transpose() + np.array(X).transpose() * (actions[0] - actions[1]) * eta
            omega = np.array([a for a in list(computation)])
            numMis = numMis + 1
            weights = omega

    return  numMis, omega

def pta(S, S1, S2, S1Prime, S2Prime, weightVector, eta):
    epochs = []
    epoch = 1
    numMisclassifications = []
    passingWeights = np.array(weightVector)
    actions = [1, 0]
    S1Misclass = misclassifications(S1, S1Prime, actions, passingWeights, eta)
    actions = [0, 1]
    passingWeights = np.array(S1Misclass[1])
    S2Misclass = misclassifications(S2, S2Prime, actions, passingWeights, eta)
    t = S1Misclass[0] + S2Misclass[0]
    epochs.append(epoch)
    numMisclassifications.append(t)
    epoch += 1
    print("Total Misclassifications after epoch 1: {}".format(t))
    newWeights = S2Misclass[1]
    print("The Weight Vector after epoch 1 : {}".format(newWeights))

    while(t != 0):
        S1Prime = list()
        S2Prime = list()
        S1Prime, S2Prime = differentiateSamples(S, newWeights)
        weightsP = np.array(newWeights)
        actions = [1, 0]
        S1Misclass = misclassifications(S1, S1Prime, actions, weightsP, eta)
        actions = [0, 1]
        S2Misclass = misclassifications(S2, S2Prime, actions, S1Misclass)




def mainProgram(n):
    S = []
    S1 = []
    S2 = []

    weightVector = weightVectorGenerator()
    weightsPrime = weightVectorPrimeGenerator()
    print(weightVector)

    S = randomInputGenerator(n)

    S1, S2 = differentiateSamples(S, weightVector)
    S1Prime, S2Prime = differentiateSamples(S, weightsPrime)

    eta = 1
    pta(S, S1, S2, S1Prime, S2Prime, weightsPrime, eta)





if __name__=="__main__":

    # Set random seed
    random.seed(10)
    np.random.seed(10)

    n = 100
    mainProgram(n)


