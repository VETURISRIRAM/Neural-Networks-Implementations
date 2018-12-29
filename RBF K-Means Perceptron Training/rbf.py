import numpy as np
import matplotlib.pyplot as plt
import random

def pointDistances(cluster1, cluster2, axis=1):
	d = cluster1-cluster2
	distances = np.linalg.norm(d, axis=axis)
	return distances

def kMeans(centers, classType):
	from copy import deepcopy

	previousCenters = np.zeros(centers.shape)
	classClusters = np.zeros(len(classType))
	error = pointDistances(centers, previousCenters, None)

	while error > 0:
		for point in range(len(classType)):
			distances = pointDistances(classType[point], centers)
			classClusters[point] = np.argmin(distances)

		previousCenters = deepcopy(centers)

		for clusterCenter in range(k):
			clusterPoints = [classType[j] for j in range(len(classType)) if classClusters[j] == clusterCenter]
			centers[clusterCenter] = np.mean(clusterPoints, axis=0)
		error = pointDistances(centers, previousCenters, None)

	return centers

def activationSignum(output):
	if output >= 0:
		return 1
	else:
		return -1

def misclassified(weights, bias, inputPattern, desiredOutput):
	mis = 0
	for input in range(len(inputPattern)):
		output = np.dot(weights.reshape(1, numCenters), np.asarray(inputPattern[input]).reshape(numCenters, 1)) + bias
		predictedOutput = activationSignum(output)
		if predictedOutput != desiredOutput[input]:
			mis += 1
	return mis

def rbf(input):
	import math
	return [math.exp(-input[x]**2) for x in range(len(input))]

sampleSize = 100
inputPattern = np.random.uniform(low=0, high=1, size=(sampleSize, 2))

d, posClass, negClass = list(), list(), list()

for inp in inputPattern:
	if (inp[1] < (((1/5) * np.sin(10 * inp[0])) + 0.3)) or (((inp[1] - 0.8) ** 2) + ((inp[0] - 0.5) ** 2) < ((0.15) ** 2)):
		posClass.append(inp)
		d.append(1)
	else:
		negClass.append(inp)
		d.append(-1)

if __name__=='__main__':
	numCenters = 20
	# numCenters = 4
	k = int(numCenters/2)

	posCenters = np.random.randint(low=0, high=len(posClass), size=k)
	negCenters = np.random.randint(low=0, high=len(negClass), size=k)

	positiveCenters = [posClass[i] for i in range(len(posCenters))]
	negativeCenters = [negClass[i] for i in range(len(negCenters))]

	centroids = positiveCenters + negativeCenters

	# Plot all the points in the positive class
	posClass = np.asarray(posClass)
	a = [point[0] for point in posClass]
	b = [point[1] for point in posClass]
	plt.scatter(a, b, c='orange')
	# Plot all the points in the negative class
	negClass = np.asarray(negClass)
	a = [point[0] for point in negClass]
	b = [point[1] for point in negClass]
	plt.scatter(a, b, c='yellow')
	# Plot all the positive centroids
	positiveCenters = np.asarray(positiveCenters)
	a = [point[0] for point in positiveCenters]
	b = [point[1] for point in positiveCenters]
	plt.scatter(a, b, c='black')
	# Plot all the negative centroids
	negativeCenters = np.asarray(negativeCenters)
	a = [point[0] for point in negativeCenters]
	b = [point[1] for point in negativeCenters]
	plt.scatter(a, b, c='green')
	plt.show()



	updatedPosCenters = kMeans(positiveCenters, posClass)
	updatedNegCenters = kMeans(negativeCenters, negClass)

	# Plot all the points in the positive class
	posClass = np.asarray(posClass)
	a = [point[0] for point in posClass]
	b = [point[1] for point in posClass]
	plt.scatter(a, b, c='orange')
	# Plot all the points in the negative class
	negClass = np.asarray(negClass)
	a = [point[0] for point in negClass]
	b = [point[1] for point in negClass]
	plt.scatter(a, b, c='yellow')
	# Plot all the positive centroids
	updatedPosCenters = np.asarray(updatedPosCenters)
	a = [point[0] for point in updatedPosCenters]
	b = [point[1] for point in updatedPosCenters]
	plt.scatter(a, b, c='black')
	# Plot all the negative centroids
	updatedNegCenters = np.asarray(updatedNegCenters)
	a = [point[0] for point in updatedNegCenters]
	b = [point[1] for point in updatedNegCenters]
	plt.scatter(a, b, c='green')
	plt.show()

	updatedCenters = np.concatenate((updatedPosCenters, updatedNegCenters))

	weights = np.random.uniform(-1, 1, size=numCenters)
	bias = np.random.uniform(-1, 1)
	eta = 0.1

	newPattern = []
	for input in range(len(inputPattern)):
		each = [pointDistances(inputPattern[input], updatedCenters[c], None) for c in range(len(updatedCenters))]
		newPattern.append(each)



	totalError = misclassified(weights, bias, newPattern, d)
	print(totalError)
	threshold = 10
	while totalError > threshold:
		for i in range(len(newPattern)):
			output = np.add(np.dot(weights.reshape(1, numCenters), np.asarray(newPattern[i]).reshape(numCenters, 1)), bias)
			predictedOutput = activationSignum(output)
			if predictedOutput != d[i]:
				pat = np.asarray(newPattern[i])
				diff = d[i]-predictedOutput
				update = pat*eta*diff
				weights = np.add(weights, update)
		totalError = misclassified(weights, bias, newPattern, d)
		print(totalError)

	xPlane = np.linspace(0.0, 1.0, 500)
	yPlane = np.linspace(0.0, 1.0, 500)
	decisionBoundary = list()

	for i in xPlane:
		print(i)
		for j in yPlane:
			coordinate = [i, j]
			# print(coordinate)
			newCoordinate = []
			for k in range(numCenters):
				newCoordinate.append(pointDistances(coordinate, updatedCenters[k], None))
			temp = 0
			for i in range(numCenters):
				temp += weights[i] * newCoordinate[i]
			g = temp + bias

			if -0.05 < g < 0.05:
				decisionBoundary.append(coordinate)


	fig, ax = plt.subplots(figsize=(10,10))
	posClass = np.asarray(posClass)
	a = [point[0] for point in posClass]
	b = [point[1] for point in posClass]
	plt.scatter(a, b, c = 'blue',s=1, label = 'Positive Class')
	negClass = np.asarray(negClass)
	a = [point[0] for point in negClass]
	b = [point[1] for point in negClass]
	plt.scatter(a, b, c = 'blue',s=1, label = 'Negative Class')
	decisionBoundary = np.asarray(decisionBoundary)
	a = [point[0] for point in decisionBoundary]
	b = [point[1] for point in decisionBoundary]
	plt.scatter(a, b, c = 'blue',s=1, label = 'Margin')
	plt.legend(loc = 'best')
	plt.show()
	print('Done')