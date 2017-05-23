# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import cm
from scipy import optimize
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math


# Constants
bSize = 3

# Regularization Parameter:
Lambda = 0.0001


# New complete class, with changes:
class Neural_Network(object):
    def __init__(self, Lambda=0):
        # Define Hyperparameters
        self.inputLayerSize = 9
        self.outputLayerSize = 1
        self.hiddenLayerSize = 18

        # Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        # Regularization Parameter:
        self.Lambda = Lambda

    def forward(self, X):
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Gradient of sigmoid
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2) / X.shape[0] + (self.Lambda / 2) * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        # Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda * self.W2

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        # Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.Lambda * self.W1

        return dJdW1, dJdW2

    # Helper functions for interacting with other methods/classes
    def getParams(self):
        # Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        # Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)

        # Return the value we changed to zero:
        perturb[p] = 0

    # Return Params to original value:
    N.setParams(paramsInitial)

    return numgrad


##Need to modify trainer class a bit to check testing error during training:
class Trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)

        return cost, grad

    def train(self, trainX, trainY, testX, testY):
        # Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY

        self.testX = testX
        self.testY = testY

        # Make empty list to store training costs:
        self.J = []
        self.testJ = []

        params0 = self.N.getParams()

        options = {'maxiter': 1000, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


''' BOARD '''
# 1 - tic, 0 - empty, -1 - tac

def GenerateEmptyBoard():
    return [0]*bSize**2

def DisplayBoard(board):
    symb = ['_', 'X', 'O']
    for i in range(1, bSize**2+1):
        print(symb[board[i-1]], end='')
        if i % 3 != 0:
            print('|', end='')
        else:
            print()
    print()

def IsBoardComplete(board):
    for square in board:
        if not square:
            return False
    return True

def BoardGetEmptySqueres(board):
    emptySquares = []
    for index in range(len(board)):
        if not board[index]:
            emptySquares.append(index)
    return emptySquares

def IsWin(board):
    h = [sum(board[:3]), sum(board[3:6]), sum(board[6:])]
    v = [[board[i], board[i + bSize], board[i + 2 * bSize]] for i in range(bSize)]
    v = [sum(v[0]), sum(v[1]), sum(v[2])]
    d = [board[0] + board[4] + board[8], board[2] + board[4] + board[6]]
    lines = h + v + d
    if 3 in lines:
        return 1
    elif -3 in lines:
        return -1
    else:
        return 0

def GetWinner(arg):
    if arg == 1:
        return "'X'"
    elif arg == -1:
        return "'O'"
    else:
        return "None"


''' DATA (input/training/testing) '''

fin = open("training_sets.txt")
X = []
y = []
for line in fin.readlines():
    line = list(map(int, line.split()))
    X.append(line[:9])
    y.append([line[9]/8])
X = np.array(X)
y = np.array(y, dtype=float)
# print(X)
# print(y)

# Training Data:
trainX = X
trainY = y

# Testing Data:
testX = X
testY = y


''' OPTIMISATION '''

# Train network with new data:
NN = Neural_Network(Lambda=0.0001)

# Make sure our gradients our correct after making changes:
numgrad = computeNumericalGradient(NN, X, y)
grad = NN.computeGradients(X, y)

# Should be less than 1e-8:
print("Gradient error:", LA.norm(grad - numgrad) / LA.norm(grad + numgrad))

T = Trainer(NN)
T.train(X, y, testX, testY)


''' CHECKING NN's INTELLIGENCE '''

#curpos = [0,  0,  0,
#          1,  1,  0,
#         -1,  0,  0]
curpos = [0, 0, 0,
          0, 0, 0,
          0, 0, 0]
board = curpos
print('How to move:')
print('1|2|3\n4|5|6\n7|8|9\n')
print('Initial position:')
DisplayBoard(board)

movenum = sum(abs(board[i]) for i in range(len(board)))
while not IsBoardComplete(board) and not IsWin(board):
    if movenum % 2 == 0:
        player_move = int(input("It's your turn: ")) - 1
        board[player_move] = 1
    else:
        NNmove = round(float(NN.forward(board)*9))
        if board[NNmove]:
            NNmove = random.choice(BoardGetEmptySqueres(board))
        board[NNmove] = -1
        print("NN's move:")
    DisplayBoard(board)
    movenum += 1
else:
    print(GetWinner(IsWin(board)) + " wins")

# Tip:
    #  Подсказка. Сыграйте сами с собой несколько примерных партий, записывая последовательности ходов.
    # Обучите нейросеть, задав все ходы-ответы ноликами. Далее пытайтесь играть с нейросетью, если она
    # будет выдавать неверный (или невозможный) ответ, сделайте ход за нее и включите этот пример в
    # обучающую выборку, продолжите обучение.
