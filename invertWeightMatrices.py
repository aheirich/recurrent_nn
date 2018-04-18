#!/usr/bin/python
#
# invertWeightMatrices.py
#

import sys
import numpy

import elman_shakespeare_2_1024 as M


def printMatrix(M, name, modfile, datfile):
  datfile.write('param ' + name + ': ')
  for i in range(M.shape[1]):
    datfile.write(' ' + str(i + 1))
  datfile.write(' :=\n')
  modfile.write('param ' + name + '{i in 1..' + str(M.shape[0]) + ', j in 1 .. ' + str(M.shape[1]) + '};\n')
  i = 1
  for row in M:
    datfile.write(str(i))
    for value in row: datfile.write(' ' + str(value))
    datfile.write('\n')
    i = i + 1
  datfile.write(';\n')

filename = "trained/elman_shakespeare_2_1024"
if len(sys.argv) > 1:
  filename = sys.argv[1]

modfile = open(filename + "_inverse.mod", "w")
datfile = open(filename + "_inverse.dat", "w")

print 'layer 0'
weight0Inverse = numpy.linalg.pinv(M.layer_0_weights)
printMatrix(weight0Inverse, 'layer_0_weights_feedforward_inverse', modfile, datfile)


for i in range(M.numHiddenLayers):
  layerId = i + 2
  print 'layer', layerId
  W = 'M.layer_' + str(layerId) + '_weights'
  numFeedForwardWeights = eval(W + '.shape[0] - ' + W + '.shape[1]')
  left = eval(W + '[0:' + str(numFeedForwardWeights) + ']')
  right = eval(W + '[' + str(numFeedForwardWeights) + ':]')
  leftInverse = numpy.linalg.pinv(left)
  printMatrix(leftInverse, 'layer_' + str(layerId) + '_weights_feedforward_inverse', modfile, datfile)
  printMatrix(right, 'layer_' + str(layerId) + '_weights_recurrent', modfile, datfile)

lastLayerId = M.numHiddenLayers + 2
print 'layer', lastLayerId
weightLastInverse = numpy.linalg.pinv(eval('M.layer_' + str(lastLayerId) + '_weights'))
printMatrix(weightLastInverse, 'layer_' + str(lastLayerId) + '_weights_feedforward_inverse', modfile, datfile)

modfile.close()
datfile.close()
