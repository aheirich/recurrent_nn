#!/usr/bin/python

import sys
import math
import numpy
import model_ampl as M

VERBOSE = False

# recurrent state
a2 = [0.0] * M.layer_2_weights[1]
a3 = [0.0] * M.layer_3_weights[1]



def preactivation(a0, W0, b1):
  z1 = numpy.matmul(a0, W0)
  if b1 is not None:
    z1 = z1 + b1
  return z1


def activation(z):
  a = []
  for zz in z: a.append(math.tanh(zz))
  return a




def recurrentPreactivation(a1, a2, W, b2):
  za1 = numpy.matmul(a1, W[0:len(a1)])
  za2 = numpy.matmul(a2, W[len(a1):])
  z2 = za1 + za2
  if b2 is not None:
    z2 = z2 + b2
  return z2


def infer(x):
  global a2, a3
  z1 = preactivation(x, M.layer_0_weights, None)
  z2 = recurrentPreactivation(z1, a2, M.layer_2_weights, M.layer_2_bias)
  a2 = activation(z2)
  z3 = recurrentPreactivation(a2, a3, M.layer_3_weights, M.layer_3_bias)
  a3 = activation(z3)
  z4 = preactivation(a3, numpy.transpose(M.layer_4_weights), M.layer_4_bias)
  return z4



for line in sys.stdin:
  words = line.strip().split(' ')
  a0 = []
  for word in words: a0.append(float(word))
  print 'x', a0
  z4 = infer(a0)
  print 'y', z4
