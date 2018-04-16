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
  if VERBOSE: print 'x', x
  z1 = preactivation(x, M.layer_0_weights, None)
  if VERBOSE: print 'z1', z1
  z2 = recurrentPreactivation(z1, a2, M.layer_2_weights, M.layer_2_bias)
  if VERBOSE: print 'z2', z2[0:16]
  a2 = activation(z2)
  if VERBOSE: print 'a2', a2[0:16]
  z3 = recurrentPreactivation(a2, a3, M.layer_3_weights, M.layer_3_bias)
  if VERBOSE: print 'z3', z3[0:16]
  a3 = activation(z3)
  if VERBOSE: print 'a3', a3[0:16]
  z4 = preactivation(a3, numpy.transpose(M.layer_4_weights), M.layer_4_bias)
  if VERBOSE: print 'z4', z4
  return z4

if len(sys.argv) == 3:
  startChar = int(sys.argv[1])
  length = int(sys.argv[2])
else:
  print 'must provide starting character index, output length'
  print 'eg 43 2'
  sys.exit(1)

a0 = [0] * 65
a0[startChar] = 1
print 'starting character index', startChar

for i in range(length):
  print 'x', a0
  z4 = infer(a0)
  print 'y', z4
  maxx = -99999
  for j in range(len(z4)):
    if z4[j] > maxx: maxx = z4[j]
  for j in range(len(z4)):
    if z4[j] == maxx:
      a0[j] = 1
      print 'maximum character index', j
    else: a0[j] = 0

