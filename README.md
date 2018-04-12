#!/usr/bin/python

import sys
import model_ampl as M

print 'layer_0_weights', M.layer_0_weights.shape
print 'layer_2_weights', M.layer_2_weights.shape
print 'layer_3_weights', M.layer_3_weights.shape
print 'layer_4_weights', M.layer_4_weights.shape



def Relu(x):
  return max(x, 0)

def network_inference(a0):
  z1 = []
  a1 = []
  layer1Width = M.layer_0_weights.shape[1] # 64

  for i in range(layer1Width):
    sum = 0
    for j in range(M.layer_0_weights.shape[0]):
      sum = sum + a0[j] * M.layer_0_weights[i][j]
    z1.append(sum)
    a1.append(Relu(sum))
    
  z2 = []
  a2 = []
  sum = 0
  layer2Width = M.layer_2_weights.shape[1]
  for i in range(layer2Width):
    for j in range(layer1Width):
      sum = sum + a1[j] * M.layer_2_weights[i][j]
    for j in range(1 + layer1Width, M.layer_2_weights.shape[0]):
      sum = sum + a2[j - layer1Width * M.layer_2_weights[j][i]
    z2.append(sum)
    a2.append(Relu(sum))




with line in sys.stdin:
  line = line.strip()
  words = line.split(' ')
  a0 = [for w in words: float(w)]
  print "a0:", a0
  z4 = network_inference(a0)
  print "z4:", z4
