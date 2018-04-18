#
# logToNumpy.py
#
# Takes a log produced by PRINT_MODEL.lua and produces numpy source
#

import sys
import numpy

"""
  How to invert the elman_shakespeare_2_1024 recurrent network
  a2(t) means a2 at time t
  
  Given z4
  1) Find a3(1) = w4^-1 z4
  
  Given a3(1) [1024], find a2(1)
  
  a3(1) = tanh( A a3(0) + B a2(1) + bias3 )
  ..
  atanh(a3(1)) = A a3(0) + B a2(1) + bias3
  ..
  2) a2(1) = B^-1 ( atanh( a3(1) ) - A a3(0) - bias3 ) ***
  
  Where A is the recurrent dense connections in weights_3.
  And B is the non recurrent dense connections from layer 2 in weights_3.
  [1024] = [1024] - [1024]
  
  One equation in two unknowns, a2(1) and a3(0)
  
  Given a2(1), or constraint
  3) a1(1) = D^-1 ( atanh( a2(1) ) - C a2(0) - bias2 ) ***
  
  One equation in two unknowns, a1(1) and a2(0)
  
  Given a1, or constraint
  4) Find a0(1) = W1^-1 a1(1)
"""


def readFormat(file):
  line = ''
  while '{' not in line:
    line = file.readline().strip()
  format = []
  while '}' not in line:
    line = file.readline().strip()
    if ':' in line:
      words = line.split(' ')
      dimensions = words[-1]
      sizes = dimensions.split('x')
      format.append(sizes)
  return format


def readToStart2D(file):
  line = ''
  words = []
  while len(words) < 3:
    while not line.startswith('p['):
      line = file.readline().strip()
      line = line.replace('\t', ' ')
      line = line.replace('  ', ' ')
    words = line.split(' ')
    if len(words) < 3: line = ''
  return words


def readArray2D(file, dimensions):
  rows = int(dimensions[0])
  columns = int(dimensions[1])
  readToStart2D(file)
  readColumn = 0
  array = []
  words = []
  while readColumn < columns:
    for row in range(rows):
      line = file.readline().strip()
      line = line.replace('\t', ' ')
      line = line.replace('  ', ' ')
      words = line.split(' ')
      if len(array) < rows: array.append([])
      for word in words: array[row].append(word)
    readColumn = readColumn + len(words)
    file.readline()
    file.readline()
  return array


def readToStart1D(file):
  line = ''
  words = []
  while len(words) < 2:
    while not line.startswith('p['):
      line = file.readline().strip()
      line = line.replace('\t', ' ')
      line = line.replace('  ', ' ')
    words = line.split(' ')
    if len(words) < 2: line = ''
  return words


def readArray1D(file, dimensions):
  rows = int(dimensions[0])
  words = readToStart1D(file)
  multiplier = 1
  numValues = rows
  array = []
  if len(words) == 6 and words[5] == '*':
    multiplier = float(words[4])
  elif len(words) == 6:
    array = [float(words[5])]
    numValues = numValues - 1
  for row in range(numValues):
    line = file.readline().strip()
    value = float(line)
    array.append(value * multiplier)
  return array


def readArray(file, dimensions):
  if len(dimensions) == 1: return numpy.array(readArray1D(file, dimensions))
  if len(dimensions) == 2: return numpy.array(readArray2D(file, dimensions))


def printArray2D(array, dimensions, name, modfile, datfile, pyfile, rows, columns):
  modfile.write('\n')
  datfile.write('\n')
  modfile.write('param ' + name + '{i in 1..' + rows + ', j in 1..' + columns + '};\n')
  datfile.write('param ' + name + ': ')
  pyfile.write(name + ' = numpy.array([\\\n')
  rows = int(dimensions[0])
  columns = int(dimensions[1])
  line = ''
  for column in range(columns):
    line = line + str(column + 1) + ' '
  line = line + ':= \n'
  datfile.write(line)
  for row in range(rows):
    line = str(row + 1) + ' '
    pyline = '[ '
    for column in range(columns):
      line = line + str(array[row][column]) + ' '
      pyline = pyline + str(array[row][column]) + ', '
    datfile.write(line + '\n')
    pyfile.write(pyline + '],\\\n')
  datfile.write(';\n')
  datfile.write('\n')
  pyfile.write('])\n\n')


def printArray1D(array, dimensions, name, modfile, datfile, pyfile, rows):
  modfile.write('\n')
  datfile.write('\n')
  modfile.write('param ' + name + '{i in 1..' + rows + '};\n')
  datfile.write('param ' + name + ' :=\n')
  pyfile.write(name + ' = numpy.array([ ')
  for row in range(int(dimensions[0])):
    datfile.write(str(row + 1) + ' ' + str(array[row]) + '\n')
    pyfile.write(str(array[row]) + ', ')
  datfile.write(';\n')
  datfile.write('\n')
  pyfile.write('])\n\n')


def printArray(array, dimensions, name, modfile, datfile, pyfile, rows, columns):
  if len(dimensions) == 1: return printArray1D(array, dimensions, name, modfile, datfile, pyfile, rows)
  if len(dimensions) == 2: return printArray2D(array, dimensions, name, modfile, datfile, pyfile, rows, columns)


def emitLayer(layerId, layerWidth, rows, columns, matrix=None):
  modfile.write('# layer ' + str(layerId) + '\n')
  modfile.write('param layer_' + str(layerId) + '_width;\n')
  modfile.write('param rows_' + str(layerId) + ';\n')
  modfile.write('param columns_' + str(layerId) + ';\n')
  modfile.write('param layer_' + str(layerId) + '_weight{i in 1..rows_' + str(layerId) + ', j in 1..columns_' + str(layerId) + '};\n')
  modfile.write('var layer_' + str(layerId) + '{i in 1..layer_' + str(layerId) + '_width};\n')
  modfile.write('param layer_' + str(layerId) + '_bias{i in 1..layer_' + str(layerId) + '_width};\n')
  
  datfile.write('param layer_' + str(layerId) + '_width := ' + str(layerWidth) + ';\n')
  datfile.write('param rows_' + str(layerId) + ' := ' + str(rows) + ';\n')
  datfile.write('param columns' + str(layerId) + ' := ' + str(columns) + ';\n')


def writeConstraints(file, layerId, isRecurrent, isBiased, isFirst, isLast):
  file.write("\n# range constraints\n")
  rangeLimit = 100
  l = str(layerId)
  lm = str(layerId - 1)
  
  file.write("subject to rangemax" + l + "{i in 1..layer_" + l + "_width}: z" + l + "[i] <= " + str(rangeLimit) + ";\n")
  file.write("subject to rangemin" + l + "{i in 1..layer_" + l + "_width}: z" + l + "[i] >= " + str(-rangeLimit) + ";\n")
  file.write("\n")
  file.write("# compute preactivations\n")
  file.write("subject to preactivation" + l +"{i in 1..layer_" + l + "_width}:\n")
  
  if isFirst:
    file.write("z" + l + "[i] = sum{j in 1..layer_" + lm + "_width} (layer_" + lm + "_weights[j, i] * a" + lm + "[j])\n")
  elif isLast:
    file.write("z" + l + "[i] = sum{j in 1..layer_" + lm + "_width} (layer_" + lm + "_weights[i, j] * a" + lm + "[j])\n")
  else:
    file.write("z" + l + "[i] = sum{j in 1..layer_" + lm + "_width} (layer_" + l + "_weights[j, i] * a" + lm + "[j])\n")
  
  if isRecurrent:
    file.write("+ sum{j in 1+layer_" + lm + "_width..rows_" + l + "} (layer_" + l + "_weights[j, i] * a" + l + "[j - layer_" + lm + "_width])\n")
  
  if isBiased:
    file.write("+ layer_" + l + "_bias[i]\n")
  file.write(";\n\n")

  file.write("# compute tanh activations\n")
  file.write("subject to activation" + l + "{i in 1..layer_" + l + "_width}:\n")
  if layerId == 1:
    file.write("a1[i] = z1[i];\n")
  else:
    file.write("a" + l + "[i] = tanh(z" + l + "[i]);\n")
  file.write("\n")







filename = "trained/elman_shakespeare._2_1024_178000.t7"
if len(sys.argv) > 1: filename = sys.argv[1]

outputname = filename
if len(sys.argv) > 2: outputname = sys.argv[2]

steps = 16
if len(sys.argv) > 3: steps = int(sys.argv[2])

print 'reading', filename, 'writing', outputname + '.*'

file = open(filename, "r")
format = readFormat(file)
lastLayerId = (len(format) - 2) / 2 + 2

i = 0

outputFilename = outputname + '_' + str(steps)
modfile = open(outputFilename + '.mod', 'w')
datfile = open(outputFilename + '.dat', 'w')
pyfile = open(outputFilename + '.py', 'w')
pyfile.write('import numpy\n')
numHiddenLayers = (len(format) - 3) / 2
pyfile.write('numHiddenLayers = ' + str(numHiddenLayers) + '\n')

modfile.write("param one_hot_encoding_width;\n")
modfile.write("param compressed_input_width;\n")

modfile.write("param rows_0 := one_hot_encoding_width;\n")

for i in range(numHiddenLayers):
  modfile.write('param rows_' + str(i + 2) + ';\n')
  datfile.write('param rows_' + str(i + 2) + ' := ' + str(format[i * 2 + 1][0]) + ';\n')
modfile.write("param rows_" + str(lastLayerId) + ";\n")
modfile.write('\n')


modfile.write("param layer_0_width := one_hot_encoding_width;\n")
modfile.write('param layer_1_width := compressed_input_width;\n')
for i in  range(numHiddenLayers):
  modfile.write('param layer_' + str(i + 2) + '_width := rows_' + str(i + 2) + ' - layer_' + str(i + 1) + '_width;\n')
modfile.write('param layer_' + str(lastLayerId) + '_width;\n')
datfile.write('param layer_' + str(lastLayerId) + '_width := ' + str(format[-2][0]) + ';\n')

for i in range(len(format)):
  name = None
  dimensions = format[i]
  print 'i', i, dimensions
  
  if i == 0:
    modfile.write("\n# layer 0\n")
    modfile.write("param columns_0 := compressed_input_width;\n")
    
    datfile.write("param one_hot_encoding_width := 65;\n")
    datfile.write("param compressed_input_width := 64;\n")
  
    array = readArray(file, dimensions)
    printArray(array, dimensions, "layer_0_weights", modfile, datfile, pyfile, "rows_0", "columns_0")
  

  elif (i % 2) == 0:
    layerId = (i - 1) / 2 + 2
    array = readArray(file, dimensions)
    rows = "rows_" + str(layerId)
    columns = "columns_" + str(layerId)
    printArray(array, dimensions, "layer_" + str(layerId) + "_bias", modfile, datfile, pyfile, 'layer_' + str(layerId) + '_width', None)

  else:
    layerId = (i - 1) / 2 + 2
    
    modfile.write("\n# layer " + str(layerId) + "\n")
    modfile.write("param columns_" + str(layerId) + ";\n")

    datfile.write("param columns_" + str(layerId) + " := " + str(dimensions[1]) + ";\n")
  
    rows = "rows_" + str(layerId)
    columns = "columns_" + str(layerId)
    array = readArray(file, dimensions)
    printArray(array, dimensions, "layer_" + str(layerId) + "_weights", modfile, datfile, pyfile, rows, columns)


def name(variable, layerId, step):
  return variable + str(layerId) + '_' + str(step)

lastLayerId = (len(format) - 2) / 2 + 2

for t in range(steps - 1, -1, -1):
  a0 = name('a', 0, t)
  modfile.write('var ' + a0 + '{i in 1..layer_0_width};\n')
  a1 = name('a', 1, t)
  modfile.write('var ' + a1 + '{i in 1..layer_1_width};\n')
  
  for i in range(numHiddenLayers):
    layerId = i + 2
    z = name('z', layerId, t)
    modfile.write('var ' + z + '{i in 1..layer_' + str(layerId) + '_width};\n')
    a = name('a', layerId, t)
    modfile.write('var ' + a + '{i in 1..layer_' + str(layerId) + '_width};\n')
  
  zLast = name('z', lastLayerId, t)
  modfile.write('var ' + zLast + '{i in 1..layer_' + str(lastLayerId) + '_width};\n')

modfile.write('param y_target{i in 1..layer_' + str(lastLayerId) + '_width};\n')
modfile.write('minimize loss{i in 1..layer_' + str(lastLayerId) + '_width}: ')
modfile.write('(y_target[i] - z' + str(lastLayerId) + '_' + str(steps - 1) + '[i])^2;\n')

for t in range(steps - 1, -1, -1):
  modfile.write('\n# step ' + str(t) + '\n')
  a0 = name('a', 0, t)
  a1 = name('a', 1, t)
  zLast = name('z', lastLayerId, t)

  modfile.write('\n')
  modfile.write('subject to target_' + str(t) + '{i in 1..layer_0_width}:\n')
  if t == steps - 1:
    modfile.write('z' + str(lastLayerId) + '_' + str(t) + '[i] = y_target[i];\n')
  else:
    modfile.write('z' + str(lastLayerId) + '_' + str(t) + '[i] = a0_' + str(t + 1) + '[i];\n')

  for i in range(numHiddenLayers, 0, -1):
    layerId = i + 1
    z = name('z', layerId, t)
    aP = name('a', layerId + 1, t)
    a = name('a', layerId, t)
    aPPrevious = name('a', layerId + 1, t - 1)
    zP = name('z', layerId + 1, t)

    modfile.write('\n')
    modfile.write('subject to activation' + str(layerId) + '_' + str(t))
    modfile.write('{i in 1..layer_' + str(layerId) + '_width}:\n')
    modfile.write(a + '[i] = sum{j in 1..layer_' + str(layerId + 1) + '_width} ')
    modfile.write('layer_' + str(layerId + 1) + '_weights_feedforward_inverse')
    if layerId + 1 == lastLayerId:
      modfile.write('[i, j] * ' + zP + '[j];\n')
    else:
      modfile.write('[j, i] * ( ' + zP + '[j] - \n')
      if t > 0:
        modfile.write('sum{k in 1..layer_' + str(layerId + 1) + '_width} ')
        modfile.write('layer_' + str(layerId + 1) + '_weights_recurrent[k, j] * ' + aPPrevious + '[k] - ')
      modfile.write('layer_' + str(layerId + 1) + '_bias[j] );\n')
    
    modfile.write('subject to preactivation' + str(layerId) + '_' + str(t))
    modfile.write('{i in 1..layer_' + str(layerId) + '_width}:\n')
    modfile.write(z + '[i] = atanh(' + a + '[i]);\n')

  modfile.write('\n')
  modfile.write('subject to activation1_' + str(t) + '{i in 1..layer_1_width}:\n')
  modfile.write(a1 + '[i] = sum{j in 1..layer_2_width} ')
  modfile.write('layer_2_weights_feedforward_inverse[j, i] * ')
  modfile.write('( z2_' + str(t) + '[j] -\n')
  if t > 0:
    a2M = name('a', 2, t - 1)
    modfile.write('sum{k in 1..layer_' + str(layerId) + '_width} layer_2_weights_recurrent[k, j] * ' + a2M + '[k] - ')
  modfile.write('layer_2_bias[j] );\n')

  modfile.write('\n')
  modfile.write('subject to activation0_' + str(t) + '{i in 1..layer_0_width}:\n')
  modfile.write(a0 + '[i] = sum{j in 1..layer_1_width} layer_0_weights_feedforward_inverse[j, i] * ' + a1 + '[j];\n')
                


