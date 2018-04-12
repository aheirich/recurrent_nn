#
# logToNumpy.py
#
# Takes a log produced by PRINT_MODEL.lua and produces numpy source
#

import sys
import numpy

"""

dimensions for elman network with 2 hidden layers of 1024 neurons each:

i 0 ['65', '64'] a0, a1
i 1 ['1088', '1024'] a2
i 2 ['1024']
i 3 ['2048', '1024'] a3
i 4 ['1024']
i 5 ['65', '1024'] a4
i 6 ['65']

model file:

  param one_hot_encoding_width;
  param compressed_input_width;
  
  # layer 0
  param rows_0 := one_hot_encoding_width;
  param columns_0 := compressed_input_width;
  param layer_0_width := one_hot_encoding_width;
  var a0{i in 1..layer_0_width};
  
  param layer_0_weights{i in 1..rows_0, j in 1..columns_0};
  
  # layer 1
  param layer_1_width := compressed_input_width;
  var a1{i in 1..layer_1_width};
  var z1{i in 1..layer_1_width};
  
  # range constraints
  subject to rangemax1{i in 1..layer_1_width}: z1[i] <= 10;
  subject to rangemin1{i in 1..layer_1_width}: z1[i] >= -10;
  
  # compute preactivations
  subject to preactivation1{i in 1..layer_1_width}:
  z1[i] = sum{j in 1..layer_0_width} (layer_0_weights[i, j] * a0[j])
  ;
  
  # compute Relu activations
  subject to activation1{i in 1..layer_1_width}:
  a1[i] = z1[i] * (tanh(100.0*z1[i]) + 1) * 0.5;
  
  
  # layer 2
  param rows_2;
  param columns_2;
  param layer_2_width := rows_2 - layer_1_width;
  var a2{i in 1..layer_2_width};
  var z2{i in 1..layer_2_width};
  
  param layer_2_weights{i in 1..rows_2, j in 1..columns_2};
  
  param layer_2_bias{i in 1..rows_2};
  
  # range constraints
  subject to rangemax2{i in 1..layer_2_width}: z2[i] <= 10;
  subject to rangemin2{i in 1..layer_2_width}: z2[i] >= -10;
  
  # compute preactivations
  subject to preactivation2{i in 1..layer_2_width}:
  z2[i] = sum{j in 1..layer_1_width} (layer_2_weights[i, j] * a1[j])
  + sum{j in 1+layer_1_width..rows_2} (layer_2_weights[j, i] * a2[j - layer_1_width])
  + layer_2_bias[i]
  ;
  
  # compute Relu activations
  subject to activation2{i in 1..layer_2_width}:
  a2[i] = z2[i] * (tanh(100.0*z2[i]) + 1) * 0.5;
  
  
  # layer 3
  param rows_3;
  param columns_3;
  param layer_3_width := rows_3 - layer_2_width;
  var a3{i in 1..layer_3_width};
  var z3{i in 1..layer_3_width};
  
  param layer_3_weights{i in 1..rows_3, j in 1..columns_3};
  
  param layer_3_bias{i in 1..rows_3};
  
  # range constraints
  subject to rangemax3{i in 1..layer_3_width}: z3[i] <= 10;
  subject to rangemin3{i in 1..layer_3_width}: z3[i] >= -10;
  
  # compute preactivations
  subject to preactivation3{i in 1..layer_3_width}:
  z3[i] = sum{j in 1..layer_2_width} (layer_3_weights[i, j] * a2[j])
  + sum{j in 1+layer_2_width..rows_3} (layer_3_weights[j, i] * a3[j - layer_2_width])
  + layer_3_bias[i]
  ;
  
  # compute Relu activations
  subject to activation3{i in 1..layer_3_width}:
  a3[i] = z3[i] * (tanh(100.0*z3[i]) + 1) * 0.5;
  
  
  # layer 4
  param rows_4;
  param columns_4;
  param layer_4_width;
  var a4{i in 1..layer_4_width};
  var z4{i in 1..layer_4_width};
  
  param layer_4_weights{i in 1..rows_4, j in 1..columns_4};
  
  param layer_4_bias{i in 1..rows_4};
  
  # range constraints
  subject to rangemax4{i in 1..layer_4_width}: z4[i] <= 10;
  subject to rangemin4{i in 1..layer_4_width}: z4[i] >= -10;
  
  # compute preactivations
  subject to preactivation4{i in 1..layer_4_width}:
  z4[i] = sum{j in 1..layer_3_width} (layer_4_weights[i, j] * a3[j])
  + layer_4_bias[i]
  ;
  
  # compute Relu activations
  subject to activation4{i in 1..layer_4_width}:
  a4[i] = z4[i] * (tanh(100.0*z4[i]) + 1) * 0.5;
  
  
  
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
  array = [ float(words[1]) ]
  for row in range(rows - 1):
    line = file.readline().strip()
    value = float(line)
    array.append(value)
  return array


def readArray(file, dimensions):
  if len(dimensions) == 1: return numpy.array(readArray1D(file, dimensions))
  if len(dimensions) == 2: return numpy.array(readArray2D(file, dimensions))


def printArray2D(array, dimensions, name, modfile, datfile, rows, columns):
  modfile.write('\n')
  datfile.write('\n')
  modfile.write('param ' + name + '{i in 1..' + rows + ', j in 1..' + columns + '};\n')
  datfile.write('param ' + name + ': ')
  rows = int(dimensions[0])
  columns = int(dimensions[1])
  line = ''
  for column in range(columns):
    line = line + str(column + 1) + ' '
  line = line + ':= \n'
  datfile.write(line)
  for row in range(rows):
    line = str(row + 1) + ' '
    for column in range(columns):
      line = line + str(array[row][column]) + ' '
    datfile.write(line + '\n')
  datfile.write(';\n')
  datfile.write('\n')


def printArray1D(array, dimensions, name, modfile, datfile, rows):
  modfile.write('\n')
  datfile.write('\n')
  modfile.write('param ' + name + '{i in 1..' + rows + '};\n')
  datfile.write('param ' + name + ' :=\n')
  for row in range(int(dimensions[0])):
    datfile.write(str(row + 1) + ' ' + str(array[row]) + '\n')
  datfile.write(';\n')
  datfile.write('\n')


def printArray(array, dimensions, name, modfile, datfile, rows, columns):
  if len(dimensions) == 1: return printArray1D(array, dimensions, name, modfile, datfile, rows)
  if len(dimensions) == 2: return printArray2D(array, dimensions, name, modfile, datfile, rows, columns)




filename = "trained/PRINT_MODEL.log"
if len(sys.argv) > 1: filename = sys.argv[1]

outputname = "model_ampl"
if len(sys.argv) > 2: outputname = sys.argv[2]

file = open(filename, "r")
format = readFormat(file)
i = 0

modfile = open(outputname + '.mod', 'w')
datfile = open(outputname + '.dat', 'w')



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


def writeConstraints(file, layerId, isRecurrent, isBiased, isFirst):
  file.write("\n# range constraints\n")
  rangeLimit = 10
  l = str(layerId)
  lm = str(layerId - 1)
  
  file.write("subject to rangemax" + l + "{i in 1..layer_" + l + "_width}: z" + l + "[i] <= " + str(rangeLimit) + ";\n")
  file.write("subject to rangemin" + l + "{i in 1..layer_" + l + "_width}: z" + l + "[i] >= " + str(-rangeLimit) + ";\n")
  file.write("\n")
  file.write("# compute preactivations\n")
  file.write("subject to preactivation" + l +"{i in 1..layer_" + l + "_width}:\n")

  if isFirst:
    file.write("z" + l + "[i] = sum{j in 1..layer_" + lm + "_width} (layer_" + lm + "_weights[i, j] * a" + lm + "[j])\n")
  else:
    file.write("z" + l + "[i] = sum{j in 1..layer_" + lm + "_width} (layer_" + l + "_weights[i, j] * a" + lm + "[j])\n")

  if isRecurrent:
    file.write("+ sum{j in 1+layer_" + lm + "_width..rows_" + l + "} (layer_" + l + "_weights[j, i] * a" + l + "[j - layer_" + lm + "_width])\n")

  if isBiased:
    file.write("+ layer_" + l + "_bias[i]\n")
  file.write(";\n\n")

  file.write("# compute Relu activations\n")
  file.write("subject to activation" + l + "{i in 1..layer_" + l + "_width}:\n")
  file.write("a" + l + "[i] = z" + l + "[i] * (tanh(100.0*z" + l + "[i]) + 1) * 0.5;\n")
  file.write("\n")


for i in range(len(format)):
  name = None
  dimensions = format[i]
  print 'i', i, dimensions
  
  if i == 0:
    modfile.write("param one_hot_encoding_width;\n")
    modfile.write("param compressed_input_width;\n")
    modfile.write("\n# layer 0\n")
    modfile.write("param rows_0 := one_hot_encoding_width;\n")
    modfile.write("param columns_0 := compressed_input_width;\n")
    modfile.write("param layer_0_width := one_hot_encoding_width;\n")
    modfile.write("var a0{i in 1..layer_0_width};\n")
    
    datfile.write("param one_hot_encoding_width := 65;\n")
    datfile.write("param compressed_input_width := 64;\n")
  
    array = readArray(file, dimensions)
    printArray(array, dimensions, "layer_0_weights", modfile, datfile, "rows_0", "columns_0")

    modfile.write("\n# layer 1\n")
    modfile.write("param layer_1_width := compressed_input_width;\n")
    modfile.write("var a1{i in 1..layer_1_width};\n")
    modfile.write("var z1{i in 1..layer_1_width};\n")

    writeConstraints(modfile, 1, False, False, True)

  elif (i % 2) == 0:
    layerId = (i - 1) / 2 + 2
    array = readArray(file, dimensions)
    rows = "rows_" + str(layerId)
    columns = "columns_" + str(layerId)
    printArray(array, dimensions, "layer_" + str(layerId) + "_bias", modfile, datfile, rows, columns)

    isRecurrent = i < len(format) - 1
    writeConstraints(modfile, layerId, isRecurrent, True, False)


  else:
    layerId = (i - 1) / 2 + 2
    
    modfile.write("\n# layer " + str(layerId) + "\n")
    modfile.write("param rows_" + str(layerId) + ";\n")
    modfile.write("param columns_" + str(layerId) + ";\n")

    if i == len(format) - 2:
      modfile.write("param layer_" + str(layerId) + "_width;\n")
      datfile.write("param layer_" + str(layerId) + "_width := 65;\n")
    else:
      modfile.write("param layer_" + str(layerId) + "_width := rows_" + str(layerId) + " - layer_" + str(layerId - 1) + "_width;\n")

    modfile.write("var a" + str(layerId) + "{i in 1..layer_" + str(layerId) + "_width};\n")
    modfile.write("var z" + str(layerId) + "{i in 1..layer_" + str(layerId) + "_width};\n")

    datfile.write("param rows_" + str(layerId) + " := " + str(dimensions[0]) + ";\n")
    datfile.write("param columns_" + str(layerId) + " := " + str(dimensions[1]) + ";\n")
  
    rows = "rows_" + str(layerId)
    columns = "columns_" + str(layerId)
    array = readArray(file, dimensions)
    printArray(array, dimensions, "layer_" + str(layerId) + "_weights", modfile, datfile, rows, columns)

