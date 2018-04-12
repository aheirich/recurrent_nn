#
# logToNumpy.py
#
# Takes a log produced by PRINT_MODEL.lua and produces numpy source
#

import sys
import numpy

"""
{
  1 : FloatTensor - size: 65x64
  2 : FloatTensor - size: 192x128
  3 : FloatTensor - size: 128
  4 : FloatTensor - size: 65x128
  5 : FloatTensor - size: 65
}
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

"""
  
  i 0 ['65', '64'] a0, a1
  i 1 ['1088', '1024'] a2
  i 2 ['1024']
  i 3 ['2048', '1024'] a3
  i 4 ['1024']
  i 5 ['65', '1024'] a4
  i 6 ['65']



  i==0:
  i 0 ['65', '64']

  layer 0 65
  
  rows0 = 65, columns0 = 64
  weights 65 to 64
  
  layer 1 64
  
  i==1:
  i 1 ['1088', '1024']

  rows2 = 1088, columns2 = 1024
  weights 64 + 1024 to 1024
  
  i==2:
  i 2 ['1024']

  bias 1024
  
  i==3:
  i 3 ['2048', '1024']

  layer 2 1024
  
  rows3 = 2048, columns3 = 1024
  weights 1024 + 1024 to 1024
  
  i==4:
  i 4 ['1024']

  bias 1024
  
  i==5:
  i 5 ['65', '1024']

  layer 3 1024
  
  rows4 = 65, columns4 = 1024
  weights 1024 to 65
  
  i==6:
  bias 65
  i 6 ['65']

  layer 4 65
  
  
  
  """


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
    modfile.write("param layer_1_width;\n")
    modfile.write("var a1{i in 1..layer_1_width};\n")
    modfile.write("var z1{i in 1..layer_1_width};\n")


  elif (i % 2) == 0:
    layerId = (i - 1) / 2 + 2
    array = readArray(file, dimensions)
    rows = "rows_" + str(layerId)
    columns = "columns_" + str(layerId)
    printArray(array, dimensions, "layer_" + str(layerId) + "_bias", modfile, datfile, rows, columns)



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

    modfile.write("var a_" + str(layerId) + "{i in 1..layer_" + str(layerId) + "_width};\n")
    modfile.write("var z_" + str(layerId) + "{i in 1..layer_" + str(layerId) + "_width};\n")

    datfile.write("param rows_" + str(layerId) + " := " + str(dimensions[0]) + ";\n")
    datfile.write("param columns_" + str(layerId) + " := " + str(dimensions[1]) + ";\n")
  
    rows = "rows_" + str(layerId)
    columns = "columns_" + str(layerId)
    array = readArray(file, dimensions)
    printArray(array, dimensions, "layer_" + str(layerId) + "_weights", modfile, datfile, rows, columns)

