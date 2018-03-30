#
# logToNumpy.py
#
# Takes a log produced by PRINT_MODEL.lua and produces numpy source
#

import sys

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
  if len(dimensions) == 1: return readArray1D(file, dimensions)
  if len(dimensions) == 2: return readArray2D(file, dimensions)


def printArray2D(array, dimensions, name, modfile, datfile):
  modfile.write('\n')
  datfile.write('\n')
  modfile.write('param ' + name + '{i in 1..' + str(dimensions[0]) + ', 1..' + str(dimensions[1]) + '};\n')
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


def printArray1D(array, dimensions, name, modfile, datfile):
  modfile.write('\n')
  datfile.write('\n')
  modfile.write('param ' + name + '{i in 1..' + str(dimensions[0]) + '};\n')
  datfile.write('param ' + name + ' :=\n')
  rows = int(dimensions[0])
  for row in range(rows):
    datfile.write(str(row + 1) + ' ' + str(array[row]) + '\n')
  datfile.write(';\n')
  datfile.write('\n')


def printArray(array, dimensions, name, modfile, datfile):
  if len(dimensions) == 1: return printArray1D(array, dimensions, name, modfile, datfile)
  if len(dimensions) == 2: return printArray2D(array, dimensions, name, modfile, datfile)




filename = "trained/PRINT_MODEL.log"
if len(sys.argv) > 1: filename = sys.argv[1]

outputname = "model_ampl"
if len(sys.argv) > 2: outputname = sys.argv[2]

file = open(filename, "r")
format = readFormat(file)
i = 0

modfile = open(outputname + '.mod', 'w')
datfile = open(outputname + '.dat', 'w')



for dimensions in format:
  name = None
  if i == 0:
    datfile.write('param embedding_layer_width := ' + str(format[i][0]) + ';\n')
    datfile.write('param input_width := ' + str(format[i][1]) + ';\n')
    modfile.write('param embedding_layer_width ;\n')
    modfile.write('param input_width ;\n')
    name = 'embedInputWeight'
  elif i == len(format) - 1:
    modfile.write('param output_width ;\n')
    datfile.write('param output_width := ' + str(format[i][0]) + ';\n')
    name = 'outputBias'
  elif i % 2 == 1:
    layerId = i / 2
    modfile.write('param layer_' + str(layerId) + '_width ;\n')
    datfile.write('param layer_' + str(layerId) + '_width := ' + str(format[i][0]) + ' ;\n')
    name = 'layer_' + str(layerId) + '_weight'
  else:
    layerId = i / 2
    name = 'layer_' + str(layerId) + '_bias'

  array = readArray(file, dimensions)
  printArray(array, dimensions, name, modfile, datfile)
  i = i + 1

