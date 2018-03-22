#
# logToNumpy.py
#
# Takes a log produced by PRINT_MODEL.lua and produces numpy source
#



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


def printArray(array, dimensions, name):
  print ''
  print name, '= numpy.array(['
  for row in array: print row, ','
  print '])'
  print ''




names = [ 'embedding', 'weights_layer1', 'bias_layer1', 'weights_layer2', 'bias_layer2',
  'projection', 'bias_projection' ]
file = open("trained/PRINT_MODEL.log", "r")
format = readFormat(file)
i = 0
print 'import numpy'
print ''
for dimensions in format:
  array = readArray(file, dimensions)
  printArray(array, dimensions, names[i])
  i = i + 1

