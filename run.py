from Utils import genData

codesfile = 'codesKOR.txt'
trainingAmt = 1000
trainTestSplit = 0.7
outputdir = 'data'

# chars = getCharList('charsFixed2.txt')
# genCodes(chars, 'codesKOR.txt', 10000)

genData(trainingAmt,codesfile, trainTestSplit, outputdir)
