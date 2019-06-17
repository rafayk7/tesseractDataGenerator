from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd
from tqdm import tqdm
from getSimilarity import SimilarityConstants, Similarity
import codecs, json
import numpy as np

#TODO
#Take picture of blank paper, change getTrainingImage to take this picture as instead of white
#Change getTextImgForTraining to randomly generate float and place image at that angle

#									                    CONTENTS
# --------------------------------------#--------------------------------------#--------------------------------------
#   1. getCharList(file) -- opens the file with characters and returns a list of characters from file
#   2. genCodes(outputfile) -- generates a list of codes and writes to outputfile
#   3. levenshteinDistance(str1, str2) -- calculates the similarity between the two strings. used in genCodes
#   4. replaceCost(char1, char2, matrix) -- gives similarity of two chars from matrix. used in levenshteinDistance
#   5. fixCharList(file, outputfile) -- fixed chars.txt to produce charsFixed2.txt
#   6. getImg(letter, number) -- saves a (150,300) image of a character to textImgs dir
#   7. getSimilarity(img1, img2) -- returns similarity between two Images
#	8. genMatrix(file) -- generates similarities between chars from file and creates a pandas df and saves to a csv file
#   9. saveImgs(file) -- calls getImg on all characters in file
#   10. getTextImgForTraining(code, number) - generates a (170,170) image of a code for training TODO: Add more variation
#   11. getTrainingImage(i, trainOrTest, textImg, code) - Saves a training image from a textImg along with the json metadata file for the image
#   12. getMetaJson(testOrTrain) -- returns json object of metadata (USED ONLY FOR SUPERVISELY)
#   13. getImgJson(trainOrTest, img, TL, BR, i, code) -- returns json object with bounding box and code with corresponding image file
#   14. getCodeList(file) -- like getCharList but for codes
#   15. genData(n) -- generates n training images (95/5) (train/test) split
# --------------------------------------#--------------------------------------#--------------------------------------

def getCharList(file):
    charlist = []
    with open(file, 'r') as f:
        for line in f:
            charlist.append(line)

    charlist = [x.strip("\n") for x in charlist]
    return charlist


def genCodes(outputfile):
    outputFile = open(outputfile, 'w')
    charList = getCharList('chars.txt')

    for i in range(10000):
        code = ''
        indexOne = random.randrange(len(charList) - 1)
        indexTwo = random.randrange(len(charList) - 1)

        code = charList[indexOne] + charList[indexTwo]

        if (len(code) > 2):
            print("ERROR: STOP LOOP/CODE LENGTH >2")

    outputFile.write(code)
    outputFile.write("\n")
    outputFile.close()


def levenshteinDistance(str1, str2):
    data = pd.read_csv("simVals.csv", index_col=0)
    m = len(str1)
    n = len(str2)
    lensum = float(m + n)
    d = []
    for i in range(m + 1):
        d.append([i])
    del d[0][0]
    for j in range(n + 1):
        d[0].append(j)
    for j in range(1, n + 1):  # cost(j,i).
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                d[i].insert(j, d[i - 1][j - 1])
            else:
                minimum = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 2)
                d[i].insert(j, minimum)
    ldist = d[-1][-1]
    ratio = (lensum - ldist) / lensum
    return {'distance': ldist, 'ratio': ratio}


def replaceCost(char1, char2, matrix):
    return matrix[char1][char2]


def fixCharList(file, output_file):
    onlyChars = []
    i = 0

    # Remove unicode codes, only want characters
    with open(file, 'r') as f:
        for line in f:
            if i % 2 == 0:
                onlyChars.append(line.strip("\n"))
            i += 1

    with open(output_file, 'w+') as f:
        for line in onlyChars:
            f.write(line)
            f.write('\n')


def getImg(letter, number):
    W, H = (150, 300)
    arial = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 150)
    newimg = Image.new('1', (W, H), 1)
    draw = ImageDraw.Draw(newimg)

    w, h = draw.textsize(letter, font=arial)

    draw.text(((W - w) / 2, (H - h) / 2), letter, font=arial)

    newimg.save("textImgs/" + str(number) + letter + ".jpeg")
    return newimg


def getSimilarity(path1, path2, Similarity):
    # imgA = cv2.imread(path1)
    # imgB = cv2.imread(path2)
    #
    # imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    # imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    #
    # comparer = CompareImage(path1, path2)
    # s = comparer.compare_image()
    #

    return Similarity.getSimilarity(path1, path2)

    # s = measure.compare_ssim(imgA, imgB)
    # return s


def getCharList(file):
    charlist = []
    with open(file, 'r') as f:
        for line in f:
            charlist.append(line)

    charlist = [x.strip("\n") for x in charlist]
    return charlist


def genMatrix(file):
    charList = getCharList(file)
    matrixSize = (len(charList), len(charList))
    x = Similarity(SimilarityConstants().ORB)

    matrix = np.zeros(matrixSize)

    for i in tqdm(range(len(charList))):
        charA = charList[i]
        pathA = "textImgs/" + str(i) + charA + ".jpeg"
        for j in range(len(charList)):
            charB = charList[j]
            pathB = "textImgs/" + str(j) + charB + ".jpeg"

            matrix[i, j] = getSimilarity(pathA, pathB, x)

    df = pd.DataFrame(data=matrix, index=charList, columns=charList)
    df.to_csv("simVals.csv")


def saveImgs(file):
    charList = getCharList(file)
    for i in tqdm(range(len(charList))):
        getImg(charList[i], i)


def getTextImgForTraining(letter, number):
    # print(letter)
    W, H = (170, 170)  # For text image
    arial = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 75)

    newimg = Image.new('1', (W, H), 1)
    draw = ImageDraw.Draw(newimg)

    w, h = draw.textsize(letter, font=arial)

    draw.text(((W - w) / 2, (H - h) / 2), letter, font=arial)
    newimg = newimg.rotate(17.5, expand=1)

    # newimg.save("textImgs/" + str(number) + letter + ".jpeg")
    newimg.show()
    return newimg



def getTrainingImage(i, trainOrTest, textImg, code):
    pathToTrainImgs = 'data/train/img/'
    pathToTrainAnns = 'data/train/ann/'

    pathToTestImgs = 'data/test/img/'
    pathToTestAnns = 'data/test/ann/'

    arial = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 75)
    Wimg, Himg = (2560, 1440)
    textLocation = (random.randint(0, 2261), random.randint(0, 1141))

    trainImg = Image.new('1', (Wimg, Himg), 1)
    trainImg.paste(textImg, textLocation)

    draw = ImageDraw.Draw(trainImg)
    w, h = draw.textsize(code, font=arial)

    #Top left and Bottom Right of rectangle
    rectangleTL = (textLocation[0] + ((170 - w) / 2), textLocation[1] + ((170 - h) / 2))
    rectangleBR = (rectangleTL[0] + w, rectangleTL[1] + h + 5)

    # draw.rectangle((rectangleTL, rectangleBR))

    jsonX = getImgJson(trainOrTest, trainImg, rectangleTL, rectangleBR, i, code)

    if trainOrTest == "train":
        trainImg.save(pathToTrainImgs + str(i) + code + trainOrTest + '.jpeg')
        with codecs.open(pathToTrainAnns + str(i) + code + trainOrTest + '.json', 'w', 'utf8') as f:
            string = json.dumps(jsonX, sort_keys=True, ensure_ascii=False)
            f.write(string)

    elif trainOrTest == "test":
        trainImg.save(pathToTestImgs + str(i) + code + trainOrTest + '.jpeg')
        with codecs.open(pathToTestAnns + str(i) + code + trainOrTest + '.json', 'w', 'utf8') as f:
            string = json.dumps(jsonX, sort_keys=True, ensure_ascii=False)
            f.write(string)

    # print(json)
    trainImg.show()
    return trainImg



def getMetaJson(testOrTrain):
    json = {"tags_objects": ["text"],
            "tags_images": ["codes"],
            "classes": [
                {
                    "title": "text",
                    "shape": "rectangle",
                    "color": "#000000"
                }
            ]
            }

    return json


def getImgJson(trainOrTest, img, TL, BR, i, code):
    json = {"description": trainOrTest + " data",
            "name": str(i) + code + trainOrTest,
            "size": {
                "width": 2560,
                "height": 1440
            },
            "tags": ["text"],
            "objects": [
                {
                    "description": code,
                    "tags": ["codes"],
                    "bitmap": None,
                    "classTitle": "text",
                    "points": {
                        "exterior": [
                            [TL[0], TL[1]],
                            [BR[0], BR[1]]
                        ],
                        "interior": []
                    }
                }
            ]
            }

    return json


def getCodeList(file):
    charlist = []
    with open(file, 'r') as f:
        for line in f:
            charlist.append(line)

    charlist = [x.strip("\n") for x in charlist]
    return charlist

def genData(number, codesfile):
    trainNumber = round(0.95 * number)
    testNumber = number - trainNumber

    codesList = getCodeList(codesfile)
    k = 0

    testJson = getMetaJson('test')
    trainJson = getMetaJson('train')

    pathToTest = 'data/'

    with codecs.open(pathToTest + 'meta.json', 'w', 'utf8') as f:
        string = json.dumps(testJson, sort_keys=True, ensure_ascii=False)
        f.write(string)

    for i in tqdm(range(trainNumber)):
        if k < len(codesList):
            code = codesList[k]
            textImage = getTextImgForTraining(code, i)
            getTrainingImage(i, "train", textImage, code)
        else:
            randomIndex = random.randint(0, len(codesList))
            code = codesList[randomIndex]

            textImage = getTextImgForTraining(code, i)
            getTrainingImage(i, "train", textImage, code)
        k = k + 1

    for i in tqdm(range(testNumber)):
        randomIndex = random.randint(0, len(codesList))
        code = codesList[randomIndex]

        textImage = getTextImgForTraining(code, i)
        getTrainingImage(i, "test", textImage, code)

# genData(2, 'charsFixed2.txt')

