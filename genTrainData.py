import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
import cv2
from tqdm import tqdm
from similarity import CompareImage
import random
import codecs, json
from tqdm import tqdm

def getTextImg(letter, number):
	print(letter)
	W, H = (170,170) #For text image
	arial = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 75)

	newimg = Image.new('1', (W,H), 1)
	draw = ImageDraw.Draw(newimg)

	w, h = draw.textsize(letter, font=arial)

	draw.text(((W-w)/2, (H-h)/2),letter,font=arial)

	# newimg.save("textImgs/" + str(number) + letter + ".jpeg")
	# newimg.show()
	# newimg.show()
	return newimg

def getTrainingImage(i, trainOrTest, textImg, code):
	pathToTrainImgs = 'data/train/img/'
	pathToTrainAnns = 'data/train/ann/'

	pathToTestImgs = 'data/test/img/'
	pathToTestAnns = 'data/test/ann/'

	arial = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 75)
	Wimg, Himg = (2560, 1440)
	textLocation = (random.randint(0,2261), random.randint(0,1141))

	trainImg = Image.new('1',(Wimg,Himg), 1)
	trainImg.paste(textImg, textLocation)	

	draw = ImageDraw.Draw(trainImg)
	w, h = draw.textsize(code, font=arial)
	rectangleTL = (textLocation[0]+((170-w)/2), textLocation[1]+((170-h)/2))
	rectangleBR = (rectangleTL[0]+w, rectangleTL[1]+h+5)



	draw.rectangle((rectangleTL, rectangleBR))

	jsonX = getImgJson(trainOrTest, trainImg, rectangleTL, rectangleBR, i, code)

	if trainOrTest=="train":
		trainImg.save(pathToTrainImgs + str(i) + code + trainOrTest + '.jpeg')
		with codecs.open(pathToTrainAnns + str(i) + code+trainOrTest + '.json', 'w','utf8') as f:
			string = json.dumps(jsonX, sort_keys = True, ensure_ascii=False)
			f.write(string)

	elif trainOrTest=="test":
		trainImg.save(pathToTestImgs + str(i) + code + trainOrTest + '.jpeg')
		with codecs.open(pathToTestAnns + str(i)+code+trainOrTest + '.json', 'w','utf8') as f:
			string = json.dumps(jsonX, sort_keys = True, ensure_ascii=False)
			f.write(string)

	# print(json)
	# trainImg.show()
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

def getImgJson(trainOrTest, img, TL,BR, i, code):
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

def genData(number):
	trainNumber = round(0.95*number)
	testNumber = number - trainNumber

	codesList = getCodeList('codes.txt')

	k = 0

	testJson = getMetaJson('test')
	trainJson = getMetaJson('train')

	pathToTest = 'data/'

	with codecs.open(pathToTest + 'meta.json', 'w','utf8') as f:
		string = json.dumps(testJson, sort_keys = True, ensure_ascii=False)
		f.write(string)


	for i in tqdm(range(trainNumber)):
		if k<len(codesList):
			code = codesList[k]
			textImage = getTextImg(code, i)
			getTrainingImage(i, "train", textImage, code)
		else:
			randomIndex = random.randint(0, len(codesList))
			code = codesList[randomIndex]

			textImage = getTextImg(code, i)
			getTrainingImage(i, "train", textImage, code)
		k=k+1

	for i in tqdm(range(testNumber)):
		randomIndex = random.randint(0, len(codesList))
		code = codesList[randomIndex]

		textImage = getTextImg(code, i)
		getTrainingImage(i, "test", textImage, code)

genData(10000)




#													TODO LIST
#--------------------------------------#--------------------------------------#--------------------------------------
#	1. Get a random 2 letter code from codeList
#	2. Generate image - the code placed in a random location. 
#	3. Save image in proper directory
#	4. Save the json in the proper format in proper directory
#--------------------------------------#--------------------------------------#--------------------------------------


