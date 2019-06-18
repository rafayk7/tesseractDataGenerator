# Summary 
Generates training data with JSON annotations for training Tesseract OCR on custom text characters/codes and is fully compatible with Supervisely.

# Output
Generates the outputdir directory with the following tree structure:
```bash
├── outputdir
│   ├── train
│   │   ├── img
│   │   ├── ann
│   ├── test
│   │   ├── img
│   │   ├── ann
│   ├── meta.json
```
With images stored in their respective train (or test) /img folder and annotations in the train (or test) /ann folder. The meta.json file contains data for Supervisely. The annotations Json contains the text in the image, and the Top Left and Bottom Right co-ordinates of the bounding box. The text can be accessed by `json['objects'][0]['description']` and the points can be accessed by `json['objects'][0]['points']['exterior']`. 
# How to use

1. Clone the repo with `git clone https://github.com/rafayk7/korOCR.git`
2. Download requirements with `pip install requirements.txt`
3. Run run.py with `python3 run.py`

# Parameters to change
# In run.py

1. trainingAmt - Number of total images to be generated (default 1000)
2. trainTestSplit - Split between number of training and testing images  in range [0,1] (default 0.7)
3. codesfile - path to the file with codes/chars to generate training images from (default codesKOR.txt)
4. outputdir - directory to store all generated data (default /data)

# In Utils.py
1. getTextImgForTraining 
    1. minAngle - lower bound angle for skew generation
    2. maxAngle - upper bound angle for skew generation
    3. (W, H) - size of text image (not final training image - see getTrainingImage)
    4. fontSize - size of font of text - change with getTrainingImage
2. getImgJson
    1. format of JSON annotations file - currently in compatibility with Supervisely
    2. width, height - width and height of training image, change with (Wimg, Himg) in getTrainingImage
3. getTrainingImg
    1. fontSize - size of font of text - change with getTextImgForTraining
    2. (Wimg, Himg) - size of final training image
    3. backgroundImgPath - path to background image for training images

# Sample Results

![Alt text](screenshots/img1.jpeg?raw=true "Generated Training Images")
![Alt text](screenshots/img2.jpeg?raw=true)

