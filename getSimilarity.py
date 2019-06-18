import numpy as np
import cv2
from scipy import spatial
from skimage.measure import compare_ssim

class SimilarityConstants:
    def __init__(self):
        #Public constants to specify path
        self.ORB = 0
        self.KAZE = 1
        self.SSIM = 2


class Similarity:
    def __init__(self, model):
        #Initialize vars
        self.sim = 0
        self.model = model
        self.paths = []

        #Public constants
        self.ORB = 0
        self.KAZE = 1
        self.SSIM = 2

    def getSimilarity(self, path1, path2):
        sim = 0
        self.paths = [path1, path2]
        if self.model==self.ORB:
            print("USING ORB")
            sim = self.getORB()
        elif self.model==self.KAZE:
            print("USING KAZE")
            sim = self.getKAZE()
        elif self.model==self.SSIM:
            print("USING SSIM")
            sim = self.getSSIM()

        self.sim = sim
        return sim

    def getORB(self):
        descriptions = []
        for i in range(len(self.paths)):
            vector_size = 32
            orb = cv2.ORB_create()

            #Create opencv image var
            img = cv2.imread(self.paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #Get keypoints and descriptors
            keypoints = orb.detect(img,None)

            #Sort them based on response value, a bigger response value is better, extract only first 32
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]

            keypoints, description = orb.compute(img, keypoints)

            #Make it needed size
            description = description.flatten()

            if description.size < vector_size * 64:
                description = np.concatenate([description, np.zeros((vector_size * 64) - description.size)])

            descriptions.append(description)

        val = 0
        #Get cosine similarity between two descriptors
        descComparee = descriptions[0]
        descComparer = descriptions[1]

        val = self.cosineSim(descComparee, descComparer)
        print(val)

        return val

    def getKAZE(self):
        descriptions = []
        for i in range(len(self.paths)):
            vector_size = 32
            kaze = cv2.KAZE_create()

            #Create opencv image var
            img = cv2.imread(self.paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #Get keypoints and descriptors
            keypoints = kaze.detect(img)

            #Sort them based on response value, a bigger response value is better, extract only first 32
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]

            keypoints, description = kaze.compute(img, keypoints)

            #Make it needed size
            description = description.flatten()
            if description.size < vector_size * 64:
                description = np.concatenate([description, np.zeros((vector_size * 64) - description.size)])

            descriptions.append(description)

        descComparee = descriptions[0]
        descComparer = descriptions[1]

        val = self.cosineSim(descComparee, descComparer)
        return val

    def getSSIM(self):
        imgA = cv2.imread(self.paths[0])
        imgB = cv2.imread(self.paths[1])

        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(imgA, imgB, full=True)
        return score

    def cosineSim(self, vec1, vec2):
        vec1 = vec1.reshape(-1,1)
        vec2 = vec2.reshape(-1,1)

        return 1 - spatial.distance.cosine(vec1, vec2)








