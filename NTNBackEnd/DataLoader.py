import os
import random
import numpy as np
import cv2
from NTNBackEnd.SamplePreprocessor import preprocess

class Sample:
    "sample from the dataset"

    def __init__(self, gtText, filePath):
        #gtText is the ground truth (labels)
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, gtTexts, imgs):
        #gtTexts is the list of ground truths
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:
    "loads data which corresponds to IAM format"

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        "loader for dataset at given location, preprocess images and text according to parameters"

        self.dataAugmentation = False    #dataAugmentation is true for training images and false for validation images
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []                #it is a list of Samples containing the mapping between gtText and filePath

        f = open(filePath + '\lines.txt')
        #filePath is passed as an argument, which has the path of the base directory (C:\Users\ISHIKA\Desktop\EAD Project\NTN\data)
        
        chars = set()

        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']

        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + '\lines\\' + fileNameSplit[0] + '\\' + fileNameSplit[0] + '-' + fileNameSplit[1] + '\\' + \
                       lineSplit[0] + '.png'
           

            #The last element of the lineSplit gives the actual word (GT) corresponding to the image
            gtText_list = lineSplit[8].split('|')
            gtText = self.truncateLabel(' '.join(gtText_list), maxTextLen)
            #If cost of the word is greater than maxTextLen, then truncateLabel returns a part of the word
            #Function on line 97
            chars = chars.union(set(list(gtText)))
            #chars is the set of all characters in a particular word

            # check if image is not empty
            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            # put sample into list
            self.samples.append(Sample(gtText, fileName))

        # some images in the IAM dataset are known to be damaged, don't show warning for them
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        # split into training and validation set: 95% - 5%
        random.shuffle(self.samples)

        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]
        
        # put lines into lists
        self.trainLines = [x.gtText for x in self.trainSamples]
        self.validationLines = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 9500

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))


    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input labels. 
        # Repeat letters cost double because of the blank symbol needing to be inserted.
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def trainSet(self):
        "switch to randomly chosen subset of training set"
        
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples

    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        "returns true if some samples in the trainingSet are still left"
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [
            preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation)
            for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)
        #gtTexts of that batch
