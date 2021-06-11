import argparse
import os
import sys

import cv2
import editdistance
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from NTNBackEnd.DataLoader import Batch, DataLoader
from NTNBackEnd.SamplePreprocessor import preprocess, wer
from NTNBackEnd.Model import Model
from NTNBackEnd.SpellChecker import correct_sentence

path = 'C:/Users/ISHIKA/Desktop/NotebookToNotepad/NTNBackEnd'

class FilePaths:
    fnCharList = path + '/model/charList.txt'
    fnAccuracy = path + '/model/accuracy.txt'
    fnTrain = path + '/data/'
    fnInfer = path + '/data/test.png'
    fnCorpus = path + '/data/corpus.txt'
    fnDump = path + '/data/output.txt'

def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 5  # stop training after this number of epochs without improvement
    batchNum = 0

    totalEpoch = len(loader.trainSamples)//Model.batchSize
    
    while True:
        epoch += 1
        print('Epoch:', epoch, '/', totalEpoch)

        # train
        loader.trainSet()
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo() #returns current batch index and overall number of batches
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # Validate
        charErrorRate, textLineAccuracy, wordErrorRate = validate(model, loader)

        cer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
        
        # Tensorboard: Add cer_summary to writer
        model.writer.add_summary(cer_summary, epoch)
        text_line_summary = tf.Summary(value=[tf.Summary.Value(
            tag='textLineAccuracy', simple_value=textLineAccuracy)])  # Tensorboard: Track textLineAccuracy
        
        # Tensorboard: Add text_line_summary to writer
        model.writer.add_summary(text_line_summary, epoch)
        wer_summary = tf.Summary(value=[tf.Summary.Value(
            tag='wordErrorRate', simple_value=wordErrorRate)])  # Tensorboard: Track wordErrorRate

        # Tensorboard: Add wer_summary to writer
        model.writer.add_summary(wer_summary, epoch)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break

def validate(model, loader):
    "validate NN"
    
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0

    totalCER = []
    totalWER = []

    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])

            currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
            totalCER.append(currCER)

            currWER = wer(recognized[i].split(), batch.gtTexts[i].split())
            totalWER.append(currWER)

            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # print validation result
    try:
        charErrorRate = sum(totalCER)/len(totalCER)
        wordErrorRate = sum(totalWER)/len(totalWER)
        textLineAccuracy = numWordOK / numWordTotal
    except ZeroDivisionError:
        charErrorRate = 0
        wordErrorRate = 0
        textLineAccuracy = 0
        
    print('Character error rate: %f%%. Text line accuracy: %f%%. Word error rate: %f%%' % (charErrorRate*100.0, textLineAccuracy*100.0, wordErrorRate*100.0))

    return charErrorRate, textLineAccuracy, wordErrorRate


def load_different_image():
    imgs = []
    for i in range(1):
       imgs.append(preprocess(cv2.imread("../data/check_image/a ({}).png".format(i), cv2.IMREAD_GRAYSCALE), Model.imgSize))
    return imgs

def infer(model, fnImg):
    """ Recognize text in image provided by file path """
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), imgSize=Model.imgSize)
    if img is None:
        print("Image not found")

    imgs = load_different_image()
    imgs = [img] + imgs
    batch = Batch(None, imgs)
    recognized = model.inferBatch(batch)  # recognize text

    print("\n\n\n\n\nWithout Correction: ", recognized[0])
    ans = correct_sentence(recognized[0])
    print("\nWith Correction: ", ans)
    print("\n\n\n\n\n")
    f = open(r'C:/Users/ISHIKA/Desktop/NotebookToNotepad/output.txt', 'w')
    f.write(ans)
    return ans


def main():
    """ Main function """
    # Optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", help="train the neural network", action="store_true")
    parser.add_argument(
        "--validate", help="validate the neural network", action="store_true")
    args = parser.parse_args()

    # Train or validate on Cinnamon dataset
    if args.train or args.validate:
        # Load training data, create TF model
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

        # Execute training or validation

        if args.train:
            model = Model(loader.charList)
            train(model, loader)

        elif args.validate:
            model = Model(loader.charList, mustRestore=False)
            validate(model, loader)

    # Infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), mustRestore=False)
        infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
    main()