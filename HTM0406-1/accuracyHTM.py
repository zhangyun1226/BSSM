# -*- coding: UTF-8 -*-
from util_functions import *
import time
# import matplotlib.pyplot as plt
import multiprocessing

from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder

# plt.ion()

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'figure.autolayout': True})


def rdse_encoder_nearest_neighbor(trainData, trainLabel, unknownSequence, encoder):
    overlapSum = np.zeros((trainData.shape[0],))
    for i in range(trainData.shape[0]):
        overlapI = np.zeros((len(unknownSequence),))
        for t in range(len(unknownSequence)):
            overlapI[t] = np.sum(np.logical_and(encoder.encode(unknownSequence[t]),
                                                encoder.encode(trainData[i, t])))
        overlapSum[i] = np.sum(overlapI)

    predictedClass = trainLabel[np.argmax(overlapSum)]
    return predictedClass


def constructDistanceMat(distMatColumn, distMatCell, trainLabel, wOpt, bOpt):
    numTest, numTrain = distMatColumn.shape
    classList = np.unique(trainLabel).tolist()
    distanceMat = np.zeros((numTest, numTrain))
    for classI in classList:
        classIidx = np.where(trainLabel == classI)[0]
        distanceMat[:, classIidx] = \
            (1 - wOpt[classI]) * distMatColumn[:, classIidx] + \
            wOpt[classI] * distMatCell[:, classIidx] + bOpt[classI]

    return distanceMat


def runTMOnSequence(tm, activeColumns, unionLength=1):
    numCells = tm.getCellsPerColumn() * tm.getColumnDimensions()[0] #总神经元的个数

    activeCellsTrace = []     #激活神经元
    predictiveCellsTrace = [] #预测神经元
    predictedActiveCellsTrace = []#预测神经元的并集
    activeColumnTrace = []        #激活柱的并集

    activationFrequency = np.zeros((numCells,))#每个神经元被激活的次数
    predictedActiveFrequency = np.zeros((numCells,))#每个神经元被预测的次数

    unionStepInBatch = 0
    unionBatchIdx = 0
    unionCells = set()
    unionCols = set()

    tm.reset()
    for t in range(len(activeColumns)): #遍历每个单词
        tm.compute(activeColumns[t], learn=False)
        activeCellsTrace.append(set(tm.getActiveCells())) #追加激活神经元
        predictiveCellsTrace.append(set(tm.getPredictiveCells()))#追加预测神经元
        if t == 0:
            predictedActiveCells = set()
        else:
            predictedActiveCells = activeCellsTrace[t].intersection(
                predictiveCellsTrace[t - 1])

        activationFrequency[tm.getActiveCells()] += 1             #激活的神经元次数加1
        predictedActiveFrequency[list(predictedActiveCells)] += 1 #预测的神经元次数加1

        unionCells = unionCells.union(predictedActiveCells)       #预测神经元的并集
        unionCols = unionCols.union(activeColumns[t])             #激活神经柱的并集

        unionStepInBatch += 1
        if unionStepInBatch == unionLength:
            predictedActiveCellsTrace.append(unionCells)
            activeColumnTrace.append(unionCols)
            unionStepInBatch = 0
            unionBatchIdx += 1
            unionCells = set()
            unionCols = set()

    if unionStepInBatch > 0:
        predictedActiveCellsTrace.append(unionCells)
        activeColumnTrace.append(unionCols)

    activationFrequency = activationFrequency / np.sum(activationFrequency)#每个神经元的激活频率
    predictedActiveFrequency = predictedActiveFrequency / np.sum(
        predictedActiveFrequency)#每个神经元的预测频率-预测的次数/所有预测神经元预测总和
    return (activeColumnTrace,
            predictedActiveCellsTrace,
            activationFrequency,
            predictedActiveFrequency)

"""返回所有句子的预测神经元并集，激活柱并集，激活频率和预测频率"""
def runTMOverDatasetFast(tm, activeColumns, unionLength=1):
    """
    Run encoder -> tm network over dataset, save activeColumn and activeCells
    traces
    :param tm:
    :param encoder:
    :param dataset:
    :return:
    """
    numSequence = len(activeColumns)  #句子个数
    predictedActiveCellsUnionTrace = []
    activationFrequencyTrace = []
    predictedActiveFrequencyTrace = []
    activeColumnUnionTrace = []

    for i in range(numSequence):
        (activeColumnTrace,
         predictedActiveCellsTrace,
         activationFrequency,
         predictedActiveFrequency) = runTMOnSequence(tm, activeColumns[i], unionLength)#计算第i句的激活柱并集，预测神经元并集，激活频率和预测频率

        predictedActiveCellsUnionTrace.append(predictedActiveCellsTrace)#所有句子的预测神经元并集
        activeColumnUnionTrace.append(activeColumnTrace)                #所有句子的激活柱并集
        activationFrequencyTrace.append(activationFrequency)            #所有句子的激活频率
        predictedActiveFrequencyTrace.append(predictedActiveFrequency)  #所有句子的预测频率
        # print "{} out of {} done ".format(i, numSequence)

    return (activeColumnUnionTrace,
            predictedActiveCellsUnionTrace,
            activationFrequencyTrace,
            predictedActiveFrequencyTrace)


def runEncoderOverDataset(encoder, dataset):
    activeColumnsData = []

    for i in range(dataset.shape[0]):
        activeColumnsTrace = []

        for element in dataset[i, :]:
            encoderOutput = encoder.encode(element)
            activeColumns = set(np.where(encoderOutput > 0)[0])
            activeColumnsTrace.append(activeColumns)

        activeColumnsData.append(activeColumnsTrace)
        # print "{} out of {} done ".format(i, dataset.shape[0])
    return activeColumnsData


def calcualteEncoderModelWorker(taskQueue, resultQueue, *args):
    while True:
        nextTask = taskQueue.get()
        print "Next task is : ", nextTask
        if nextTask is None:
            break
        nBuckets = nextTask["nBuckets"]
        accuracyColumnOnly = calculateEncoderModelAccuracy(nBuckets, *args)
        resultQueue.put({nBuckets: accuracyColumnOnly})
        print "Column Only model, Resolution: {} Accuracy: {}".format(
            nBuckets, accuracyColumnOnly)
    return


def calculateEncoderModelAccuracy(nBuckets, numCols, w, trainData, trainLabel):
    maxValue = np.max(trainData)
    minValue = np.min(trainData)

    resolution = (maxValue - minValue) / nBuckets
    encoder = RandomDistributedScalarEncoder(resolution, w=w, n=numCols)

    activeColumnsTrain = runEncoderOverDataset(encoder, trainData)
    distMatColumnTrain = calculateDistanceMatTrain(activeColumnsTrain)
    meanAccuracy, outcomeColumn = calculateAccuracy(distMatColumnTrain,
                                                    trainLabel, trainLabel)
    accuracyColumnOnly = np.mean(outcomeColumn)
    return accuracyColumnOnly


def searchForOptimalEncoderResolution(nBucketList, trainData, trainLabel, numCols, w):
    numCPU = multiprocessing.cpu_count()

    # Establish communication queues
    taskQueue = multiprocessing.JoinableQueue()
    resultQueue = multiprocessing.Queue()

    for nBuckets in nBucketList:
        taskQueue.put({"nBuckets": nBuckets})
    for _ in range(numCPU):
        taskQueue.put(None)
    jobs = []
    for i in range(numCPU):
        print "Start process ", i
        p = multiprocessing.Process(target=calcualteEncoderModelWorker,
                                    args=(taskQueue, resultQueue, numCols, w, trainData, trainLabel))
        jobs.append(p)
        p.daemon = True
        p.start()
        # p.join()
    # taskQueue.join()
    while not taskQueue.empty():
        time.sleep(0.1)
    accuracyVsResolution = np.zeros((len(nBucketList, )))
    while not resultQueue.empty():
        exptResult = resultQueue.get()
        nBuckets = exptResult.keys()[0]
        accuracyVsResolution[nBucketList.index(nBuckets)] = exptResult[nBuckets]

    return accuracyVsResolution