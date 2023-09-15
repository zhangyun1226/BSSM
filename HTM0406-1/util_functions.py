#coding=utf-8
import copy
import os

import matplotlib.lines as lines
import numpy as np


def loadDataset(dataName, datasetName, useDeltaEncoder=False):
    fileDir = os.path.join('./{}'.format(datasetName),
                           dataName, dataName + '_TRAIN')
    trainData = np.loadtxt(fileDir, delimiter=',')
    trainLabel = trainData[:, 0].astype('int')
    trainData = trainData[:, 1:]

    fileDir = os.path.join('./{}'.format(datasetName),
                           dataName, dataName + '_TEST')
    testData = np.loadtxt(fileDir, delimiter=',')
    testLabel = testData[:, 0].astype('int')
    testData = testData[:, 1:]

    if useDeltaEncoder:
        trainData = np.diff(trainData)
        testData = np.diff(testData)

    classList = np.unique(trainLabel)
    classMap = {}
    for i in range(len(classList)):
        classMap[classList[i]] = i

    for i in range(len(trainLabel)):
        trainLabel[i] = classMap[trainLabel[i]]
    for i in range(len(testLabel)):
        testLabel[i] = classMap[testLabel[i]]

    return trainData, trainLabel, testData, testLabel


def listDataSets(datasetName):
    dataSets = [d for d in os.listdir('./{}'.format(datasetName)) if os.path.isdir(
        os.path.join('./{}'.format(datasetName), d))]
    return dataSets

"""计算与训练句子最相似的测试句子的标签，若标签一致，则为1，最后返回平均相似度和最终的结果"""
def calculateAccuracy(distanceMat, trainLabel, testLabel):
    outcome = []
    for i in range(len(testLabel)):
        predictedClass = trainLabel[np.argmax(distanceMat[i, :])] #获取第i句所对应的最大相似度的句子的标签-预测类别
        correct = 1 if predictedClass == testLabel[i] else 0      #如果这个预测类别和测试类别一致，就正确为1，否则为0
        outcome.append(correct)                                   #把正确率加上
    accuracy = np.mean(np.array(outcome))                         #计算平均正确率
    return accuracy, outcome                                      #返回平均正确率和具体的正确列表

"""找出与测试句子欧式距离最近的训练句子，并和真实的测试句子的标签作对比，正确为1，错误为0"""
def calculateEuclideanModelAccuracy(trainData, trainLabel, testData, testLabel):
    outcomeEuclidean = []
    for i in range(testData.shape[0]):
        predictedClass = one_nearest_neighbor(trainData, trainLabel, testData[i, :])#测试句子i（10维）和哪个训练句子的欧式距离最近
        correct = 1 if predictedClass == testLabel[i] else 0                        #如果预测的标签和真实的一致，就正确
        outcomeEuclidean.append(correct)                                            #追加，正确1，错误0
    return outcomeEuclidean

"""计算每个句子的数据与unknownSequence-测试数据的欧式距离，返回最小的欧式距离的训练句子的标签"""
def one_nearest_neighbor(trainData, trainLabel, unknownSequence):
    """
    One nearest neighbor with Euclidean Distance
    @param trainData (nSample, NT) training data
    @param trainLabel (nSample, ) training data labels
    @param unknownSequence (1, NT) sequence to be classified
    """
    distance = np.zeros((trainData.shape[0],))
    for i in range(trainData.shape[0]):
        distance[i] = np.sqrt(np.sum(np.square(trainData[i, :] - unknownSequence)))
    predictedClass = trainLabel[np.argmin(distance)]
    return predictedClass


def sortDistanceMat(distanceMat, trainLabel, testLabel):
    """
    Sort Distance Matrix according to training/testing class labels such that
    nearby entries shares same class labels
    :param distanceMat: original (unsorted) distance matrix
    :param trainLabel: list of training labels
    :param testLabel: list of testing labels
    :return:
    """
    numTrain = len(trainLabel)
    numTest = len(testLabel)
    sortIdxTrain = np.argsort(trainLabel)
    sortIdxTest = np.argsort(testLabel)
    distanceMatSort = np.zeros((numTest, numTrain))

    for i in xrange(numTest):
        for j in xrange(numTrain):
            distanceMatSort[i, j] = distanceMat[sortIdxTest[i], sortIdxTrain[j]]

    return distanceMatSort


def smoothArgMax(array):
    idx = np.where(array == np.max(array))[0]
    return np.median(idx).astype('int')


def calculateClassLines(trainLabel, testLabel, classList):
    sortIdxTrain = np.argsort(trainLabel)
    sortIdxTest = np.argsort(testLabel)
    vLineLocs = []
    hLineLocs = []
    for c in classList[:-1]:
        hLineLocs.append(np.max(np.where(testLabel[sortIdxTest] == c)[0]) + .5)
        vLineLocs.append(np.max(np.where(trainLabel[sortIdxTrain] == c)[0]) + .5)
    return vLineLocs, hLineLocs


def addClassLines(ax, vLineLocs, hLineLocs):
    for vline in vLineLocs:
        ax.add_line(lines.Line2D([vline, vline], ax.get_ylim(), color='k'))
    for hline in hLineLocs:
        ax.add_line(lines.Line2D(ax.get_xlim(), [hline, hline], color='k'))

"""计算欧式距离，每个训练句子和每个测试句子的数据相似度"""
def calculateEuclideanDistanceMat(testData, trainData):
    EuclideanDistanceMat = np.zeros((testData.shape[0], trainData.shape[0]))
    for i in range(testData.shape[0]):
        for j in range(trainData.shape[0]):
            EuclideanDistanceMat[i, j] = np.sqrt(np.sum(
                np.square(testData[i, :] - trainData[j, :])))

    return EuclideanDistanceMat

"""两个向量的相似度=交集/并集"""
def overlapDist(s1, s2):
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2))) / len(s1.union(s2))

"""计算两个向量之间的相似度，交集/并集"""
def calculateDistanceMat(activeColumnsTest, activeColumnsTrain):
    nTest = len(activeColumnsTest)    #测试样本数
    nTrain = len(activeColumnsTrain)  #训练样本数
    sequenceLength = len(activeColumnsTrain[0]) #单词个数
    activeColumnOverlapTest = np.zeros((nTest, nTrain))  #训练句子和测试句子之间的相似度

    for i in range(nTest):
        for j in range(nTrain):
            if type(activeColumnsTest[0]) is np.ndarray:
                activeColumnOverlapTest[i, j] = np.sum(
                    np.sqrt(np.multiply(activeColumnsTest[i], activeColumnsTrain[j])))
                # activeColumnOverlapTest[i, j] = np.sum(np.minimum(activeColumnsTest[i], activeColumnsTrain[j]))
            else:
                #句子i和句子j的每个单词的相似度相加之和
                for t in range(sequenceLength):
                    # print(i,j,t)
                    activeColumnOverlapTest[i, j] += overlapDist(
                        activeColumnsTest[i][t], activeColumnsTrain[j][t]) #句子中每个单词的激活柱的相似度，两单词的交集/并集

    return activeColumnOverlapTest


def calculateDistanceMatTrain(activeColumnsTrain):
    nTrain = len(activeColumnsTrain)
    sequenceLength = len(activeColumnsTrain[0])
    activeColumnOverlap = np.zeros((nTrain, nTrain))

    for i in range(nTrain):
        for j in range(i + 1, nTrain):
            for t in range(sequenceLength):
                activeColumnOverlap[i, j] += len(
                    activeColumnsTrain[i][t].intersection(activeColumnsTrain[j][t]))
            activeColumnOverlap[j, i] = activeColumnOverlap[i, j]
    return activeColumnOverlap


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


def costFuncSharedW(newW, w, b, distMatColumn, distMatCell,
                    trainLabel, classList):
    wTest = copy.deepcopy(w)
    for classI in classList:
        wTest[classI] = newW

    distanceMatXV = constructDistanceMat(
        distMatColumn, distMatCell, trainLabel, wTest, b)

    accuracy, outcome = calculateAccuracy(distanceMatXV, trainLabel, trainLabel)
    return -accuracy


def costFuncW(newW, classI, w, b, activeColumnOverlap, activeCellOverlap, trainLabel, classList):
    wTest = copy.deepcopy(w)
    wTest[classList[classI]] = newW

    numXVRpts = 10
    accuracyRpt = np.zeros((numXVRpts,))
    for rpt in range(numXVRpts):
        (activeColumnOverlapXV, activeCellOverlapXV,
         trainLabelXV, trainLabeltrain) = generateNestedXCdata(
            trainLabel, activeColumnOverlap, activeCellOverlap, seed=rpt)

        distanceMat = constructDistanceMat(
            activeColumnOverlapXV, activeCellOverlapXV, trainLabeltrain, wTest, b)

        accuracy, outcome = calculateAccuracy(
            distanceMat, trainLabeltrain, trainLabelXV)
        accuracyRpt[rpt] = accuracy
    return -np.mean(accuracyRpt)


def costFuncB(newB, classI, w, b, activeColumnOverlap, activeCellOverlap, trainLabel, classList):
    bTest = copy.deepcopy(b)
    bTest[classList[classI]] = newB

    numXVRpts = 10
    accuracyRpt = np.zeros((numXVRpts,))
    for rpt in range(numXVRpts):
        (activeColumnOverlapXV, activeCellOverlapXV,
         trainLabelXV, trainLabeltrain) = generateNestedXCdata(
            trainLabel, activeColumnOverlap, activeCellOverlap, seed=rpt)

        distanceMat = constructDistanceMat(
            activeColumnOverlapXV, activeCellOverlapXV, trainLabeltrain, w, bTest)

        accuracy, outcome = calculateAccuracy(
            distanceMat, trainLabeltrain, trainLabelXV)
        accuracyRpt[rpt] = accuracy

    return -np.mean(accuracyRpt)


def prepareClassifierInput(distMatColumn, distMatCell, classList, classLabel, options):
    classIdxMap = {}
    for classIdx in classList:
        classIdxMap[classIdx] = np.where(classLabel == classIdx)[0]

    classifierInput = []
    numSample, numTrain = distMatColumn.shape#测试样本和训练样本的个数
    classList = classIdxMap.keys()
    numClass = len(classList) #类别的个数
    for i in range(numSample):
        if options['useColumnRepresentation']:
            columnNN = np.zeros((numClass,))
        else:
            columnNN = np.array([])

        if options['useCellRepresentation']:
            cellNN = np.zeros((numClass,))
        else:
            cellNN = np.array([])

        for classIdx in classList:
            if options['useColumnRepresentation']:
                columnNN[classIdx] = np.max(
                    distMatColumn[i, classIdxMap[classIdx]])
            if options['useCellRepresentation']:
                cellNN[classIdx] = np.max(distMatCell[i, classIdxMap[classIdx]])

        # if options['useColumnRepresentation']:
        #   columnNN[columnNN < np.max(columnNN)] = 0
        #   columnNN[columnNN == np.max(columnNN)] = 1
        #
        # if options['useCellRepresentation']:
        #   cellNN[cellNN < np.max(cellNN)] = 0
        #   cellNN[cellNN == np.max(cellNN)] = 1
        # classifierInput.append(np.concatenate((columnNN, cellNN)))

    return classifierInput


def generateNestedXCdata(trainLabel, distMatColumn, distMatCell,
                         seed=1, xcPrct=0.5):
    """
    Set aside a portion of the training data for nested cross-validation
    :param trainLabel:
    :param distMatColumn:
    :param distMatCell:
    :param xcPrct:
    :return:
    """
    np.random.seed(seed)
    randomIdx = np.random.permutation(len(trainLabel))

    numXVsamples = int(len(trainLabel) * xcPrct)
    numTrainSample = len(trainLabel) - numXVsamples

    selectXVSamples = randomIdx[:numXVsamples]
    selectTrainSamples = randomIdx[numXVsamples:]
    selectXVSamples = np.sort(selectXVSamples)
    selectTrainSamples = np.sort(selectTrainSamples)

    distMatColumnXV = np.zeros((numXVsamples, numTrainSample))
    distMatCellXV = np.zeros((numXVsamples, numTrainSample))
    for i in range(numXVsamples):
        distMatColumnXV[i, :] = distMatColumn[
            selectXVSamples[i], selectTrainSamples]
        distMatCellXV[i, :] = distMatCell[
            selectXVSamples[i], selectTrainSamples]

    trainLabelXV = trainLabel[selectXVSamples]
    trainLabeltrain = trainLabel[selectTrainSamples]

    return (distMatColumnXV, distMatCellXV,
            trainLabelXV, trainLabeltrain)