# -*- coding: utf-8 -*-
'''
@Time    : 2020/11/20
@Author  : Yun zhang
@FileName: HTMSequenceNew
@Software: PyCharm
'''
import nupic
import scipy.io as scio
from accuracyHTM import *
import numpy as np
import random

from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory


if __name__ == "__main__":

    """训练标签的加载"""
    mat_path = 'inputData1000.mat'
    load_mat = scio.loadmat(mat_path)
    trainSequence = load_mat['sequence']
    trainSequence = trainSequence - 1      #100*10，每个位置是单词i的标签从0-N-1
    repLen = 10                            #替换单词的数量0-9
    numTrain = len(trainSequence)          #训练句子的数量
    sequenceLength = len(trainSequence[0]) #每句多少个单词
    wordsNum = np.max(trainSequence)       #单词的总个数N-1

    """根据标签生成数据"""
    trainLabel = np.array(range(numTrain))    #训练句子的标签，每个句子一个标签
    testLabel = trainLabel                    #测试句子的标签，每个句子一个标签
    classList = np.unique(trainLabel).tolist()#类别的列表，每句一个标签
    numClass = len(classList)                 #类别的总数
    sample = np.random.random(np.max(trainSequence) + 1) * numTrain#为每个单词样本生成一个随机数
    trainData = np.zeros((numTrain, sequenceLength))               #初始化训练数据
    for i in range(numTrain):
        for j in range(sequenceLength):
            trainData[i][j] = sample[trainSequence[i][j]]          #将相应位置的类别的数据添加进训练数据


    CellDistAccuracy = np.zeros((repLen, numTrain))                     #测试方法1的正确率
    ActiveFreqDistAccuracy = np.zeros((repLen, numTrain))               #测试方法2的正确率
    PredActiveFreqDistAccuracy = np.zeros((repLen, numTrain))           #测试方法3的正确率

    for repNum in range(repLen):

        for trail in range(5):                                                    #100个trail，每次替换的位置都随机
            testSequence = trainSequence.copy()                                     #测试数据初使化为训练数据
            if repNum > 0 :
                for i  in range(numTrain):
                    currentSeq = testSequence[i]                                     #当前句子i
                    idx = np.sort(random.sample(range(0,sequenceLength), repNum))    #当前句子变化单词的位置
                    newWords = random.sample(range(0,wordsNum), repNum)              #生成新的单词
                    currentSeq[idx] = newWords                                       #将新单词替换这些位置的单词
                    testSequence[i] = currentSeq                                     #将新添加的替换单词后的句子放入测试标签类标


            """训练测试样本数，以及对应的测试数据"""
            numTest = len(testSequence)                                              #测试样本的数量
            testData = np.zeros((numTest, sequenceLength))                           #测试数据初始化
            for i in range(numTest):
                for j in range(sequenceLength):
                    testData[i][j] = sample[testSequence[i][j]]                      #填充测试数据

                """注释部分是用于判断训练数据和测试数据的相似度"""
                # print "Train Sample # {}, Test Sample # {}".format(numTrain, numTest)
                # print "Sequence Length {} Class # {}".format(sequenceLength, len(classList))

                # """判断标准1-欧式距离"""
                #
                # # EuclideanDistanceMat = calculateEuclideanDistanceMat(testData, trainData)
                # outcomeEuclidean = calculateEuclideanModelAccuracy(trainData, trainLabel,
                #                                                    testData, testLabel)#计算训练数据和测试句子的欧式距离，并对比预测标签和真实标签，正确为1
                # accuracyEuclideanDist = np.mean(outcomeEuclidean)
                # print
                # print "原始数据样本的欧式距离的相似度: {}".format(accuracyEuclideanDist)
                # print


            """对每个单词进行编码，用于训练数据和测试数据"""
            # 参数区域
            w = 41             #每个数据，例如testData[0][0]被编码成41维的数据，用于表示空间层的激活柱
            numCols = 1024  #所有柱的数量
            """注释部分是编码的参数优化"""
            # maxValue = np.max(trainData)
            # minValue = np.min(trainData)
            # nBucketList = range(20, 200, 10)
            # accuracyVsResolution = searchForOptimalEncoderResolution(
            # nBucketList, trainData, trainLabel, numCols, w)
            # optNumBucket = nBucketList[np.argmax(np.array(accuracyVsResolution))]
            # optimalResolution = (maxValue - minValue) / optNumBucket
            optimalResolution = 0.5
            encoder = RandomDistributedScalarEncoder(resolution=optimalResolution, w=w, n= numCols )#RDSE编码器
            encodeConsumption = []
            consumptions = []
            print "encoding train data ..."
            activeColumnsTrain = runEncoderOverDataset(encoder, trainData) #将trainData的每个数据编码成2048维的数据，表示的激活柱
            print "encoding test data ..."
            activeColumnsTest = runEncoderOverDataset(encoder, testData)   #将testData的每个数据编码成2048维的数据，表示的激活柱

            """柱的平均相似度"""
            # print "编码后，柱的平均相似度 ..."
            # distMatColumnTest = calculateDistanceMat(activeColumnsTest, activeColumnsTrain)#计算测试激活柱和训练激活柱之间的相似度=交集/并集
            # testAccuracyColumnOnly, outcomeColumn = calculateAccuracy(
            #     distMatColumnTest, trainLabel, testLabel)#返回与之最相似的句子，对比训练句子的标签，计算平均正确率testAccuracyColumnOnly，每个句子正确与否放在outcomeColumn
            #
            # print
            # print "柱的平均相似度为: {}".format(testAccuracyColumnOnly)

            """时间池初始化"""
            from nupic.bindings.algorithms import TemporalMemory as TemporalMemoryCPP

            tm = TemporalMemoryCPP(columnDimensions=(numCols,), #2048柱
                                   cellsPerColumn=32,           #每柱32个神经元
                                   permanenceIncrement=0.1,
                                   permanenceDecrement=0.1,
                                   predictedSegmentDecrement=0.01,
                                   minThreshold=10,
                                   activationThreshold=15,
                                   maxNewSynapseCount=20)

            # print
            # print "在句子上训练时间池... "

            numRepeatsBatch = 1
            numRptsPerSequence = 1
            """training,只有一个epoch"""
            for epoch in range(5):
                epoch
                for rpt in xrange(numRepeatsBatch):
                    # randomize the order of training sequences
                    randomIdx = np.random.permutation(range(numTrain))
                    for i in range(numTrain):
                        for _ in xrange(numRptsPerSequence):
                            for t in range(sequenceLength):
                                tm.compute(activeColumnsTrain[i][t], learn=True) #时间池的运行
                            tm.reset()                                           #清空激活神经元等信息
                        # print "Rpt: {}, {} out of {} 完成 ".format(rpt, i, trainData.shape[0])
                #for epoch in tqdm(range(1)): # tqdm 长循环中添加一个进度提示信息，训练1个epoch
                print("trainning is over")

            unionLength = 20
            # print "在训练数据上运行时间池,联合窗口为{}".format(unionLength)
            #所有句子的预测神经元并集，激活柱并集，激活频率和预测频率
            (activeColTrain,
             activeCellsTrain,
             activeFreqTrain,
             predActiveFreqTrain) = runTMOverDatasetFast(tm, activeColumnsTrain, unionLength)

            # # construct two distance matrices using training data
            # print("distMatColumnTrain")
            # distMatColumnTrain = calculateDistanceMat(activeColTrain, activeColTrain) #计算柱之间的相似度
            print("distMatCellTrain")
            distMatCellTrain = calculateDistanceMat(activeCellsTrain, activeCellsTrain)#计算神经元之间的相似度
            # print("distMatActiveFreqTrain")
            # distMatActiveFreqTrain = calculateDistanceMat(activeFreqTrain, activeFreqTrain)#计算激活频率之间的相似度
            # print("distMatPredActiveFreqTrain")
            # distMatPredActiveFreqTrain = calculateDistanceMat(predActiveFreqTrain, predActiveFreqTrain)#计算预测频率之间的相似度
            #
            # maxColumnOverlap = np.max(distMatColumnTrain) #柱之间的最大相似度
            maxCellOverlap = np.max(distMatCellTrain)     #神经元间的最大相似度
            # distMatColumnTrain /= maxColumnOverlap        #按比例缩放
            # distMatCellTrain /= maxCellOverlap            #按比例缩放

            print "在测试数据上运行时间池... "
            (activeColTest,
             activeCellsTest,
             activeFreqTest,
             predActiveFreqTest) = runTMOverDatasetFast(tm, activeColumnsTest, unionLength)
            print "计算准确率"
            print("distMatColumnTest")
            # distMatColumnTest = calculateDistanceMat(activeColTest, activeColTrain)
            print("distMatActiveFreqTest")
            distMatCellTest = calculateDistanceMat(activeCellsTest, activeCellsTrain)
            print("distMatActiveFreqTest")
            distMatActiveFreqTest = calculateDistanceMat(activeFreqTest, activeFreqTrain)
            print("distMatActiveFreqTest")
            distMatPredActiveFreqTest = calculateDistanceMat(predActiveFreqTest, predActiveFreqTrain)
            # distMatColumnTest /= maxColumnOverlap
            distMatCellTest /= maxCellOverlap

            classIdxMapTrain = {}      #哪些句子包含这个单词
            classIdxMapTest = {}

            testAccuracyCellOnly, outcomeCellOnly = calculateAccuracy(
                distMatCellTest, trainLabel, testLabel)
            CellDistAccuracy[repNum,trail] = testAccuracyCellOnly
            print "trail {}, Cell Dist accuracy {}".format(trail,testAccuracyCellOnly)

            testAccuracyActiveFreq, outcomeFreq = calculateAccuracy(
            distMatActiveFreqTest, trainLabel, testLabel)
            ActiveFreqDistAccuracy[repNum,trail] = testAccuracyActiveFreq
            print "Active Freq Dist Accuracy {}".format(testAccuracyActiveFreq)

            testAccuracyPredActiveFreq, outcomeFreq = calculateAccuracy(
            distMatPredActiveFreqTest, trainLabel, testLabel)
            PredActiveFreqDistAccuracy[repNum,trail] = testAccuracyPredActiveFreq
            print "Pred-Active Freq Dist Accuracy {}".format(testAccuracyPredActiveFreq)

    #     print "replace {} words, Mean Cell Dist accuracy {}".format(repNum,np.mean(CellDistAccuracy[repNum]))
    #     print "Mean Active Freq Dist Accuracy {}".format(np.mean(ActiveFreqDistAccuracy[repNum]))
    #     print "Mean Pred-Active Freq Dist Accuracy {}".format(np.mean(PredActiveFreqDistAccuracy[repNum]))
    #     print
    #
    # for repNum in range(repLen):
    #     print "replace {} words, Mean Cell Dist accuracy {}".format(repNum, np.mean(CellDistAccuracy[repNum]))
    #     print "Mean Active Freq Dist Accuracy {}".format(np.mean(ActiveFreqDistAccuracy[repNum]))
    #     print "Mean Pred-Active Freq Dist Accuracy {}".format(np.mean(PredActiveFreqDistAccuracy[repNum]))
    #     print



