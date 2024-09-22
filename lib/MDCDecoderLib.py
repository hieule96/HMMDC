#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:07:38 2022

@author: ubuntu
"""
import YUVLib
import numpy as np
from skimage import metrics
from skimage.metrics import structural_similarity as ssim
import Transform as tf
import numpy as np
import Metric
import pdb

class MDC_DEC_STATE:
    SEEK_FRAME = 0
    FRAME_PROCESSING = 1
    FRAME_WRITE = 2
def checkDecisionOracle(x,y,blockSizeH,blockSizeW,imgO,img1,img2):
    decision_MSE = 0
    mseblock1 = metrics.mean_squared_error(imgO._Y[y:y+blockSizeH,x:x+blockSizeW],img1._Y[y:y+blockSizeH,x:x+blockSizeW])
    mseblock2 = metrics.mean_squared_error(imgO._Y[y:y+blockSizeH,x:x+blockSizeW],img2._Y[y:y+blockSizeH,x:x+blockSizeW])
    if mseblock1<mseblock2:
        decision_MSE = 1
    elif mseblock1>mseblock2:
        decision_MSE = 2
    else:
        decision_MSE = 2
    return decision_MSE
class ParsingLoss:
    def __init__(self,filename):
        self.filename = filename
        self.tableFrameBlockLoss = [[]]
        self.POC = []
    def parse(self):
        with open(self.filename,"r") as file:
            for line in file:
                splitline = line.split(",")
                if (len(splitline)==1):
                    self.tableFrameBlockLoss.append([])
                    self.POC.append(int(splitline[0]))
                elif(len(splitline)==2):
                    self.tableFrameBlockLoss[len(self.tableFrameBlockLoss)-1].append([int (splitline[0]),int (splitline[1])])
    def getCTULossTable(self,POC):
        if POC in self.POC:
            pos = self.POC.index(POC)
            return self.tableFrameBlockLoss[pos]
        else:
            return []
    def getPOCTable(self):
        return self.POC
class DecoderMDCConfigFile():
    def __init__(self,yuvOrgFileName:str,reconD0FileName:str,reconD1FileName:str,
                 reconD2FileName:str,qpp:int, qpr:int, qTreeFileName:str="",q1FileName:str="",q2FileName=""):
        self.qTreeFileName = qTreeFileName
        self.reconD1FileName = reconD1FileName
        self.reconD2FileName = reconD2FileName
        self.reconD0FileName = reconD0FileName
        self.yuvOrgFileName = yuvOrgFileName
        self.q1FileName = q1FileName
        self.q2FileName = q2FileName
        self.qpp = qpp
        self.qpr = qpr
class PictureParamter():
    def __init__(self,w=0,h=0,frameToEncode=0):
        self.bord_h = h
        self.bord_w = w
        self.frameToEncode = frameToEncode
        self.step_w = np.ceil(w/64)
        self.step_h = np.ceil(h/64)
def writenpArrayToFile(YUVArray,outputFileName,mode='wb'):
    with open(outputFileName,mode) as output:
        output.write(YUVArray[0].tobytes())
        output.write(YUVArray[1].tobytes())
        output.write(YUVArray[2].tobytes())
def assignBlock(x,y,blockSizeH,blockSizeW,YUVJoint,Frame):
    x_half = x//2
    y_half = y//2
    blockSizeH_half = blockSizeH//2
    blockSizeW_half = blockSizeW//2
    YUVJoint[0][y:y+blockSizeH,x:x+blockSizeW] = Frame._Y[y:y+blockSizeH,x:x+blockSizeW]
    YUVJoint[2][y_half:y_half+blockSizeH_half,x_half:x_half+blockSizeW_half] = Frame._V[y_half:y_half+blockSizeH_half,x_half:x_half+blockSizeW_half] 
    YUVJoint[1][y_half:y_half+blockSizeH_half,x_half:x_half+blockSizeW_half] = Frame._U[y_half:y_half+blockSizeH_half,x_half:x_half+blockSizeW_half]
def addelementToClassifier(model,description1,description2,descriptioncentral,selection):
    #tranform to DCT
    dct1 = tf.dct(description1)
    dct2 = tf.dct(description2)
    if (selection==1):
        model.addelement(dct1.ravel(),1)
        model.addelement(dct2.ravel(),0)
    elif (selection==2):
        model.addelement(dct1.ravel(),0)
        model.addelement(dct2.ravel(),1)
def predictClassfier(model,CTU1,CTU2):
    decision = 0
    if (CTU1.shape==(32,64)):
        CTU1 = np.concatenate((CTU1,np.zeros((32,64))),axis=0)
        CTU2 = np.concatenate((CTU2,np.zeros((32,64))),axis=0)
    if (CTU1.shape==(64,32)):
        CTU1 = np.concatenate((CTU1,np.zeros((64,32))),axis=1)
        CTU2 = np.concatenate((CTU2,np.zeros((64,32))),axis=1)
    if (CTU1.shape==(32,32)):
        CTU1 = np.concatenate((CTU1,np.zeros((32,32))),axis=0)
        CTU2 = np.concatenate((CTU2,np.zeros((32,32))),axis=0)
        CTU1 = np.concatenate((CTU1,np.zeros((64,32))),axis=1)
        CTU2 = np.concatenate((CTU2,np.zeros((64,32))),axis=1)
    dct1 = tf.dct(CTU1)
    dct2 = tf.dct(CTU2)
    ans1 = model.predict(dct1.ravel().reshape(1,-1))
    ans2 = model.predict(dct2.ravel().reshape(1,-1))
    if (ans1[0]>ans2[0]):
        decision = ans1[0]
    elif (ans1[0]<ans2[0]):
        decision = ans2[0]
    return decision
class DecoderMDC:
    def __init__(self,decoderParam,pictureParam):
        self.decoderParam = decoderParam
        self.pictureParam = pictureParam
        self.POC1 = set()
        self.POC2 = set()
        self.MatchCTUtoXY = {}
    def resampleD1(self,descriptionFile,POCFile1,nbFrame):
        description1 = ParsingLoss(POCFile1)
        description1.parse()
        self.POC1 = description1.getPOCTable()
        countPOC = 0
        for j in range (0,nbFrame):
            imgD1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC)
            print (countPOC)

            if j in self.POC1:
                countPOC = countPOC + 1
            if (j == 0):
                YUVLib.writenpArrayToFile(imgD1,descriptionFile,'wb')
            else:
                YUVLib.writenpArrayToFile(imgD1,descriptionFile,'ab')
    def resampleD2(self,descriptionFile,POCFile2,nbFrame):
        description2 = ParsingLoss(POCFile2)
        description2.parse()
        self.POC2 = description2.getPOCTable()
        countPOC = 0
        for j in range (0,nbFrame):
            ## Note that the erronous frame can be the next or the acatual frame
            imgD2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC)
            if j in self.POC2:
                countPOC = countPOC + 1
            if (j == 0):
                YUVLib.writenpArrayToFile(imgD2,descriptionFile,'wb')
            else:
                YUVLib.writenpArrayToFile(imgD2,descriptionFile,'ab')
    def merge2Frame8x8(self,frameBegin,nbFrame):
        frame = 0
        P1_mean = 0
        P2_mean = 0
        P0_mean = 0
        PSNR_list = []
        for frame in range(frameBegin,frameBegin+nbFrame):
            img1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,frame)
            img2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,frame)
            imgO = YUVLib.read_YUV420_frame(open(self.decoderParam.yuvOrgFileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,frame)
            YUVJoint = [[],[],[]]
            YUVJoint[0] = np.zeros((self.pictureParam.bord_h,self.pictureParam.bord_w),dtype=np.uint8)
            YUVJoint[1] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            YUVJoint[2] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            for x0 in range(0,self.pictureParam.bord_w,32):
                for y0 in range(0,self.pictureParam.bord_h,32):
                    decision = checkDecisionOracle(x0,y0,32,32,imgO,img1,img2)
                    if decision == 1:
                        assignBlock(x0,y0,32,32,YUVJoint,img1)
                    elif decision == 2:
                        assignBlock(x0,y0,32,32,YUVJoint,img2)
            P1 = metrics.peak_signal_noise_ratio(imgO._Y,img1._Y,data_range=255)
            P2 = metrics.peak_signal_noise_ratio(imgO._Y,img2._Y,data_range=255)
            P0 = metrics.peak_signal_noise_ratio(imgO._Y,YUVJoint[0])
            P0_mean = P0 + P0_mean
            P1_mean = P1 + P1_mean
            P2_mean = P2 + P2_mean

            YUVJoint[0] = YUVJoint[0].ravel()
            YUVJoint[1] = YUVJoint[1].ravel()
            YUVJoint[2] = YUVJoint[2].ravel()
            if (frame == 0):
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'wb')
            else:
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'ab')
            # print ("WRITE_FRAME %s PNSR1: %s PSNR2: %s PSNR0: %s" %(frame,P1,P2,P0))
            PSNR_list.append(P0)
        P0_mean = P0_mean/nbFrame
        P1_mean = P1_mean/nbFrame
        P2_mean = P2_mean/nbFrame
        return P0_mean,P1_mean,P2_mean,PSNR_list
    def merge2Frame8x8withlostFrameOracle(self,nbFrame,POCFile1,POCFile2):
        P0_mean = 0
        P1_mean = 0 
        P2_mean = 0
        PSNR_list = []
        description1 = ParsingLoss(POCFile1)
        description1.parse()
        self.POC1 = description1.getPOCTable()
        description2 = ParsingLoss(POCFile2)
        description2.parse()
        self.POC2 = description2.getPOCTable()
        countPOC1 = 0
        countPOC2 = 0
        frame = 0
        YUVJoint = [[],[],[]]
        blockSize = 8
        countCTU = 0
        for y0 in range(0,self.pictureParam.bord_h,blockSize):
            for x0 in range(0,self.pictureParam.bord_w,blockSize):
                self.MatchCTUtoXY[(x0,y0)] = countCTU
                countCTU = countCTU + 1
        for i in range (0,nbFrame):
            imgO = YUVLib.read_YUV420_frame(open(self.decoderParam.yuvOrgFileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            YUVJoint[0] = np.zeros((self.pictureParam.bord_h,self.pictureParam.bord_w),dtype=np.uint8)
            YUVJoint[1] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            YUVJoint[2] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            if (i in self.POC1 and i in self.POC2):
                img1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC1)
                img2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC2)
                countPOC1 = countPOC1 + 1
                countPOC2 = countPOC2 + 1
                for y0 in range(0,self.pictureParam.bord_h,blockSize):
                    for x0 in range(0,self.pictureParam.bord_w,blockSize):
                        blkSizeH = blockSize if (self.pictureParam.bord_h-y0>blockSize) else self.pictureParam.bord_h-y0
                        blkSizeW = blockSize if (self.pictureParam.bord_w-x0>blockSize) else self.pictureParam.bord_w-x0
                        decisionOracle = checkDecisionOracle(x0,y0,blkSizeH,blkSizeW,imgO,img1,img2)
                        if (decisionOracle==1):
                            assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img1)
                        elif (decisionOracle==2):
                            assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img2)
                        else:
                            pdb.set_trace()
                P1 = metrics.peak_signal_noise_ratio(imgO._Y,img1._Y,data_range=255)
                P2 = metrics.peak_signal_noise_ratio(imgO._Y,img2._Y,data_range=255)
                P0 = metrics.peak_signal_noise_ratio(imgO._Y,YUVJoint[0],data_range=255)
            elif (i in self.POC1 and i not in self.POC2):
                img1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC1)
                countPOC1 = countPOC1 + 1
                P0 = P1 = metrics.peak_signal_noise_ratio(imgO._Y,img1._Y,data_range=255)
                P2 = 0
                YUVJoint[0] = img1._Y
                YUVJoint[2] = img1._V
                YUVJoint[1] = img1._U
            elif (i in self.POC2 and i not in self.POC1):
                img2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC2)
                countPOC2 = countPOC2 + 1
                P0 = P2 = metrics.peak_signal_noise_ratio(imgO._Y,img2._Y,data_range=255)
                P1 = 0
                YUVJoint[0] = img2._Y
                YUVJoint[2] = img2._V
                YUVJoint[1] = img2._U 
            else:
                P1 = 0
                P2 = 0
                P0 = 0         
            P0_mean = P0 + P0_mean
            P1_mean = P1 + P1_mean
            P2_mean = P2 + P2_mean
            YUVJoint[0] = YUVJoint[0].ravel()
            YUVJoint[1] = YUVJoint[1].ravel()
            YUVJoint[2] = YUVJoint[2].ravel()
            if (frame == 0):
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'wb')
            else:
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'ab')
            # print ("WRITE_FRAME %s PNSR1: %s PSNR2: %s PSNR0: %s" %(frame,P1,P2,P0))
            PSNR_list.append(P0)
            frame = frame + 1
        P0_mean = P0_mean/(frame+1)
        P1_mean = P1_mean/(frame+1)
        P2_mean = P2_mean/(frame+1)
        return P0_mean,P1_mean,P2_mean,self.decoderParam.qpp,self.decoderParam.qpr,PSNR_list 
    def merge2Frame8x8withlostFrameSansOracle(self,nbFrame,POCFile1,POCFile2):
        P0_mean = 0
        P1_mean = 0 
        P2_mean = 0
        PSNR_list = []
        countPOC1 = 0
        countPOC2 = 0
        frame = 0
        YUVJoint = [[],[],[]]
        blockSize = 64
        frameTminus1 = np.zeros((self.pictureParam.bord_h,self.pictureParam.bord_w),dtype=np.uint8)
        nbCaseSelection = 0
        precision = 0
        description1 = ParsingLoss(POCFile1)
        description1.parse()
        self.POC1 = description1.getPOCTable()
        description2 = ParsingLoss(POCFile2)
        description2.parse()
        self.POC2 = description2.getPOCTable()
        # test = NBClassifier()
        countArbitraire = 0
        nbCTU = 0
        for y0 in range(0,self.pictureParam.bord_h,blockSize):
            for x0 in range(0,self.pictureParam.bord_w,blockSize):
                self.MatchCTUtoXY[(x0,y0)] = nbCTU
                nbCTU = nbCTU + 1
        for i in range (0,nbFrame):
            imgO = YUVLib.read_YUV420_frame(open(self.decoderParam.yuvOrgFileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            YUVJoint[0] = np.zeros((self.pictureParam.bord_h,self.pictureParam.bord_w),dtype=np.uint8)
            YUVJoint[1] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            YUVJoint[2] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            mlecount = 0
            metricKL = Metric.DecisionSpatial(mesureMethod=Metric.MesureKL())
            if (i in self.POC1 and i in self.POC2):
                img1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC1)
                img2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC2)
                countPOC1 = countPOC1 + 1
                countPOC2 = countPOC2 + 1
                listCTUloss1 = description1.getCTULossTable(i)
                listCTUloss2 = description2.getCTULossTable(i)
                dictCTUloss1 = {}
                dictCTUloss2 = {}
                for j in listCTUloss1:
                        dictCTUloss1[j[0]] = j[1]
                for j in listCTUloss2:
                        dictCTUloss2[j[0]] = j[1]
                turn = -1
                for y0 in range(0,self.pictureParam.bord_h,blockSize):
                    turn = turn + 1
                    for x0 in range(0,self.pictureParam.bord_w,blockSize):
                        decisionQP = 0
                        blkSizeH = blockSize if (self.pictureParam.bord_h-y0>blockSize) else self.pictureParam.bord_h-y0
                        blkSizeW = blockSize if (self.pictureParam.bord_w-x0>blockSize) else self.pictureParam.bord_w-x0
                        decisionOracle = checkDecisionOracle(x0,y0,blkSizeH,blkSizeW,imgO,img1,img2)
                        if ((dictCTUloss1.get(self.MatchCTUtoXY[((np.ceil(self.pictureParam.bord_w/64)-1)*64,y0)],0)==1 
                            and dictCTUloss2.get(self.MatchCTUtoXY[((np.ceil(self.pictureParam.bord_w/64)-1)*64,y0)],0))==1):
                            if (turn%2 == 0):
                                decisionQP = 2
                            else:
                                decisionQP = 1
                        elif dictCTUloss1.get(self.MatchCTUtoXY[((np.ceil(self.pictureParam.bord_w/64)-1)*64,y0)],0)==1:
                                decisionQP = 1
                        elif dictCTUloss1.get(self.MatchCTUtoXY[((np.ceil(self.pictureParam.bord_w/64)-1)*64,y0)],0)==1:
                                decisionQP = 2
                        if (decisionQP==0):
                            # T = [0,0]
                            resultY = []
                            W1_Y,H1_Y,D1_Y = metricKL.decideHVD(x0,y0,blkSizeH,blkSizeW,img1._Y,img2._Y,YUVJoint[0])
                            if (frame>0):
                                T1_Y = metricKL.decideTemporal(x0,y0,blkSizeH,blkSizeW,img1._Y,img2._Y,frameTminus1)
                                if (len(T1_Y)==2):
                                    resultY.append(T1_Y)
                            if (len(W1_Y)==2):
                                resultY.append(W1_Y)                  
                            if (len(H1_Y)==2):
                                resultY.append(H1_Y)   
                            if (len(D1_Y)==2):
                                resultY.append(D1_Y)
                            decisionQP = metricKL.decideDescription(resultY)  
                        if (decisionQP==0):
                            if (turn%2 == 0):
                                decisionQP = 2
                            else:
                                decisionQP = 1
                            countArbitraire = countArbitraire + 1
                        if (decisionQP==1):
                            assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img1)
                        elif (decisionQP==2):
                            assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img2)
                        if ((decisionQP==decisionOracle) and decisionOracle!=0):
                            mlecount = mlecount + 1
                precision = precision + mlecount/nbCTU
                nbCaseSelection = nbCaseSelection + 1
                P1 = metrics.peak_signal_noise_ratio(imgO._Y,img1._Y,data_range=255)
                P2 = metrics.peak_signal_noise_ratio(imgO._Y,img2._Y,data_range=255)
                P0 = metrics.peak_signal_noise_ratio(imgO._Y,YUVJoint[0],data_range=255)
            elif (i in self.POC1 and i not in self.POC2):
                img1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC1)
                countPOC1 = countPOC1 + 1
                P0 = P1 = metrics.peak_signal_noise_ratio(imgO._Y,img1._Y,data_range=255)
                P2 = 0
                YUVJoint[0] = img1._Y
                YUVJoint[2] = img1._V
                YUVJoint[1] = img1._U
            elif (i in self.POC2 and i not in self.POC1):
                img2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,countPOC2)
                countPOC2 = countPOC2 + 1
                P0 = P2 = metrics.peak_signal_noise_ratio(imgO._Y,img2._Y,data_range=255)
                P1 = 0
                YUVJoint[0] = img2._Y
                YUVJoint[2] = img2._V
                YUVJoint[1] = img2._U 
            else:
                P1 = 0
                P2 = 0
                P0 = 0         
            P0_mean = P0 + P0_mean
            P1_mean = P1 + P1_mean
            P2_mean = P2 + P2_mean
            frameTminus1 = np.copy(YUVJoint[0])
            YUVJoint[0] = YUVJoint[0].ravel()
            YUVJoint[1] = YUVJoint[1].ravel()
            YUVJoint[2] = YUVJoint[2].ravel()
            if (frame == 0):
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'wb')
            else:
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'ab')
            # print (i,countPOC1,countPOC2)
            # print ("WRITE_FRAME %s PNSR1: %s PSNR2: %s PSNR0: %s" %(frame,P1,P2,P0))
            PSNR_list.append(P0)
            frame = frame + 1
        precision = precision / nbCaseSelection if nbCaseSelection>0 else 0
        print ("Prediction Accuracy",precision)
        P0_mean = P0_mean/(frame+1)
        P1_mean = P1_mean/(frame+1)
        P2_mean = P2_mean/(frame+1)
        return P0_mean,P1_mean,P2_mean,self.decoderParam.qpp,self.decoderParam.qpr,PSNR_list
    def merge2Frame8x8withlostFrameSansOracle_ffmpeg(self,nbFrame):
        P0_mean = 0
        P1_mean = 0 
        P2_mean = 0
        PSNR_list = []
        frame = 0
        YUVJoint = [[],[],[]]
        blockSize = 8
        nbCaseSelection = 0
        precision = 0
        countArbitraire = 0
        countCTU = np.ceil(self.pictureParam.bord_h/64) * np.ceil(self.pictureParam.bord_w/64)
        for i in range (0,nbFrame):
            imgO = YUVLib.read_YUV420_frame(open(self.decoderParam.yuvOrgFileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            YUVJoint[0] = np.zeros((self.pictureParam.bord_h,self.pictureParam.bord_w),dtype=np.uint8)
            YUVJoint[1] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            YUVJoint[2] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            mlecount = 0
            metricKL = Metric.DecisionSpatial(mesureMethod=Metric.MesureKL())
            img1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            img2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            turn = 0
            for y0 in range(0,self.pictureParam.bord_h,blockSize):
                turn = turn + 1
                for x0 in range(0,self.pictureParam.bord_w,blockSize):
                    decisionQP = 0
                    blkSizeH = blockSize if (self.pictureParam.bord_h-y0>blockSize) else self.pictureParam.bord_h-y0
                    blkSizeW = blockSize if (self.pictureParam.bord_w-x0>blockSize) else self.pictureParam.bord_w-x0
                    decisionOracle = checkDecisionOracle(x0,y0,blkSizeH,blkSizeW,imgO,img1,img2)
                    resultY = []
                    W1_Y,H1_Y,D1_Y = metricKL.decideHVD(x0,y0,blkSizeH,blkSizeW,img1._Y,img2._Y,YUVJoint[0])
                    # if (frame>0):
                    #     T1_Y = metricKL.decideTemporal(x0,y0,blkSizeH,blkSizeW,img1._Y,img2._Y,frameTminus1)
                    #     if (len(T1_Y)==2):
                    #         resultY.append(T1_Y)
                    if (len(W1_Y)==2):
                        resultY.append(W1_Y)                  
                    if (len(H1_Y)==2):
                        resultY.append(H1_Y)   
                    if (len(D1_Y)==2):
                        resultY.append(D1_Y)
                    # if (len(resultY)==0):
                    #     pdb.set_trace()
                    decisionQP = metricKL.decideDescription(resultY)  
                    if (decisionQP==0):
                        if (turn%2 == 0):
                            decisionQP = 2
                        else:
                            decisionQP = 1
                        countArbitraire = countArbitraire + 1
                    if (decisionQP==1):
                        assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img1)
                    elif (decisionQP==2):
                        assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img2)
                    # else:
                    #     pdb.set_trace()
                    if ((decisionQP==decisionOracle) and decisionOracle!=0):
                        mlecount = mlecount + 1
            precision = precision + mlecount/(self.pictureParam.bord_h/64)
            nbCaseSelection = nbCaseSelection + 1
            P1 = metrics.peak_signal_noise_ratio(imgO._Y,img1._Y,data_range=255)
            P2 = metrics.peak_signal_noise_ratio(imgO._Y,img2._Y,data_range=255)
            P0 = metrics.peak_signal_noise_ratio(imgO._Y,YUVJoint[0],data_range=255)      
            P0_mean = P0 + P0_mean
            P1_mean = P1 + P1_mean
            P2_mean = P2 + P2_mean
            frameTminus1 = np.copy(YUVJoint[0])
            YUVJoint[0] = YUVJoint[0].ravel()
            YUVJoint[1] = YUVJoint[1].ravel()
            YUVJoint[2] = YUVJoint[2].ravel()
            if (frame == 0):
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'wb')
            else:
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'ab')
            # print (i,countPOC1,countPOC2)
            print ("WRITE_FRAME %s PNSR1: %s PSNR2: %s PSNR0: %s" %(frame,P1,P2,P0))
            PSNR_list.append(P0)
            frame = frame + 1
        precision = precision / nbCaseSelection if nbCaseSelection>0 else 0
        print ("Prediction Accuracy",precision)
        P0_mean = P0_mean/(frame+1)
        P1_mean = P1_mean/(frame+1)
        P2_mean = P2_mean/(frame+1)
        return P0_mean,P1_mean,P2_mean,PSNR_list,precision
    def merge2Frame8x8withlostFrameOracle_ffmpeg(self,nbFrame):
        P0_mean = 0
        P1_mean = 0 
        P2_mean = 0
        PSNR_list = []
        frame = 0
        YUVJoint = [[],[],[]]
        blockSize = 8
        nbCaseSelection = 0
        precision = 0
        countArbitraire = 0
        countCTU = 0
        for y0 in range(0,self.pictureParam.bord_h,blockSize):
            for x0 in range(0,self.pictureParam.bord_w,blockSize):
                countCTU = countCTU + 1
        for i in range (0,nbFrame):
            imgO = YUVLib.read_YUV420_frame(open(self.decoderParam.yuvOrgFileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            YUVJoint[0] = np.zeros((self.pictureParam.bord_h,self.pictureParam.bord_w),dtype=np.uint8)
            YUVJoint[1] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            YUVJoint[2] = np.zeros((self.pictureParam.bord_h//2,self.pictureParam.bord_w//2),dtype=np.uint8)
            img1 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD1FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            img2 = YUVLib.read_YUV420_frame(open(self.decoderParam.reconD2FileName,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,i)
            for y0 in range(0,self.pictureParam.bord_h,blockSize):
                for x0 in range(0,self.pictureParam.bord_w,blockSize):
                    decisionQP = 0
                    blkSizeH = blockSize if (self.pictureParam.bord_h-y0>blockSize) else self.pictureParam.bord_h-y0
                    blkSizeW = blockSize if (self.pictureParam.bord_w-x0>blockSize) else self.pictureParam.bord_w-x0
                    decisionQP = checkDecisionOracle(x0,y0,blkSizeH,blkSizeW,imgO,img1,img2)
                    if (decisionQP==1):
                        assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img1)
                    elif (decisionQP==2):
                        assignBlock(x0,y0,blkSizeH,blkSizeW,YUVJoint,img2)
            P1 = metrics.peak_signal_noise_ratio(imgO._Y,img1._Y,data_range=255)
            P2 = metrics.peak_signal_noise_ratio(imgO._Y,img2._Y,data_range=255)
            P0 = metrics.peak_signal_noise_ratio(imgO._Y,YUVJoint[0],data_range=255)
            print ("WRITE_FRAME %s PNSR1: %s PSNR2: %s PSNR0: %s" %(frame,P1,P2,P0))

            P0_mean = P0 + P0_mean
            P1_mean = P1 + P1_mean
            P2_mean = P2 + P2_mean
            YUVJoint[0] = YUVJoint[0].ravel()
            YUVJoint[1] = YUVJoint[1].ravel()
            YUVJoint[2] = YUVJoint[2].ravel()
            if (i == 0):
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'wb')
            else:
                writenpArrayToFile(YUVJoint,self.decoderParam.reconD0FileName,'ab')
            PSNR_list.append(P0)
        P0_mean = P0_mean/(nbFrame)
        P1_mean = P1_mean/(nbFrame)
        P2_mean = P2_mean/(nbFrame)
        return P0_mean,P1_mean,P2_mean,PSNR_list,precision
    
class DecoderSDC:
    def __init__(self,orgfile:str,picparam:PictureParamter,qp:int,pocfile:str = "",outputfile:str=""):
        self.orgFile = orgfile
        self.outputFile = outputfile
        self.picparam = picparam
        self.pocFile = pocfile
        self.QP = qp
        self.POC = []
    def computePSNRSequence(self,frameBegin,nbFrame):
        frame = 0
        Pmean = 0
        PSNR_list = []
        for frame in range(frameBegin,frameBegin+nbFrame):
            imgO = YUVLib.read_YUV420_frame(open(self.orgFile,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,frame)
            imgSDC = YUVLib.read_YUV420_frame(open(self.outputFile,"rb"),self.pictureParam.bord_w,self.pictureParam.bord_h,frame)
            PSDC = metrics.peak_signal_noise_ratio(imgO._Y,imgSDC._Y,data_range=255)
            PSNR_list.append(PSDC)
            Pmean = Pmean + PSDC
        Pmean = Pmean/nbFrame
        return Pmean,PSNR_list
    def computePSNRwithmissing(self):
        countPOC = 0
        frame = 0
        Pmean = 0
        PSNR_list = []
        SDC = ParsingLoss(self.pocFile)
        SDC.parse()
        self.POC = SDC.getPOCTable()
        for j in range (0,self.picparam.frameToEncode-1):
            imgSDC = YUVLib.createZeros(self.picparam.bord_w,self.picparam.bord_h)
            if j in self.POC:
                imgSDC = YUVLib.read_YUV420_frame(open(self.outputFile,"rb"),self.picparam.bord_w,self.picparam.bord_h,countPOC)
                countPOC = countPOC + 1
                imgO = YUVLib.read_YUV420_frame(open(self.orgFile,"rb"),self.picparam.bord_w,self.picparam.bord_h,frame)
                PSDC = metrics.peak_signal_noise_ratio(imgO._Y,imgSDC._Y,data_range=255)
            else:
                PSDC = 0
            # if (frame == 0):
            #     YUVLib.writenpArrayToFile(imgSDC,self.outputFile,'wb')
            # else:
            #     YUVLib.writenpArrayToFile(imgSDC,self.outputFile,'ab')
            # print ("WRITE_FRAME %s PNSR:%s" %(frame,PSDC))
            PSNR_list.append(PSDC)
            Pmean = Pmean + PSDC
            frame = frame + 1
        Pmean = np.mean(PSNR_list)
        print (Pmean)
        return Pmean,self.QP,PSNR_list