#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:46:31 2022

@author: ubuntu
"""
from __future__ import annotations
from pickletools import uint8

import numpy as np
from scipy.stats import pearsonr,spearmanr,norm
from skimage import metrics
import Transform as tf
import pdb
from abc import ABC, abstractmethod
from sklearn.metrics import mutual_info_score
from skimage.metrics import structural_similarity as ssim
from scipy.special import rel_entr,kl_div
from sklearn import preprocessing

def kl_calculate(array_ref,array_target):
    # hist_ref,_ = np.histogram(array_ref,bins=bins)
    # hist_target,_ = np.histogram(array_target,bins=bins)
    array_ref = np.array(array_ref,dtype=np.uint8)
    array_target = np.array(array_target,dtype=np.uint8)
    try:
        hist_ref = np.bincount(array_ref,minlength=256)
    except TypeError:
        print(array_ref)
    try:
        hist_target = np.bincount(array_target,minlength=256)
    except TypeError:
        print (array_target)
    #Zero frequency problem
    hist_ref = hist_ref + 1
    hist_target = hist_target + 1
    #Convert to Prob
    p_hist_ref = hist_ref / (len(array_ref) + 256)
    p_hist_target = hist_target / (len(array_target) + 256)
    #Do the intergral (rectangle method)
    kl_distance=sum(kl_div(p_hist_ref,p_hist_target))
    # for i in range(0,256):
    #     countMatch = countMatch + 1
    #     kl_distance = kl_distance+(p_hist_ref[i]*np.log(p_hist_ref[i])/p_hist_target[i])
    # # Two distribution is totaly decorrelated.
    # if (countMatch==0):
    #     return float("+inf")
    assert(kl_distance!=float("+inf"))
    return kl_distance
def selectMin(result:list)->int:
    min1 = float ('+inf')
    min2 = float ('+inf')
    decisionQP = 0
    for element in result:
        if (len(element)>0):
            min1 = min (min1,element[0])
            min2 = min (min2,element[1])
    if (min1<min2):
        decisionQP = 1
    elif(min1>min2):
        decisionQP = 2
    return decisionQP
def selectMax(result:list)->int:
    max1 = 0
    max2 = 0
    decisionQP = 0
    for element in result:
        if (len(element)>0):
            max1 = max (max1,element[0])
            max2 = max (max2,element[1])
        if (max1>max2):
            decisionQP = 1
        elif(max1<max2):
            decisionQP = 2
    return decisionQP

def selectMinMean(result:list)->int:
    decisionQP = 0
    mean1 = 0
    mean2 = 0
    for element in result:
        if (len(element)>0):
            if (element[0]!=0 and element[0]!=float("+inf") 
                and element[1]!=0 and element[1]!=float("+inf")):
                mean1 = mean1+element[0]
                mean2 = mean2+element[1]
        if (mean1<mean2):
            decisionQP = 1
        elif(mean1>mean2):
            decisionQP = 2
    return decisionQP
class Mesure(ABC):
    @abstractmethod
    def do_mesure(self,imgRef,imgComp):
        pass
    def select_best_quality(self,result:list):
        pass
class MesureKL(Mesure):
    def do_mesure(self,imgRef,imgComp) -> int:
        return (abs(kl_calculate(imgRef,imgComp))+abs(kl_calculate(imgComp,imgRef)))/2
    def select_best_quality(self,result:list)->int:
        return selectMin(result)
class MesureCorrelation(Mesure):
    def do_mesure(self,imgRef,imgComp) -> int:
        corr,_ = pearsonr(imgRef,imgComp)
        if (np.isnan(corr)):
            corr = 0
        corr = abs(corr)
        return corr
    def select_best_quality(self,result:list)->int:
        return selectMax(result)
class MesureMSE(Mesure):
    def do_mesure(self,imgRef,imgComp) -> int:
        return abs(metrics.mean_squared_error(imgRef,imgComp))
    def select_best_quality(self,result:list)->int:
        return selectMin(result)
class MesureEuler(Mesure):
    def do_mesure(self, imgRef, imgComp) -> int:
        return abs(np.linalg.norm(imgRef-imgComp))
    def select_best_quality(self,result:list)->int:
        return selectMin(result)
class MesureMutualInformationScore(Mesure):
    def do_mesure(self, imgRef, imgComp) -> int:
        scaler = preprocessing.MinMaxScaler()
        norm_minmax = scaler.fit(imgRef.reshape(-1,1))
        norm_imgRef = norm_minmax.transform(imgRef.reshape(-1,1))
        norm_imgComp = norm_minmax.transform(imgComp.reshape(-1,1))
        return mutual_info_score(norm_imgRef.ravel(),norm_imgComp.ravel())
    def select_best_quality(self,result:list)->int:
        return selectMax(result)
class MesureSSIM(Mesure):
    def do_mesure(self, imgRef, imgComp) -> int:
        return ssim(imgRef,imgComp,data_range=255)
    def select_best_quality(self,result:list)->int:
        return selectMax(result)    
class Decision():
    def __init__(self,mesureMethod:Mesure)->None:
        self._mesureMethod = mesureMethod
        self.mesureH=[]
        self.mesureW=[]
        self.mesureD=[]
        self.mesureT=[]
    @abstractmethod
    def decideHVD(self,x,y,blockSizeH,blockSizeW,img1,img2,imgC):
        pass
    @abstractmethod    
    def decideTemporal(self,x,y,blockSizeH,blockSizeW,img1,img2,imgTminus1):
        pass
    def decideDescription(self,result:list):
        return self._mesureMethod.select_best_quality(result)
    def decideDescriptionYUV(self,resultY:list,resultU:list, resultV:list):
        kl_yuv = []
        for y,u,v in zip (resultY,resultU,resultV):
            kl_yuv.append(y+u+v)
        return self._mesureMethod.select_best_quality(kl_yuv)
class DecisionSpatial(Decision):
    def decideHVD(self,x,y,blockSizeH,blockSizeW,img1,img2,imgC):
        D1 = img1[y:y+blockSizeH,x:x+blockSizeW].flatten()
        D2 = img2[y:y+blockSizeH,x:x+blockSizeW].flatten()
        self.mesureW = []
        self.mesureH = []
        self.mesureD = []
        if (x>0):
            ref = imgC[y:y+blockSizeH,x-blockSizeW:x].flatten()
            self.mesureW = self.doMesure(ref,D1,D2)
        if (y>0):
            ref = imgC[y-blockSizeH:y,x:x+blockSizeW].flatten()
            ref = ref.flatten()
            self.mesureH = self.doMesure(ref,D1,D2)
        if (x>0 and y>0):
            ref = imgC[y-blockSizeH:y,x-blockSizeW:x].flatten()
            ref = ref.flatten()
            self.mesureD = self.doMesure(ref,D1,D2)
        return self.mesureW,self.mesureH,self.mesureD
    def decideTemporal(self,x,y,blockSizeH,blockSizeW,img1,img2,imgC):
        self.mesureT=[]
        D1 = img1[y:y+blockSizeH,x:x+blockSizeW]
        D1 = D1.flatten()
        D2 = img2[y:y+blockSizeH,x:x+blockSizeW]
        D2 = D2.flatten()
        ref = imgC[y:y+blockSizeH,x:x+blockSizeW]
        ref=ref.flatten()
        self.mesureT = self.doMesure(ref,D1,D2)
        return self.mesureT
    def doMesure(self,ref,D1,D2):
        return  [self._mesureMethod.do_mesure(ref, D1),
                 self._mesureMethod.do_mesure(ref, D2)]