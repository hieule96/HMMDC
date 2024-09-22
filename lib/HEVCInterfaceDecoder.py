#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 17:15:10 2022

@author: ubuntu
"""
import subprocess
import os,sys
class HEVCInterfaceDecoder:
    def __init__(self,inputFileName,outputFileName,qtreeDecFile):
        self.inputFileName=inputFileName
        self.outputFileName=outputFileName
        self.qtreeDecFile = qtreeDecFile
    def decodeHEVC(self):
        origWD = os.getcwd()
        relPathToLaunch = '../Lib'
        os.chdir(os.path.join(os.path.abspath(sys.path[0]), relPathToLaunch))
        process = subprocess.Popen(["./dec265","-c" ,"-o",self.outputFileName,"-a",self.qtreeDecFile,self.inputFileName],stdout=subprocess.PIPE)
        process.wait()
        os.chdir(origWD)
