#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:39:16 2022

@author: ubuntu
"""

import subprocess
import re
import sys
import os
#HEVC backend
class HEVCEncoder:
    def __init__(self,encoderConfigFile,
                 srcConfigFile,
                 bsFile,recFile,qpFile,qpmean,
                 qtreeFileName=""):
        self.qtreeFileName = qtreeFileName
        self.encoderConfigFile = encoderConfigFile
        self.srcConfigFile = srcConfigFile
        self.bsFile = bsFile
        self.recFile = recFile
        self.qpFile = qpFile
        self.qpmean = qpmean
    def runHEVCEncoder(self,qpmean):
        origWD = os.getcwd()
        relPathToLaunch = '../Lib'
        os.chdir(os.path.join(os.path.abspath(sys.path[0]), relPathToLaunch))
        process = subprocess.Popen(["./TAppEncoder","-c" ,self.encoderConfigFile,"-c",self.srcConfigFile,"-q","%d" %(qpmean)],stdout=subprocess.PIPE)
        print ("Exec ./TAppEncoder -c" + self.encoderConfigFile + "-c" + self.srcConfigFile +"-q"+"%d"%(qpmean))
        bitrate = 0
        while True:
          output = process.stdout.readline()
          if process.poll() is not None:
            break
          if output:
            outputparse = output.strip().decode("utf-8")
            parse_result = re.match("POC *\s *(-?[0-9]+) .*\  *(-?[0-9]+) bits",outputparse)
            if (parse_result != None):
                bitrate = parse_result.group(parse_result.lastindex)
        return bitrate
        os.chdir(origWD)
    def runHEVCEncoderNoVerbose(self):
        origWD = os.getcwd()
        relPathToLaunch = '../Lib'
        os.chdir(os.path.join(os.path.abspath(sys.path[0]), relPathToLaunch))
        subprocess.run(["./TAppEncoder","-c" ,self.encoderConfigFile,"-c",self.srcConfigFile,"-q","%d" %(self.qpmean),"-b",self.bsFile,"-o",self.recFile],stdout=subprocess.DEVNULL)
        os.chdir(origWD)
    def runHEVCEncoderNoVerbose2(self):
        origWD = os.getcwd()
        relPathToLaunch = '../Lib'
        os.chdir(os.path.join(os.path.abspath(sys.path[0]), relPathToLaunch))
        subprocess.run(["./TAppEncoder","-c" ,self.encoderConfigFile,"-c",self.srcConfigFile
                        ,"-q","%d" %(self.qpmean),"-b",self.bsFile,"-o",self.recFile,"-qf",self.qpFile],stdout=subprocess.DEVNULL)
        os.chdir(origWD)
    def generateQtreeEncoderPSNRMSE(self,qp,residualOutputFileName,qtreeFileName):
        process = subprocess.Popen(["./TAppEncoder","-c" ,self.encoderConfigFile,"-c",self.srcConfigFile,"-qt", qtreeFileName,"-resi",residualOutputFileName, "-q"," %d"%(qp)],stdout=subprocess.PIPE)
        bitrate = 0
        psnr_Y = 0
        mse_Y = 0
        while True:
          output = process.stdout.readline()
          if process.poll() is not None:
            break
          if output:
            outputparse = output.strip().decode("utf-8")
            parse_result = re.match("POC *\s (-?[0-9]+) .*\s *(-?[0-9]+) bits \[Y *(-?[0-9.]+) dB *\s U *(-?[0-9.]+) dB *\s V (-?[0-9.]+) dB\] \[Y MSE *(-?[0-9.]+) *\s U MSE *(-?[0-9.]+) *\s V MSE *\s *(-?[0-9.]+)\]",outputparse)
            if (parse_result != None):
                bitrate = parse_result.group(2) 
                psnr_Y = parse_result.group(3) 
                mse_Y = parse_result.group(6)
        return bitrate,psnr_Y,mse_Y
class MDCEncoder2Description:
    def __init__(self,qtreeFileNameD1,encoderConfigFileD1,srcConfigFileD1,
                 qtreeFileNameD2,encoderConfigFileD2,srcConfigFileD2):
        self.HEVCEncoder1 = HEVCEncoder(qtreeFileNameD1,encoderConfigFileD1,srcConfigFileD1)
        self.HEVCEncoder2 = HEVCEncoder(qtreeFileNameD2,encoderConfigFileD2,srcConfigFileD2)
    def mdcEncode(self,qp1,qp2):
        size1=self.HEVCEncoder1.runHEVCEncoder(qp1)
        size2=self.HEVCEncoder2.runHEVCEncoder(qp2)
        return size1,size2
