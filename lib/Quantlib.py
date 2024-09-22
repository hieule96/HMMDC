import numpy as np
import Quadtreelib as qd
import Transform as tf
import random as random
import DPCM
from huffman import huffman_compute
import pdb
#Default quantification matrix
MQ8x8 = np.zeros((8,8))
MQ8x8 = np.array( [[16,16,16,16,17,18,21,24],
[16,16,16,16,17,19,22,25],
[16,16,17,18,20,22,25,29],
[16,16,18,21,24,27,31,36],
[17,17,20,24,30,35,41,47],
[18,19,22,27,35,44,54,65],
[21,22,25,31,41,54,70,88],
[24,25,29,36,47,65,88,115]])

MQ4x4 = [[16,16,16,16],[16,16,16,16],[16,16,16,16],[16,16,16,16]]
ScalingQuant = [26214,23302,20560,18396,16384,14564]
ScalingDeQuant = [40,45,51,57,64,72]
def matQuant(a,b):
    Quant_matrix = np.zeros((a,b))
    if (a==4 and b==4):
        Quant_matrix = np.int16(MQ4x4)
    else:
        for i in range(8):
            for j in range (8):
                for k in range (int (a//8)):
                    for l in range ((b//8)):
                        Quant_matrix[(a//8)*i+k][(b//8)*j+l] = MQ8x8[i][j]
    return Quant_matrix

def QuantCU(dct_cu,QP):
    dct_cu = np.int32(dct_cu)
    log2shape = np.int32(np.log2(dct_cu.shape[0]))
    offsetQ = 1<<(log2shape - 6 + 8)
    shift2 = 29 - log2shape - 8
    sign = np.sign(dct_cu)
    dct_cu_Q = ((np.abs(dct_cu) *ScalingQuant[QP%6]* np.int16(16//matQuant(dct_cu.shape[0],dct_cu.shape[1])) + offsetQ) >> (QP//6))>>shift2
    dct_cu_Q = sign*dct_cu_Q
    return dct_cu_Q

def DeQuantCU(level_cu,QP):
    log2shape = np.int32(np.log2(level_cu.shape[0])) 
    offsetQ = 1<<(log2shape - 6 + 8)
    shift = log2shape - 5 + 8
    dct_cu = (level_cu*np.int32(matQuant(level_cu.shape[0],level_cu.shape[1]))*(ScalingDeQuant[QP%6]<<QP//6)+offsetQ)>>shift
    return dct_cu    

def quantCUSimple(dct_cu,QP):
    Q = 2 ** ((1 / 6) * (QP - 4))
    mQ = matQuant(dct_cu.shape[0],dct_cu.shape[1])/16
    dct_cu_Q = np.round(dct_cu/(mQ*Q))
    return dct_cu_Q
def deQuantCUSimple(level_cu,QP):
    Q = 2 ** ((1 / 6) * (QP - 4))
    mQ = matQuant(level_cu.shape[0],level_cu.shape[1])/16
    dct_cu_dQ= level_cu*mQ*Q
    return dct_cu_dQ
def LCU_CalRateQt(LCU,imgdct,AC_Remove = True):
    taille_dct = 0
    imgdct = imgdct.astype(int)
    for ctu in LCU.CTUs:
        for n in ctu:
            cu_x = n.get_x()
            cu_y = n.get_y()
            cu_h = n.get_height()
            cu_w = n.get_width()
            imgdct_CU = imgdct[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w].ravel()
            imgdct_AC = imgdct_CU[1:] if AC_Remove == True else imgdct_CU
            # (f, bi) = huffman_compute(imgdct_AC)
            unique,counts =np.unique(imgdct_AC,return_counts=True)
            probalitity = counts/len(imgdct_AC)
            entropy = -np.sum(probalitity*np.log2(probalitity))
            taille_dct = taille_dct + entropy*(len(imgdct_AC))
            # for j in range(np.size(unique_dct)):
            #     taille_dct = taille_dct + f[unique_dct[j]]*np.size(bi[unique_dct[j]])
    return taille_dct
def LCU_CalRateQt_DC_DPCM(LCU,imgdct):
    imgdct = imgdct.astype(int)
    DC_component_Img = []
    for ctu in LCU.CTUs:
        for cu in ctu:
            cu_x = cu.get_x()
            cu_y = cu.get_y()
            DC_component_Img.append(imgdct[cu_y][cu_x])
    print("Number of element DC %s", len(DC_component_Img))
    half_table = np.array([1,2,4,8,16,32,64,128])
    diff_table = np.hstack([half_table,-half_table])
    DPCM_codec = DPCM.DPCM(diff_table)
    symbol = DPCM_codec.encode(DC_component_Img)
    symbol_diff0 = symbol[symbol!=0]
    symbol_ones = len(symbol) - len(symbol_diff0)
    #Calculate Entropy:
    nb_bit_symbol = np.log2(symbol_diff0)
    return np.sum(nb_bit_symbol) + symbol_ones
    

def LCU_QuantDCTQ1Q2(LCU):
    newimgD1 = np.zeros((LCU.h,LCU.w))
    newimgD2 = np.zeros((LCU.h,LCU.w))
    newimgDC = np.zeros((LCU.h,LCU.w))
    imgdctDC = np.zeros((LCU.h,LCU.w))
    imgdctD1 = np.zeros((LCU.h,LCU.w))
    imgdctD2 = np.zeros((LCU.h,LCU.w))
    i = 0
    for ctu in LCU.CTUs:
        cus = qd.find_children(ctu.root)
        j = 0
        for cu in cus:
            matQ1 = matQuant(cu.get_width(),cu.get_height(),LCU.Qi1[i][j])
            matQ1[0,0] = 15
            matQ2 = matQuant(cu.get_width(),cu.get_height(),LCU.Qi2[i][j])
            matQ2[0,0] = 15
            img_dct_cu = tf.dct(cu.get_points(LCU.img))
            # matQ1 = np.full((cu.get_width(),cu.get_height()),LCU.Qi1[i][j])
            # matQ2 = np.full((cu.get_width(),cu.get_height()),LCU.Qi2[i][j])
            # matQ1 = 15
            # matQ2 = 15
            imgDi1 = np.around(img_dct_cu/matQ1)
            imgDi2 = np.around(img_dct_cu/matQ2)
            cu_x = cu.get_x()
            cu_y = cu.get_y()
            cu_h = cu.get_height()
            cu_w = cu.get_width()
            print(i,j,cu_x,cu_y,cu_h,cu_w,len(imgDi1))
            imgdctD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1
            imgdctD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi2
            imgdctDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2
            
            newimgDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2*matQ2)
            newimgD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1)
            newimgD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi2*matQ2)
            j=j+1
        i=i+1
    return imgdctDC,imgdctD1,imgdctD2,newimgDC,newimgD1,newimgD2
def convert_QP_Q(QP):
    Q_value = 2**((1/6)*(QP-4))
    return Q_value
def LCU_QuantDCTQP1QP2(LCU):
    newimgD1 = np.zeros((LCU.h,LCU.w))
    newimgD2 = np.zeros((LCU.h,LCU.w))
    newimgDC = np.zeros((LCU.h,LCU.w))
    imgdctDC = np.zeros((LCU.h,LCU.w))
    imgdctD1 = np.zeros((LCU.h,LCU.w))
    imgdctD2 = np.zeros((LCU.h,LCU.w))
    i = 0
    for ctu in LCU.CTUs:
        j = 0
        for cu in ctu:
            matQ1 = matQuant(cu.get_width(),cu.get_height())*convert_QP_Q(LCU.Qi1[i][j])
            matQ1[0,0] = 15
            matQ2 = matQuant(cu.get_width(),cu.get_height())*convert_QP_Q(LCU.Qi2[i][j])
            matQ2[0,0] = 15
            img_dct_cu = tf.dct(cu.get_points(LCU.img))
            # matQ1 = np.full((cu.get_width(),cu.get_height()),LCU.Qi1[i][j])
            # matQ2 = np.full((cu.get_width(),cu.get_height()),LCU.Qi2[i][j])
            # matQ1 = 15
            # matQ2 = 15
            imgDi1 = np.around(img_dct_cu/matQ1)
            imgDi2 = np.around(img_dct_cu/matQ2)
            cu_x = cu.get_x()
            cu_y = cu.get_y()
            cu_h = cu.get_height()
            cu_w = cu.get_width()
            #print(i,j,cu_x,cu_y,cu_h,cu_w,len(imgDi1))
            imgdctD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1
            imgdctD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi2
            imgdctDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = imgDi1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2
            
            newimgDC[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1 if LCU.Qi1[i][j]<LCU.Qi2[i][j] else imgDi2*matQ2)
            newimgD1[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi1*matQ1)
            newimgD2[cu_y:cu_y+cu_h,cu_x:cu_x+cu_w] = tf.idct(imgDi2*matQ2)
            j=j+1
        i=i+1
    return imgdctDC,imgdctD1,imgdctD2,newimgDC,newimgD1,newimgD2
