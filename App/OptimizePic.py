# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 23:54:54 2021

@author: LE Trung Hieu
"""

import numpy as np
import sys, os

import re
import time
import pdb
import pathlib
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from typing import TypedDict, Optional
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.Quadtreelib import Node, Image, import_subdivide
from lib.Optimizer import Optimizer_curvefitting, OptimizerParameter, OptimizerOutput
import lib.Transform as tf


class FrameYUV(TypedDict):
    Y: np.ndarray
    U: Optional[np.ndarray]
    V: Optional[np.ndarray]


def writenpArrayToFile(YUVArray, outputFileName, mode='wb'):
    with open(outputFileName, mode) as output:
        output.write(YUVArray[0].tobytes())
        output.write(YUVArray[2].tobytes())
        output.write(YUVArray[1].tobytes())


def read_YUV420_frame(fid, width, height, frame=0) -> FrameYUV:
    # read a frame from a YUV420-formatted sequence
    d00 = height // 2
    d01 = width // 2
    fid.seek(frame * (width * height + width * height // 2))
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    out: FrameYUV = {
        'Y': Y,
        'U': U,
        'V': V
    }
    return out


def read_YUV400_frame(fid, width, height, frame=0):
    # read a frame from a YUV420-formatted sequence
    fid.seek(frame * (width * height + width * height // 2))
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    out: FrameYUV = {
        'Y': Y,
        'U': None,
        'V': None,
    }
    return out


def mseToPSNR(mse):
    if mse != 0:
        return 10 * np.log10(255 ** 2 / mse)
    else:
        return float('+inf')


def buildModel(frame_enc: int, CTU_path: str, yuv_files: str, bord_h: int, bord_w: int) -> Image:
    with open(CTU_path, 'r') as file:
        for lines in file:
            parse_line = lines
            matchObj = re.sub('[<>]', "", parse_line)
            matchObj = re.sub('[ ]', ",", matchObj)
            chunk = matchObj.split(',')
            frame_file = int(chunk[0])
            pos = int(chunk[1])
            # print (lines)
            if frame_enc == frame_file:
                if pos == 0:
                    imgY = read_YUV420_frame(open(yuv_files, "rb"), bord_w, bord_h, 0).get('Y')
                    image = Image(imgY.astype(int) - 128, 64, 64, frame_enc)
                    # import_subdivide(image.get_ctu(pos), chunk[2:], 0, pos)
                if pos == image.nbCTU - 1:
                    # image.convert_tree_node_to_array()
                    # image.remove_bord_elements(lcu.w,lcu.h)
                    image.init_aki()
                    break
            else:
                raise ValueError(f'Error: Frame in file and encoder HEVC mismatch {frame_enc} {frame_file}')
    return image


if __name__ == '__main__':
    if len(sys.argv) == 12:
        bord_W = int(sys.argv[1])
        bord_H = int(sys.argv[2])
        Rt = float(sys.argv[3])
        Qp = int(sys.argv[4])
        rN = float(sys.argv[5])
        frame = int(sys.argv[6])
        frameType = int(sys.argv[7])
        Q1FileName = sys.argv[8]
        Q2FileName = sys.argv[9]
        resi = sys.argv[10]
        CTU_path = sys.argv[11]
        Q1FileName_DecAssist = f"{pathlib.Path(Q1FileName).stem}_dec.csv"
        Q2FileName_DecAssist = f"{pathlib.Path(Q2FileName).stem}_dec.csv"
        print(f"Process Frame {frame} Rt {Rt} rN {rN} frametype {frameType} QP {Qp}")
    else:
        print(f"Optimization Failed missing arg {len(sys.argv)}")
        sys.exit()
    tf.initROM()
    process_time_begin = time.time()
    image = buildModel(frame, CTU_path, resi, bord_H, bord_W)
    opt_param = OptimizerParameter(lam={1: 20.5, 2: 20.5}, rN=rN, Rt=Rt, image=image,_QPMax=Qp+25,_QPMin=Qp-10)
    optimizer = Optimizer_curvefitting(opt_param=opt_param, curve_fitting_function="exp2")
    optimizer.computeCurveCoefficient()
    result: OptimizerOutput = optimizer.optimizeQuadtreeLambaCst()
    posCTUCount = 0
    QP1 = result.get("QP1")
    QP2 = result.get("QP2")
    mse1 = result.get('D1')
    mse2 = result.get('D2')
    psnr1 = 10 * math.log10(255 ** 2 / mse1)
    psnr2 = 10 * math.log10(255 ** 2 / mse2)
    bpp1 = result.get('r1')
    bpp2 = result.get('r2')
    print(f"Estimated D1:{psnr1:2.4f}({mse1:2.4f}) D2:{psnr2:2.4f}({mse2:2.4f}) r1:{bpp1:2.4f}"
          f"({bpp1 * bord_W * bord_H:4.0f} bits) r2:{bpp2:2.4f}({bpp2 * bord_H * bord_W:4.0f}) bits")
    with open(Q1FileName, 'w') as f:
        for i in QP1:
            f.write(",".join([str(i)] * 200) + ",\n")
    # open the file in the write mode
    with open(Q2FileName, 'w') as f:
        for i in QP2:
            f.write(",".join([str(i)] * 200) + ",\n")
    # open the file in the write mode
    writetype = 'w'
    if frame > 0:
        writetype = 'a'
    with open(Q1FileName_DecAssist, writetype) as f:
        for i in QP1:
            f.write(",".join([str(i)] * 200) + ",\n")
    # open the file in the write mode
    with open(Q2FileName_DecAssist, writetype) as f:
        for i in QP2:
            f.write(",".join([str(i)] * 200) + ",\n")
