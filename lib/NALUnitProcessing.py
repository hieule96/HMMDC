#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:18:45 2022

@author: ubuntu
"""
import sys
import os
import re
from bitstring import BitArray, BitStream

class NalUnitType:
    """
    Table 7-1 - NAL unit type codes and NAL unit type classes
    copypaste from source/Lib/TLibCommon/CommonDef.h
    """
    NAL_UNIT_CODED_SLICE_TRAIL_N = 0
    NAL_UNIT_CODED_SLICE_TRAIL_R = 1

    NAL_UNIT_CODED_SLICE_TSA_N = 2
    NAL_UNIT_CODED_SLICE_TSA_R = 3

    NAL_UNIT_CODED_SLICE_STSA_N = 4
    NAL_UNIT_CODED_SLICE_STSA_R = 5

    NAL_UNIT_CODED_SLICE_RADL_N = 6
    NAL_UNIT_CODED_SLICE_RADL_R = 7

    NAL_UNIT_CODED_SLICE_RASL_N = 8
    NAL_UNIT_CODED_SLICE_RASL_R = 9

    NAL_UNIT_RESERVED_VCL_N10 = 10
    NAL_UNIT_RESERVED_VCL_R11 = 11
    NAL_UNIT_RESERVED_VCL_N12 = 12
    NAL_UNIT_RESERVED_VCL_R13 = 13
    NAL_UNIT_RESERVED_VCL_N14 = 14
    NAL_UNIT_RESERVED_VCL_R15 = 15

    NAL_UNIT_CODED_SLICE_BLA_W_LP = 16
    NAL_UNIT_CODED_SLICE_BLA_W_RADL = 17
    NAL_UNIT_CODED_SLICE_BLA_N_LP = 18
    NAL_UNIT_CODED_SLICE_IDR_W_RADL = 19
    NAL_UNIT_CODED_SLICE_IDR_N_LP = 20
    NAL_UNIT_CODED_SLICE_CRA = 21
    NAL_UNIT_RESERVED_IRAP_VCL22 = 22
    NAL_UNIT_RESERVED_IRAP_VCL23 = 23

    NAL_UNIT_RESERVED_VCL24 = 24
    NAL_UNIT_RESERVED_VCL25 = 25
    NAL_UNIT_RESERVED_VCL26 = 26
    NAL_UNIT_RESERVED_VCL27 = 27
    NAL_UNIT_RESERVED_VCL28 = 28
    NAL_UNIT_RESERVED_VCL29 = 29
    NAL_UNIT_RESERVED_VCL30 = 30
    NAL_UNIT_RESERVED_VCL31 = 31

    NAL_UNIT_VPS = 32
    NAL_UNIT_SPS = 33
    NAL_UNIT_PPS = 34
    NAL_UNIT_ACCESS_UNIT_DELIMITER = 35
    NAL_UNIT_EOS = 36
    NAL_UNIT_EOB = 37
    NAL_UNIT_FILLER_DATA = 38
    NAL_UNIT_PREFIX_SEI = 39
    NAL_UNIT_SUFFIX_SEI = 40

    NAL_UNIT_RESERVED_NVCL41 = 41
    NAL_UNIT_RESERVED_NVCL42 = 42
    NAL_UNIT_RESERVED_NVCL43 = 43
    NAL_UNIT_RESERVED_NVCL44 = 44
    NAL_UNIT_RESERVED_NVCL45 = 45
    NAL_UNIT_RESERVED_NVCL46 = 46
    NAL_UNIT_RESERVED_NVCL47 = 47
    NAL_UNIT_UNSPECIFIED_48 = 48
    NAL_UNIT_UNSPECIFIED_49 = 49
    NAL_UNIT_UNSPECIFIED_50 = 50
    NAL_UNIT_UNSPECIFIED_51 = 51
    NAL_UNIT_UNSPECIFIED_52 = 52
    NAL_UNIT_UNSPECIFIED_53 = 53
    NAL_UNIT_UNSPECIFIED_54 = 54
    NAL_UNIT_UNSPECIFIED_55 = 55
    NAL_UNIT_UNSPECIFIED_56 = 56
    NAL_UNIT_UNSPECIFIED_57 = 57
    NAL_UNIT_UNSPECIFIED_58 = 58
    NAL_UNIT_UNSPECIFIED_59 = 59
    NAL_UNIT_UNSPECIFIED_60 = 60
    NAL_UNIT_UNSPECIFIED_61 = 61
    NAL_UNIT_UNSPECIFIED_62 = 62
    NAL_UNIT_UNSPECIFIED_63 = 63
    NAL_UNIT_INVALID = 64

class nal_unit_header(object):
    def __init__(self, s):
        """
        Interpret next bits in BitString s as a nal_unit
        """
        self.forbidden_zero_bit  = s.read('uint:1')
        self.nal_unit_type = s.read('uint:6')
        self.nuh_layer_id = s.read('uint:6')
        self.nuh_temporal_id_plus1 = s.read('uint:3')

    def show(self):
        print ('forbidden_zero_bit', self.forbidden_zero_bit)
        print ('nal_unit_type', self.nal_unit_type)
        print ('nuh_layer_id', self.nuh_layer_id)
        print ('nuh_temporal_id_plus1', self.nuh_temporal_id_plus1)
    
def read_nal_unit(s, i, NumBytesInNalUnit):
    """
    Table 7-1 - NAL unit type codes and NAL unit type classes
    """
    checknotHeader = 0
    s.pos = i + 24
    n = nal_unit_header(s)
    nal_unit_type = n.nal_unit_type
    if (nal_unit_type == NalUnitType.NAL_UNIT_VPS):
        checknotHeader = 0
    elif (nal_unit_type == NalUnitType.NAL_UNIT_SPS):
        checknotHeader = 0
    elif (nal_unit_type == NalUnitType.NAL_UNIT_PPS):
        checknotHeader = 0
    else:
        checknotHeader = 1
    return checknotHeader
    
def write_nal_units(s,i,NumBytesInNalUnit,File):
    File.write(s.read('bytes:%d' %(NumBytesInNalUnit)))
