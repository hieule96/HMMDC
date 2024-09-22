#include "Transform.h"
/** NxN forward transform (2D) using brute force matrix multiplication (3 nested loops)
 *  \param block pointer to input data (residual)
 *  \param coeff pointer to output data (transform coefficients)
 *  \param uiStride stride of input data
 *  \param uiTrSize transform size (uiTrSize x uiTrSize)
 *  \param uiMode is Intra Prediction mode used in Mode-Dependent DCT/DST only
 */
Void xTr(Int bitDepth, int *block, int *coeff, UInt uiStride, UInt uiTrSize, Bool useDST, const Int maxLog2TrDynamicRange)
{
  UInt i,j,k;
  int iSum;
  int tmp[MAX_TU_SIZE * MAX_TU_SIZE];
  const short *iT;
  UInt uiLog2TrSize = g_aucConvertToBit[ uiTrSize ] + 2;

  if (uiTrSize==4)
  {
    iT  = (useDST ? g_as_DST_MAT_4[TRANSFORM_FORWARD][0] : g_aiT4[TRANSFORM_FORWARD][0]);
  }
  else if (uiTrSize==8)
  {
    iT = g_aiT8[TRANSFORM_FORWARD][0];
  }
  else if (uiTrSize==16)
  {
    iT = g_aiT16[TRANSFORM_FORWARD][0];
  }
  else if (uiTrSize==32)
  {
    iT = g_aiT32[TRANSFORM_FORWARD][0];
  }
  else
  {
    assert(0);
  }

  const Int TRANSFORM_MATRIX_SHIFT = g_transformMatrixShift[TRANSFORM_FORWARD];

  const Int shift_1st = (uiLog2TrSize +  bitDepth + TRANSFORM_MATRIX_SHIFT) - maxLog2TrDynamicRange;
  const Int shift_2nd = uiLog2TrSize + TRANSFORM_MATRIX_SHIFT;
  const Int add_1st = (shift_1st>0) ? (1<<(shift_1st-1)) : 0;
  const Int add_2nd = 1<<(shift_2nd-1);

  /* Horizontal transform */

  for (i=0; i<uiTrSize; i++)
  {
    for (j=0; j<uiTrSize; j++)
    {
      iSum = 0;
      for (k=0; k<uiTrSize; k++)
      {
        iSum += iT[i*uiTrSize+k]*block[j*uiStride+k];
      }
      tmp[i*uiTrSize+j] = (iSum + add_1st)>>shift_1st;
    }
  }

  /* Vertical transform */
  for (i=0; i<uiTrSize; i++)
  {
    for (j=0; j<uiTrSize; j++)
    {
      iSum = 0;
      for (k=0; k<uiTrSize; k++)
      {
        iSum += iT[i*uiTrSize+k]*tmp[j*uiTrSize+k];
      }
      coeff[i*uiTrSize+j] = (iSum + add_2nd)>>shift_2nd;
    }
  }
}

/** NxN inverse transform (2D) using brute force matrix multiplication (3 nested loops)
 *  \param coeff pointer to input data (transform coefficients)
 *  \param block pointer to output data (residual)
 *  \param uiStride stride of output data
 *  \param uiTrSize transform size (uiTrSize x uiTrSize)
 *  \param uiMode is Intra Prediction mode used in Mode-Dependent DCT/DST only
 */
Void xITr(Int bitDepth, int *coeff, int *block, UInt uiStride, UInt uiTrSize, Bool useDST, const Int maxLog2TrDynamicRange)
{
  UInt i,j,k;
  TCoeff iSum;
  TCoeff tmp[MAX_TU_SIZE * MAX_TU_SIZE];
  const TMatrixCoeff *iT;

  if (uiTrSize==4)
  {
    iT  = (useDST ? g_as_DST_MAT_4[TRANSFORM_INVERSE][0] : g_aiT4[TRANSFORM_INVERSE][0]);
  }
  else if (uiTrSize==8)
  {
    iT = g_aiT8[TRANSFORM_INVERSE][0];
  }
  else if (uiTrSize==16)
  {
    iT = g_aiT16[TRANSFORM_INVERSE][0];
  }
  else if (uiTrSize==32)
  {
    iT = g_aiT32[TRANSFORM_INVERSE][0];
  }
  else
  {
    assert(0);
  }

  const Int TRANSFORM_MATRIX_SHIFT = g_transformMatrixShift[TRANSFORM_INVERSE];

  const Int shift_1st = TRANSFORM_MATRIX_SHIFT + 1; //1 has been added to shift_1st at the expense of shift_2nd
  const Int shift_2nd = (TRANSFORM_MATRIX_SHIFT + maxLog2TrDynamicRange - 1) - bitDepth;
  const TCoeff clipMinimum = -(1 << maxLog2TrDynamicRange);
  const TCoeff clipMaximum =  (1 << maxLog2TrDynamicRange) - 1;
  assert(shift_2nd>=0);
  const Int add_1st = 1<<(shift_1st-1);
  const Int add_2nd = (shift_2nd>0) ? (1<<(shift_2nd-1)) : 0;

  /* Horizontal transform */
  for (i=0; i<uiTrSize; i++)
  {
    for (j=0; j<uiTrSize; j++)
    {
      iSum = 0;
      for (k=0; k<uiTrSize; k++)
      {
        iSum += iT[k*uiTrSize+i]*coeff[k*uiTrSize+j];
      }

      // Clipping here is not in the standard, but is used to protect the "Pel" data type into which the inverse-transformed samples will be copied
      tmp[i*uiTrSize+j] = Clip3<TCoeff>(clipMinimum, clipMaximum, (iSum + add_1st)>>shift_1st);
    }
  }

  /* Vertical transform */
  for (i=0; i<uiTrSize; i++)
  {
    for (j=0; j<uiTrSize; j++)
    {
      iSum = 0;
      for (k=0; k<uiTrSize; k++)
      {
        iSum += iT[k*uiTrSize+j]*tmp[i*uiTrSize+k];
      }

      block[i*uiStride+j] = Clip3<TCoeff>(std::numeric_limits<Pel>::min(), std::numeric_limits<Pel>::max(), (iSum + add_2nd)>>shift_2nd);
    }
  }
}

Void Quant::initScalingList()
{
  for(UInt sizeId = 0; sizeId < SCALING_LIST_SIZE_NUM; sizeId++)
  {
    for(UInt qp = 0; qp < SCALING_LIST_REM_NUM; qp++)
    {
      for(UInt listId = 0; listId < SCALING_LIST_NUM; listId++)
      {
        m_quantCoef   [sizeId][listId][qp] = new Int    [g_scalingListSize[sizeId]];
        m_dequantCoef [sizeId][listId][qp] = new Int    [g_scalingListSize[sizeId]];
        m_errScale    [sizeId][listId][qp] = new Double [g_scalingListSize[sizeId]];
      } // listID loop
    }
  }
}

/** destroy quantization matrix array
 */
Void Quant::destroyScalingList()
{
  for(UInt sizeId = 0; sizeId < SCALING_LIST_SIZE_NUM; sizeId++)
  {
    for(UInt listId = 0; listId < SCALING_LIST_NUM; listId++)
    {
      for(UInt qp = 0; qp < SCALING_LIST_REM_NUM; qp++)
      {
        if(m_quantCoef[sizeId][listId][qp])
        {
          delete [] m_quantCoef[sizeId][listId][qp];
        }
        if(m_dequantCoef[sizeId][listId][qp])
        {
          delete [] m_dequantCoef[sizeId][listId][qp];
        }
        if(m_errScale[sizeId][listId][qp])
        {
          delete [] m_errScale[sizeId][listId][qp];
        }
      }
    }
  }
}

/** set quantized matrix coefficient for encode
 * \param scalingList            quantized matrix address
 * \param format                 chroma format
 * \param maxLog2TrDynamicRange
 * \param bitDepths              reference to bit depth array for all channels
 */
Void Quant::setScalingList(TComScalingList *scalingList, const Int maxLog2TrDynamicRange[MAX_NUM_CHANNEL_TYPE])
{
  const Int minimumQp = 0;
  const Int maximumQp = SCALING_LIST_REM_NUM;

  for(UInt size = 0; size < SCALING_LIST_SIZE_NUM; size++)
  {
    for(UInt list = 0; list < SCALING_LIST_NUM; list++)
    {
      for(Int qp = minimumQp; qp < maximumQp; qp++)
      {
        xSetScalingListEnc(scalingList,list,size,qp);
        xSetScalingListDec(*scalingList,list,size,qp);
        // setErrScaleCoeff(list,size,qp,maxLog2TrDynamicRange, bitDepths);
      }
    }
  }
}
/** set quantized matrix coefficient for decode
 * \param scalingList quantized matrix address
 * \param format      chroma format
 */
Void Quant::setScalingListDec(const TComScalingList &scalingList)
{
  const Int minimumQp = 0;
  const Int maximumQp = SCALING_LIST_REM_NUM;

  for(UInt size = 0; size < SCALING_LIST_SIZE_NUM; size++)
  {
    for(UInt list = 0; list < SCALING_LIST_NUM; list++)
    {
      for(Int qp = minimumQp; qp < maximumQp; qp++)
      {
        xSetScalingListDec(scalingList,list,size,qp);
      }
    }
  }
}

Void Quant::setErrScaleCoeff(UInt list, UInt size, Int qp, const Int maxLog2TrDynamicRange[MAX_NUM_CHANNEL_TYPE], const BitDepths &bitDepths)
{
  const UInt uiLog2TrSize = g_aucConvertToBit[ g_scalingListSizeX[size] ] + 2;
  const ChannelType channelType = ((list == 0) || (list == MAX_NUM_COMPONENT)) ? CHANNEL_TYPE_LUMA : CHANNEL_TYPE_CHROMA;

  const Int channelBitDepth    = bitDepths.recon[channelType];
  const Int iTransformShift = getTransformShift(channelBitDepth, uiLog2TrSize, maxLog2TrDynamicRange[channelType]);  // Represents scaling through forward transform

  UInt i,uiMaxNumCoeff = g_scalingListSize[size];
  Int *piQuantcoeff;
  Double *pdErrScale;
  piQuantcoeff   = getQuantCoeff(list, qp,size);
  pdErrScale     = getErrScaleCoeff(list, size, qp);

  Double dErrScale = (Double)(1<<SCALE_BITS);                                // Compensate for scaling of bitcount in Lagrange cost function
  dErrScale = dErrScale*pow(2.0,(-2.0*iTransformShift));                     // Compensate for scaling through forward transform

  for(i=0;i<uiMaxNumCoeff;i++)
  {
    pdErrScale[i] =  dErrScale / piQuantcoeff[i] / piQuantcoeff[i] / (1 << DISTORTION_PRECISION_ADJUSTMENT(2 * (bitDepths.recon[channelType] - 8)));
  }

  getErrScaleCoeffNoScalingList(list, size, qp) = dErrScale / g_quantScales[qp] / g_quantScales[qp] / (1 << DISTORTION_PRECISION_ADJUSTMENT(2 * (bitDepths.recon[channelType] - 8)));
}

/** set quantized matrix coefficient for decode
 * \param scalingList quantaized matrix address
 * \param listId List index
 * \param sizeId size index
 * \param qp Quantization parameter
 * \param format chroma format
 */
Void Quant::xSetScalingListDec(const TComScalingList &scalingList, UInt listId, UInt sizeId, Int qp)
{
  UInt width  = g_scalingListSizeX[sizeId];
  UInt height = g_scalingListSizeX[sizeId];
  UInt ratio  = g_scalingListSizeX[sizeId]/min(MAX_MATRIX_SIZE_NUM,(Int)g_scalingListSizeX[sizeId]);
  Int *dequantcoeff;
  const Int *coeff  = scalingList.getScalingListAddress(sizeId,listId);

  dequantcoeff = getDequantCoeff(listId, qp, sizeId);

  Int invQuantScale = g_invQuantScales[qp];

  processScalingListDec(coeff,
                        dequantcoeff,
                        invQuantScale,
                        height, width, ratio,
                        min(MAX_MATRIX_SIZE_NUM, (Int)g_scalingListSizeX[sizeId]),
                        scalingList.getScalingListDC(sizeId,listId));
}

/** set quantized matrix coefficient for encode
 * \param scalingList quantized matrix address
 * \param listId List index
 * \param sizeId size index
 * \param qp Quantization parameter
 * \param format chroma format
 */
Void Quant::xSetScalingListEnc(TComScalingList *scalingList, UInt listId, UInt sizeId, Int qp)
{
  UInt width  = g_scalingListSizeX[sizeId];
  UInt height = g_scalingListSizeX[sizeId];
  UInt ratio  = g_scalingListSizeX[sizeId]/min(MAX_MATRIX_SIZE_NUM,(Int)g_scalingListSizeX[sizeId]);
  Int *quantcoeff;
  Int *coeff  = scalingList->getScalingListAddress(sizeId,listId);
  quantcoeff  = getQuantCoeff(listId, qp, sizeId);

  Int quantScales = g_quantScales[qp];

  processScalingListEnc(coeff,
                        quantcoeff,
                        (quantScales << LOG2_SCALING_LIST_NEUTRAL_VALUE),
                        height, width, ratio,
                        min(MAX_MATRIX_SIZE_NUM, (Int)g_scalingListSizeX[sizeId]),
                        scalingList->getScalingListDC(sizeId,listId));
}
/** set quantized matrix coefficient for encode
 * \param coeff quantaized matrix address
 * \param quantcoeff quantaized matrix address
 * \param quantScales Q(QP%6)
 * \param height height
 * \param width width
 * \param ratio ratio for upscale
 * \param sizuNum matrix size
 * \param dc dc parameter
 */
Void Quant::processScalingListEnc( Int *coeff, Int *quantcoeff, Int quantScales, UInt height, UInt width, UInt ratio, Int sizuNum, UInt dc)
{
  for(UInt j=0;j<height;j++)
  {
    for(UInt i=0;i<width;i++)
    {
      quantcoeff[j*width + i] = quantScales / coeff[sizuNum * (j / ratio) + i / ratio];
    }
  }

  if(ratio > 1)
  {
    quantcoeff[0] = quantScales / dc;
  }
}

/** set quantized matrix coefficient for decode
 * \param coeff quantaized matrix address
 * \param dequantcoeff quantaized matrix address
 * \param invQuantScales IQ(QP%6))
 * \param height height
 * \param width width
 * \param ratio ratio for upscale
 * \param sizuNum matrix size
 * \param dc dc parameter
 */
Void Quant::processScalingListDec( const Int *coeff, Int *dequantcoeff, Int invQuantScales, UInt height, UInt width, UInt ratio, Int sizuNum, UInt dc)
{
  for(UInt j=0;j<height;j++)
  {
    for(UInt i=0;i<width;i++)
    {
      dequantcoeff[j*width + i] = invQuantScales * coeff[sizuNum * (j / ratio) + i / ratio];
    }
  }

  if(ratio > 1)
  {
    dequantcoeff[0] = invQuantScales * dc;
  }
}


py::array_t <int> Quant::quantize2D(py::array_t <int> block,int QP){
    py::buffer_info b_input = block.request();
    if (b_input.format!=py::format_descriptor<int>::format())
    {
        throw std::runtime_error("Incompatible format: expected a int array!");
    }
    int w = b_input.shape[1];
    int h = b_input.shape[0];
    py::array_t<int> result = py::array_t<int>(b_input.size);
    py::buffer_info b_result = result.request();
    this->absSum = 0;
    this->Quantize((int *)b_input.ptr,(int*)b_result.ptr,this->absSum,COMPONENT_Y,QP,15,log2(w),w,h);
    result.resize({w,h});
    return result;
}
py::array_t<int> Quant::dequantize2D(py::array_t <int> block,int QP){
    py::buffer_info b_input = block.request();
    if (b_input.format!=py::format_descriptor<int>::format())
    {
        throw std::runtime_error("Incompatible format: expected a int array!");
    }
    int w = b_input.shape[1];
    int h = b_input.shape[0];
    py::array_t<int> result = py::array_t<int>(b_input.size);
    py::buffer_info b_result = result.request();
    int absSum = 0;
    this->deQuantize((int *)b_input.ptr,(int*)b_result.ptr,COMPONENT_Y,QP,15,log2(w),w,h);
    result.resize({w,h});
    return result;
}

Void Quant::Quantize(Int* pSrc,Int* pDes,Int &uiAbsSum,
const ComponentID compID,const Int baseQp, Int maxLog2TrDynamicRange, Int uiLog2TrSize, Int uiWidth, Int uiHeight)
{
    Int Qp =baseQp;
    Int per=baseQp/6;
    Int rem=baseQp%6;
    TCoeff* piCoef    = pSrc;
    TCoeff* piQCoef   = pDes;
    Int channelBitDepth = 8;
    const TCoeff entropyCodingMinimum = -(1 << maxLog2TrDynamicRange);
    const TCoeff entropyCodingMaximum =  (1 << maxLog2TrDynamicRange) - 1;

    TCoeff deltaU[MAX_TU_SIZE * MAX_TU_SIZE];

    Int *piQuantCoeff = getQuantCoeff(0, rem, uiLog2TrSize-2);

    const Int  defaultQuantisationCoefficient = g_quantScales[rem];

    /* for 422 chroma blocks, the effective scaling applied during transformation is not a power of 2, hence it cannot be
     * implemented as a bit-shift (the quantised result will be sqrt(2) * larger than required). Alternatively, adjust the
     * uiLog2TrSize applied in iTransformShift, such that the result is 1/sqrt(2) the required result (i.e. smaller)
     * Then a QP+3 (sqrt(2)) or QP-3 (1/sqrt(2)) method could be used to get the required result
     */

    // Represents scaling through forward transform

    Int iTransformShift = maxLog2TrDynamicRange - channelBitDepth - uiLog2TrSize;
    // iTransformShift = std::max<Int>(0, iTransformShift);

    const Int iQBits = QUANT_SHIFT + per + iTransformShift;
    // QBits will be OK for any internal bit depth as the reduction in transform shift is balanced by an increase in Qp_per due to QpBDOffset
    const Int iAdd   = 171 << (iQBits-9);
    const Int qBits8 = iQBits - 8;

    for( Int uiBlockPos = 0; uiBlockPos < uiWidth*uiHeight; uiBlockPos++ )
    {
      const TCoeff iLevel   = piCoef[uiBlockPos];
      const TCoeff iSign    = (iLevel < 0 ? -1: 1);

      const Int64  tmpLevel = (Int64)abs(iLevel) * piQuantCoeff[uiBlockPos];
      const TCoeff quantisedMagnitude = TCoeff((tmpLevel + iAdd ) >> iQBits);
      deltaU[uiBlockPos] = (TCoeff)((tmpLevel - (quantisedMagnitude<<iQBits) )>> qBits8);

      uiAbsSum += quantisedMagnitude;
      const TCoeff quantisedCoefficient = quantisedMagnitude * iSign;

      piQCoef[uiBlockPos] = Clip3<TCoeff>( entropyCodingMinimum, entropyCodingMaximum, quantisedCoefficient );
    } // for n
}

Void Quant::deQuantize(const Int* pSrc,Int* pDes,const ComponentID compID,const Int baseQp,const Int maxLog2TrDynamicRange, Int uiLog2TrSize, Int uiWidth, Int uiHeight)
{

  const TCoeff        *const piQCoef            = pSrc;
        TCoeff        *const piCoef             = pDes;
  const TCoeff               transformMinimum   = -(1 << maxLog2TrDynamicRange);
  const TCoeff               transformMaximum   =  (1 << maxLog2TrDynamicRange) - 1;
  const Int                  scalingListType    = 0+compID;
  Int channelBitDepth = 8;
  const Int numSamplesInBlock = uiWidth*uiHeight;
  assert (scalingListType < SCALING_LIST_NUM);

  // Represents scaling through forward transform
  const Bool bClipTransformShiftTo0 = false;
  const Int  originalTransformShift = maxLog2TrDynamicRange - channelBitDepth - uiLog2TrSize;
  const Int  iTransformShift        = bClipTransformShiftTo0 ? std::max<Int>(0, originalTransformShift) : originalTransformShift;

  Int Qp =baseQp;
  Int per=baseQp/6;
  Int rem=baseQp%6;
  bool enableScalingLists = true;
  const Int rightShift = (IQUANT_SHIFT - (iTransformShift + per)) + (enableScalingLists ? LOG2_SCALING_LIST_NEUTRAL_VALUE : 0);

  if(enableScalingLists)
  {
    //from the dequantisation equation:
    //iCoeffQ                         = ((Intermediate_Int(clipQCoef) * piDequantCoef[deQuantIdx]) + iAdd ) >> rightShift
    //(sizeof(Intermediate_Int) * 8)  =              inputBitDepth    +    dequantCoefBits                   - rightShift
    const UInt             dequantCoefBits     = 1 + IQUANT_SHIFT + SCALING_LIST_BITS;
    const UInt             targetInputBitDepth = std::min<UInt>((maxLog2TrDynamicRange + 1), (((sizeof(Intermediate_Int) * 8) + rightShift) - dequantCoefBits));

    const Intermediate_Int inputMinimum        = -(1 << (targetInputBitDepth - 1));
    const Intermediate_Int inputMaximum        =  (1 << (targetInputBitDepth - 1)) - 1;

    Int *piDequantCoef = getDequantCoeff(scalingListType,rem,uiLog2TrSize-2);

    if(rightShift > 0)
    {
      const Intermediate_Int iAdd = 1 << (rightShift - 1);

      for( Int n = 0; n < numSamplesInBlock; n++ )
      {
        const TCoeff           clipQCoef = TCoeff(Clip3<Intermediate_Int>(inputMinimum, inputMaximum, piQCoef[n]));
        const Intermediate_Int iCoeffQ   = ((Intermediate_Int(clipQCoef) * piDequantCoef[n]) + iAdd ) >> rightShift;

        piCoef[n] = TCoeff(Clip3<Intermediate_Int>(transformMinimum,transformMaximum,iCoeffQ));
      }
    }
    else
    {
      const Int leftShift = -rightShift;

      for( Int n = 0; n < numSamplesInBlock; n++ )
      {
        const TCoeff           clipQCoef = TCoeff(Clip3<Intermediate_Int>(inputMinimum, inputMaximum, piQCoef[n]));
        const Intermediate_Int iCoeffQ   = (Intermediate_Int(clipQCoef) * piDequantCoef[n]) << leftShift;

        piCoef[n] = TCoeff(Clip3<Intermediate_Int>(transformMinimum,transformMaximum,iCoeffQ));
      }
    }
  }
  else
  {
    const Int scale     =  g_invQuantScales[rem];
    const Int scaleBits =     (IQUANT_SHIFT + 1)   ;

    //from the dequantisation equation:
    //iCoeffQ                         = Intermediate_Int((Int64(clipQCoef) * scale + iAdd) >> rightShift);
    //(sizeof(Intermediate_Int) * 8)  =                    inputBitDepth   + scaleBits      - rightShift
    const UInt             targetInputBitDepth = std::min<UInt>((maxLog2TrDynamicRange + 1), (((sizeof(Intermediate_Int) * 8) + rightShift) - scaleBits));
    const Intermediate_Int inputMinimum        = -(1 << (targetInputBitDepth - 1));
    const Intermediate_Int inputMaximum        =  (1 << (targetInputBitDepth - 1)) - 1;

    if (rightShift > 0)
    {
      const Intermediate_Int iAdd = 1 << (rightShift - 1);

      for( Int n = 0; n < numSamplesInBlock; n++ )
      {
        const TCoeff           clipQCoef = TCoeff(Clip3<Intermediate_Int>(inputMinimum, inputMaximum, piQCoef[n]));
        const Intermediate_Int iCoeffQ   = (Intermediate_Int(clipQCoef) * scale + iAdd) >> rightShift;

        piCoef[n] = TCoeff(Clip3<Intermediate_Int>(transformMinimum,transformMaximum,iCoeffQ));
      }
    }
    else
    {
      const Int leftShift = -rightShift;

      for( Int n = 0; n < numSamplesInBlock; n++ )
      {
        const TCoeff           clipQCoef = TCoeff(Clip3<Intermediate_Int>(inputMinimum, inputMaximum, piQCoef[n]));
        const Intermediate_Int iCoeffQ   = (Intermediate_Int(clipQCoef) * scale) << leftShift;

        piCoef[n] = TCoeff(Clip3<Intermediate_Int>(transformMinimum,transformMaximum,iCoeffQ));
      }
    }
  }
}
