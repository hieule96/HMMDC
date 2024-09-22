#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComRom.h"
#include "TLibCommon/TComSlice.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
namespace py=pybind11;
Void xTr(Int bitDepth, int *block, int *coeff, UInt uiStride, UInt uiTrSize, Bool useDST, const Int maxLog2TrDynamicRange);
Void xITr(Int bitDepth, int *coeff, int *block, UInt uiStride, UInt uiTrSize, Bool useDST, const Int maxLog2TrDynamicRange);
class Quant{
    private:
        Int      *m_quantCoef            [SCALING_LIST_SIZE_NUM][SCALING_LIST_NUM][SCALING_LIST_REM_NUM]; ///< array of quantization matrix coefficient 4x4
        Int      *m_dequantCoef          [SCALING_LIST_SIZE_NUM][SCALING_LIST_NUM][SCALING_LIST_REM_NUM]; ///< array of dequantization matrix coefficient 4x4
        Double   *m_errScale             [SCALING_LIST_SIZE_NUM][SCALING_LIST_NUM][SCALING_LIST_REM_NUM]; ///< array of quantization matrix coefficient 4x4
        Double    m_errScaleNoScalingList[SCALING_LIST_SIZE_NUM][SCALING_LIST_NUM][SCALING_LIST_REM_NUM]; ///< array of quantization matrix coefficient 4x4
        Int       absSum;
    public:
        Quant(){
            initScalingList();
            absSum = 0;
            TComScalingList scaledefault = TComScalingList();
            scaledefault.setDefaultScalingList ();
            Int maxLog2TrDynamicRange[2] = {15,15};
            setScalingList(&scaledefault,maxLog2TrDynamicRange);

        }
        ~Quant(){
            destroyScalingList();
        };
        Void destroyScalingList();
        Void initScalingList();
        Int* getQuantCoeff                    ( UInt list, Int qp, UInt size ) { return m_quantCoef            [size][list][qp]; };  //!< get Quant Coefficent
        Int* getDequantCoeff                  ( UInt list, Int qp, UInt size ) { return m_dequantCoef          [size][list][qp]; };  //!< get DeQuant Coefficent
        Double* getErrScaleCoeff              ( UInt list, UInt size, Int qp ) { return m_errScale             [size][list][qp]; };  //!< get Error Scale Coefficent
        Double& getErrScaleCoeffNoScalingList ( UInt list, UInt size, Int qp ) { return m_errScaleNoScalingList[size][list][qp]; };  //!< get Error Scale Coefficent
        Int getAbsSum(){return this->absSum;};
        Void Quantize(Int* pSrc,Int* pDes,Int &uiAbsSum,
        const ComponentID compID,const Int baseQp, Int maxLog2TrDynamicRange, Int uiLog2TrSize, Int uiWidth, Int uiHeight);
        Void deQuantize(const Int* pSrc,Int* pDes,const ComponentID compID,const Int baseQp,const Int maxLog2TrDynamicRange, Int uiLog2TrSize, Int uiWidth, Int uiHeight);
        py::array_t <int> quantize2D(py::array_t <int> block,int QP);
        py::array_t<int> dequantize2D(py::array_t <int> block,int QP);
        Void processScalingListEnc( Int *coeff, Int *quantcoeff, Int quantScales, UInt height, UInt width, UInt ratio, Int sizuNum, UInt dc);
        Void xSetScalingListEnc(TComScalingList *scalingList, UInt listId, UInt sizeId, Int qp);
        Void xSetScalingListDec(const TComScalingList &scalingList, UInt listId, UInt sizeId, Int qp);
        Void processScalingListDec( const Int *coeff, Int *dequantcoeff, Int invQuantScales, UInt height, UInt width, UInt ratio, Int sizuNum, UInt dc);
        Void setScalingListDec(const TComScalingList &scalingList);
        Void setErrScaleCoeff(UInt list, UInt size, Int qp, const Int maxLog2TrDynamicRange[MAX_NUM_CHANNEL_TYPE], const BitDepths &bitDepths);
        Void setScalingList(TComScalingList *scalingList, const Int maxLog2TrDynamicRange[MAX_NUM_CHANNEL_TYPE]);
};
