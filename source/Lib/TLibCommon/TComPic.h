/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2020, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TComPic.h
    \brief    picture class (header)
*/

#ifndef __TCOMPIC__
#define __TCOMPIC__

// Include files
#include "CommonDef.h"
#include "TComPicSym.h"
#include "TComPicYuv.h"
#include "TComBitStream.h"

//! \ingroup TLibCommon
//! \{

// ====================================================================================================================
// Class definition
// ====================================================================================================================

/// picture class (symbol + YUV buffers)

class TComPic
{
public:
  //@tle: add two other Picture D1,D2,D0
  typedef enum { PIC_YUV_ORG=0, 
  PIC_YUV_REC=1,
  PIC_YUV_REC1=2,
  PIC_YUV_REC2=3, 
  PIC_YUV_TRUE_ORG=4,
  NUM_PIC_YUV=5 } PIC_YUV_T;
     // TRUE_ORG is the input file without any pre-encoder colour space conversion (but with possible bit depth increment)
  TComPicYuv*   getPicYuvTrueOrg()        { return  m_apcPicYuv[PIC_YUV_TRUE_ORG]; }

private:
  UInt                  m_uiTLayer;               //  Temporal layer
  Bool                  m_bUsedByCurr;            //  Used by current picture
  Bool                  m_bIsLongTerm;            //  IS long term picture
  TComPicSym*            m_pcpicSym;                 //  Symbol
  std::array<TComPicSym,2> m_arrPicSym;
  TComPicYuv*           m_apcPicYuv[NUM_PIC_YUV];

  TComPicYuv*           m_pcPicYuvPred;           //  Prediction
  TComPicYuv*           m_pcPicYuvResi;           //  Residual
  Bool                  m_bReconstructed;
  Bool                  m_bNeededForOutput;
  UInt                  m_uiCurrSliceIdx;         // Index of current slice
  Bool                  m_bCheckLTMSB;
  Bool                  m_isTop;
  Bool                  m_isField;
  // tle: add information from the file
  Int                   m_iRunMode;
  Double*                   m_lamdaForcing;
  std::vector<std::vector<TComDataCU*> > m_vSliceCUDataLink;
  std::vector<UInt*>                 m_QPArray1;
  std::vector<UInt*>                 m_QPArray2;
  SEIMessages  m_SEIs; ///< Any SEI messages that have been received.  If !NULL we own the object.
  Int         m_descriptionId;
public:
  TComPic();
  virtual ~TComPic();

#if REDUCED_ENCODER_MEMORY
  Void          create( const TComSPS &sps, const TComPPS &pps, const Bool bCreateEncoderSourcePicYuv, const Bool bCreateForImmediateReconstruction );
  Void          prepareForEncoderSourcePicYuv();
  Void          prepareForReconstruction();
  Void          releaseReconstructionIntermediateData();
  Void          releaseAllReconstructionData();
  Void          releaseEncoderSourceImageData();
#else
  Void          create( const TComSPS &sps, const TComPPS &pps, const Bool bIsVirtual /*= false*/ );
#endif

  virtual Void  destroy();

  UInt          getTLayer() const               { return m_uiTLayer;   }
  Void          setTLayer( UInt uiTLayer ) { m_uiTLayer = uiTLayer; }

  Bool          getUsedByCurr() const            { return m_bUsedByCurr; }
  Void          setUsedByCurr( Bool bUsed ) { m_bUsedByCurr = bUsed; }
  Bool          getIsLongTerm() const            { return m_bIsLongTerm; }
  Void          setIsLongTerm( Bool lt ) { m_bIsLongTerm = lt; }
  Void          setCheckLTMSBPresent     (Bool b ) {m_bCheckLTMSB=b;}
  Bool          getCheckLTMSBPresent     () { return m_bCheckLTMSB;}
  TComPicSym*   getPicSym()                        { return  m_pcpicSym;    }
  const TComPicSym* getPicSym() const              { return  m_pcpicSym;    }
  TComSlice*    getSlice(Int i)                    { return  m_pcpicSym->getSlice(i);  }
  const TComSlice* getSlice(Int i) const           { return  m_pcpicSym->getSlice(i);  }
  Int           getPOC() const                     { 
      if (m_pcpicSym->getSlice(m_uiCurrSliceIdx)){
        return  m_pcpicSym->getSlice(m_uiCurrSliceIdx)->getPOC();
      }
      else{
        return MAX_INT;
      }  
    }
  TComDataCU*   getCtu( UInt ctuRsAddr )           { return  m_pcpicSym->getCtu( ctuRsAddr ); }
  const TComDataCU* getCtu( UInt ctuRsAddr ) const { return  m_pcpicSym->getCtu( ctuRsAddr ); }

  /// <summary>
  /// tle
  /// </summary>
  /// <param name="iRunMode"></param>
  Void setRunMode(Int iRunMode) { m_iRunMode = iRunMode; }
  Int getRunMode() { return m_iRunMode; }
  Void setLamdaForcing(Double* lambda) { m_lamdaForcing = lambda; }
  Double getLamdaForcing() { return *m_lamdaForcing; }
  Int             getDescriptionId              () const                                                            {return m_descriptionId;}
  FileType        getFileType() const {return FileType(m_descriptionId);}
  Void           setDescriptionId              (Int iDescription)                                            {
    m_descriptionId = iDescription;
    switch (m_descriptionId)
    {
    case 1:
      m_pcpicSym = &m_arrPicSym[0];
      break;
    case 2:
      m_pcpicSym = &m_arrPicSym[1];
      break;
    default:
      m_pcpicSym = &m_arrPicSym[0];
      break;
    }
  }



  TComPicYuv*   getPicYuvOrg()        { return  m_apcPicYuv[PIC_YUV_ORG]; }
  TComPicYuv*   getPicYuvRec()        { 
    switch (m_descriptionId)
    {
    case 1:
      return m_apcPicYuv[PIC_YUV_REC1];
      break;
    case 2:
      return m_apcPicYuv[PIC_YUV_REC2];
      break;
    default:
      return  m_apcPicYuv[PIC_YUV_REC]; 
      break;
    }
  }
  TComPicYuv *getPicYUVRec1(){return m_apcPicYuv[PIC_YUV_REC1];};
  TComPicYuv *getPicYUVRec2(){return m_apcPicYuv[PIC_YUV_REC2];};
  TComPicYuv *getPicYUVRecC(){return m_apcPicYuv[PIC_YUV_REC];};
  TComPicSym*   getPicSym(Int Description) {assert (Description>0&&Description<=2) ;return &m_arrPicSym[Description-1]; };
  TComPicYuv*   getPicYuvPred()       { return  m_pcPicYuvPred; }
  TComPicYuv*   getPicYuvResi()       { return  m_pcPicYuvResi; }
  Void          setPicYuvPred( TComPicYuv* pcPicYuv )       { m_pcPicYuvPred = pcPicYuv; }
  Void          setPicYuvResi( TComPicYuv* pcPicYuv )       { m_pcPicYuvResi = pcPicYuv; }
  UInt          getNumberOfCtusInFrame() const     { return m_pcpicSym->getNumberOfCtusInFrame(); }
  UInt          getNumPartInCtuWidth() const       { return m_pcpicSym->getNumPartInCtuWidth();   }
  UInt          getNumPartInCtuHeight() const      { return m_pcpicSym->getNumPartInCtuHeight();  }
  UInt          getNumPartitionsInCtu() const      { return m_pcpicSym->getNumPartitionsInCtu();  }
  UInt          getFrameWidthInCtus() const        { return m_pcpicSym->getFrameWidthInCtus();    }
  UInt          getFrameHeightInCtus() const       { return m_pcpicSym->getFrameHeightInCtus();   }
  UInt          getMinCUWidth() const              { return m_pcpicSym->getMinCUWidth();          }
  UInt          getMinCUHeight() const             { return m_pcpicSym->getMinCUHeight();         }

  Int           getStride(const ComponentID id) const          { return m_apcPicYuv[PIC_YUV_REC]->getStride(id); }
  Int           getComponentScaleX(const ComponentID id) const    { return m_apcPicYuv[PIC_YUV_REC]->getComponentScaleX(id); }
  Int           getComponentScaleY(const ComponentID id) const    { return m_apcPicYuv[PIC_YUV_REC]->getComponentScaleY(id); }
  ChromaFormat  getChromaFormat() const                           { return m_apcPicYuv[PIC_YUV_REC]->getChromaFormat(); }
  Int           getNumberValidComponents() const                  { return m_apcPicYuv[PIC_YUV_REC]->getNumberValidComponents(); }

  Void          setReconMark (Bool b) { m_bReconstructed = b;     }
  Bool          getReconMark () const      { return m_bReconstructed;  }

  Void          setOutputMark (Bool b) { m_bNeededForOutput = b;     }
  Bool          getOutputMark () const      { return m_bNeededForOutput;  }

  Void          compressMotion();
  Void          compressMotionMDC();
  UInt          getCurrSliceIdx() const           { return m_uiCurrSliceIdx;                }
  Void          setCurrSliceIdx(UInt i)      { m_uiCurrSliceIdx = i;                   }
  UInt          getNumAllocatedSlice() const      {return m_pcpicSym->getNumAllocatedSlice();}
  Void          allocateNewSlice()           {m_pcpicSym->allocateNewSlice();         }
  Void          clearSliceBuffer()           {m_pcpicSym->clearSliceBuffer();         }

  const Window& getConformanceWindow() const { return m_pcpicSym->getSPS().getConformanceWindow(); }
  Window        getDefDisplayWindow() const  { return m_pcpicSym->getSPS().getVuiParametersPresentFlag() ? m_pcpicSym->getSPS().getVuiParameters()->getDefaultDisplayWindow() : Window(); }

  Bool          getSAOMergeAvailability(Int currAddr, Int mergeAddr);

  UInt          getSubstreamForCtuAddr(const UInt ctuAddr, const Bool bAddressInRaster, TComSlice *pcSlice);

  /* field coding parameters*/

   Void              setTopField(Bool b)                  {m_isTop = b;}
   Bool              isTopField()                         {return m_isTop;}
   Void              setField(Bool b)                     {m_isField = b;}
   Bool              isField() const                      {return m_isField;}

  /** transfer ownership of seis to this picture */
  Void setSEIs(SEIMessages& seis) { m_SEIs = seis; }

  /**
   * return the current list of SEI messages associated with this picture.
   * Pointer is valid until this->destroy() is called */
  SEIMessages& getSEIs() { return m_SEIs; }

  /**
   * return the current list of SEI messages associated with this picture.
   * Pointer is valid until this->destroy() is called */
  const SEIMessages& getSEIs() const { return m_SEIs; }
};// END CLASS DEFINITION TComPic

//! \}

#endif // __TCOMPIC__
