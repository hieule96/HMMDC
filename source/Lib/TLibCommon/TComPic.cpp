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

/** \file     TComPic.cpp
    \brief    picture class
*/

#include "TComPic.h"
#include "SEI.h"

//! \ingroup TLibCommon
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

TComPic::TComPic()
: m_uiTLayer                              (0)
, m_bUsedByCurr                           (false)
, m_bIsLongTerm                           (false)
, m_pcPicYuvPred                          (NULL)
, m_pcPicYuvResi                          (NULL)
, m_bReconstructed                        (false)
, m_bNeededForOutput                      (false)
, m_uiCurrSliceIdx                        (0)
, m_bCheckLTMSB                           (false)
, m_iRunMode                                (0)
, m_lamdaForcing                            (0)
, m_descriptionId                           (0)
{
  for(UInt i=0; i<NUM_PIC_YUV; i++)
  {
    m_apcPicYuv[i]      = NULL;
  }
}

TComPic::~TComPic()
{
  destroy();
}

#if REDUCED_ENCODER_MEMORY
Void TComPic::create( const TComSPS &sps, const TComPPS &pps, const Bool bCreateEncoderSourcePicYuv, const Bool bCreateForImmediateReconstruction )
#else
Void TComPic::create( const TComSPS &sps, const TComPPS &pps, const Bool bIsVirtual)
#endif
{
  destroy();

  const ChromaFormat chromaFormatIDC = sps.getChromaFormatIdc();
  const Int          iWidth          = sps.getPicWidthInLumaSamples();
  const Int          iHeight         = sps.getPicHeightInLumaSamples();
  const UInt         uiMaxCuWidth    = sps.getMaxCUWidth();
  const UInt         uiMaxCuHeight   = sps.getMaxCUHeight();
  const UInt         uiMaxDepth      = sps.getMaxTotalCUDepth();

#if REDUCED_ENCODER_MEMORY
  m_arrPicSym[0].create( sps, pps, uiMaxDepth, bCreateForImmediateReconstruction );
  m_arrPicSym[1].create( sps, pps, uiMaxDepth, bCreateForImmediateReconstruction );
  m_pcpicSym = &m_arrPicSym[0];
  if (bCreateEncoderSourcePicYuv)
#else
  m_picSym.create( sps, pps, uiMaxDepth );
  if (!bIsVirtual)
#endif
  {
    m_apcPicYuv[PIC_YUV_ORG    ]   = new TComPicYuv;  m_apcPicYuv[PIC_YUV_ORG     ]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
    m_apcPicYuv[PIC_YUV_TRUE_ORG]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_TRUE_ORG]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
  }
#if REDUCED_ENCODER_MEMORY
  if (bCreateForImmediateReconstruction)
  {
#endif
    m_apcPicYuv[PIC_YUV_REC]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_REC]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
    m_apcPicYuv[PIC_YUV_REC1]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_REC1]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
    m_apcPicYuv[PIC_YUV_REC2]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_REC2]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
#if REDUCED_ENCODER_MEMORY
  }
#endif

  // there are no SEI messages associated with this picture initially
  if (m_SEIs.size() > 0)
  {
    deleteSEIs (m_SEIs);
  }
  m_bUsedByCurr = false;
}

#if REDUCED_ENCODER_MEMORY
Void TComPic::prepareForEncoderSourcePicYuv()
{
  const TComSPS &sps=m_pcpicSym->getSPS();

  const ChromaFormat chromaFormatIDC = sps.getChromaFormatIdc();
  const Int          iWidth          = sps.getPicWidthInLumaSamples();
  const Int          iHeight         = sps.getPicHeightInLumaSamples();
  const UInt         uiMaxCuWidth    = sps.getMaxCUWidth();
  const UInt         uiMaxCuHeight   = sps.getMaxCUHeight();
  const UInt         uiMaxDepth      = sps.getMaxTotalCUDepth();

  if (m_apcPicYuv[PIC_YUV_ORG    ]==NULL)
  {
    m_apcPicYuv[PIC_YUV_ORG    ]   = new TComPicYuv;  m_apcPicYuv[PIC_YUV_ORG     ]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
  }
  if (m_apcPicYuv[PIC_YUV_TRUE_ORG    ]==NULL)
  {
    m_apcPicYuv[PIC_YUV_TRUE_ORG]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_TRUE_ORG]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
  }
}

Void TComPic::prepareForReconstruction()
{
  if (m_apcPicYuv[PIC_YUV_REC] == NULL)
  {
    const TComSPS &sps=m_pcpicSym->getSPS();
    const ChromaFormat chromaFormatIDC = sps.getChromaFormatIdc();
    const Int          iWidth          = sps.getPicWidthInLumaSamples();
    const Int          iHeight         = sps.getPicHeightInLumaSamples();
    const UInt         uiMaxCuWidth    = sps.getMaxCUWidth();
    const UInt         uiMaxCuHeight   = sps.getMaxCUHeight();
    const UInt         uiMaxDepth      = sps.getMaxTotalCUDepth();

    m_apcPicYuv[PIC_YUV_REC]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_REC]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
    m_apcPicYuv[PIC_YUV_REC1]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_REC1]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
    m_apcPicYuv[PIC_YUV_REC2]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_REC2]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );

  }

  // mark it should be extended
  m_apcPicYuv[PIC_YUV_REC]->setBorderExtension(false);
  m_apcPicYuv[PIC_YUV_REC1]->setBorderExtension(false);
  m_apcPicYuv[PIC_YUV_REC2]->setBorderExtension(false);

  m_arrPicSym[0].prepareForReconstruction();
  m_arrPicSym[1].prepareForReconstruction();

}

Void TComPic::releaseReconstructionIntermediateData()
{
  m_pcpicSym->releaseReconstructionIntermediateData();
}

Void TComPic::releaseEncoderSourceImageData()
{
  if (m_apcPicYuv[PIC_YUV_ORG    ])
  {
    m_apcPicYuv[PIC_YUV_ORG]->destroy();
    delete m_apcPicYuv[PIC_YUV_ORG];
    m_apcPicYuv[PIC_YUV_ORG] = NULL;
  }
  if (m_apcPicYuv[PIC_YUV_TRUE_ORG    ])
  {
    m_apcPicYuv[PIC_YUV_TRUE_ORG]->destroy();
    delete m_apcPicYuv[PIC_YUV_TRUE_ORG];
    m_apcPicYuv[PIC_YUV_TRUE_ORG] = NULL;
  }
}

Void TComPic::releaseAllReconstructionData()
{
  if (m_apcPicYuv[PIC_YUV_REC    ])
  {
    m_apcPicYuv[PIC_YUV_REC]->destroy();
    delete m_apcPicYuv[PIC_YUV_REC];
    m_apcPicYuv[PIC_YUV_REC] = NULL;
  }
  if (m_apcPicYuv[PIC_YUV_REC1    ])
  {
    m_apcPicYuv[PIC_YUV_REC1]->destroy();
    delete m_apcPicYuv[PIC_YUV_REC1];
    m_apcPicYuv[PIC_YUV_REC1] = NULL;
  }
  if (m_apcPicYuv[PIC_YUV_REC2    ])
  {
    m_apcPicYuv[PIC_YUV_REC2]->destroy();
    delete m_apcPicYuv[PIC_YUV_REC2];
    m_apcPicYuv[PIC_YUV_REC2] = NULL;
  }
  m_arrPicSym[0].releaseAllReconstructionData();
  m_arrPicSym[1].releaseAllReconstructionData();
}
#endif

Void TComPic::destroy()
{
  m_arrPicSym[0].destroy();
  m_arrPicSym[1].destroy();

  for(UInt i=0; i<NUM_PIC_YUV; i++)
  {
    if (m_apcPicYuv[i])
    {
      m_apcPicYuv[i]->destroy();
      delete m_apcPicYuv[i];
      m_apcPicYuv[i]  = NULL;
    }
  }

  deleteSEIs(m_SEIs);
}

Void TComPic::compressMotion()
{
  TComPicSym* pPicSym = getPicSym();
  for ( UInt uiCUAddr = 0; uiCUAddr < pPicSym->getNumberOfCtusInFrame(); uiCUAddr++ )
  {
    TComDataCU* pCtu = pPicSym->getCtu(uiCUAddr);
    pCtu->compressMV();
  }
}

Void TComPic::compressMotionMDC(){
  TComPicSym* pPicSymD1 = &m_arrPicSym[0];
  TComPicSym* pPicSymD2 = &m_arrPicSym[1];
  for ( UInt uiCUAddr = 0; uiCUAddr < pPicSymD1->getNumberOfCtusInFrame(); uiCUAddr++ )
  {
    TComDataCU* pCtu = pPicSymD1->getCtu(uiCUAddr);
    pCtu->compressMV();
  }
  for ( UInt uiCUAddr = 0; uiCUAddr < pPicSymD2->getNumberOfCtusInFrame(); uiCUAddr++ )
  {
    TComDataCU* pCtu = pPicSymD2->getCtu(uiCUAddr);
    pCtu->compressMV();
  }  
}

Bool  TComPic::getSAOMergeAvailability(Int currAddr, Int mergeAddr)
{
  Bool mergeCtbInSliceSeg = (mergeAddr >= getPicSym()->getCtuTsToRsAddrMap(getCtu(currAddr)->getSlice()->getSliceCurStartCtuTsAddr()));
  Bool mergeCtbInTile     = (getPicSym()->getTileIdxMap(mergeAddr) == getPicSym()->getTileIdxMap(currAddr));
  return (mergeCtbInSliceSeg && mergeCtbInTile);
}

UInt TComPic::getSubstreamForCtuAddr(const UInt ctuAddr, const Bool bAddressInRaster, TComSlice *pcSlice)
{
  UInt subStrm;
  const bool bWPPEnabled=pcSlice->getPPS()->getEntropyCodingSyncEnabledFlag();
  const TComPicSym &picSym            = *(getPicSym());

  if ((bWPPEnabled && picSym.getFrameHeightInCtus()>1) || (picSym.getNumTiles()>1)) // wavefronts, and possibly tiles being used.
  {
    if (bWPPEnabled)
    {
      const UInt ctuRsAddr                = bAddressInRaster?ctuAddr : picSym.getCtuTsToRsAddrMap(ctuAddr);
      const UInt frameWidthInCtus         = picSym.getFrameWidthInCtus();
      const UInt tileIndex                = picSym.getTileIdxMap(ctuRsAddr);
      const UInt numTileColumns           = (picSym.getNumTileColumnsMinus1()+1);
      const TComTile *pTile               = picSym.getTComTile(tileIndex);
      const UInt firstCtuRsAddrOfTile     = pTile->getFirstCtuRsAddr();
      const UInt tileYInCtus              = firstCtuRsAddrOfTile / frameWidthInCtus;
      // independent tiles => substreams are "per tile"
      const UInt ctuLine                  = ctuRsAddr / frameWidthInCtus;
      const UInt startingSubstreamForTile =(tileYInCtus*numTileColumns) + (pTile->getTileHeightInCtus()*(tileIndex%numTileColumns));
      subStrm = startingSubstreamForTile + (ctuLine - tileYInCtus);
    }
    else
    {
      const UInt ctuRsAddr                = bAddressInRaster?ctuAddr : picSym.getCtuTsToRsAddrMap(ctuAddr);
      const UInt tileIndex                = picSym.getTileIdxMap(ctuRsAddr);
      subStrm=tileIndex;
    }
  }
  else
  {
    // dependent tiles => substreams are "per frame".
    subStrm = 0;
  }
  return subStrm;
}


//! \}
