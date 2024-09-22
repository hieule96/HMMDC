/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2022, ITU/ISO/IEC
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

/** \file     TDecTop.cpp
    \brief    decoder class
*/

#include "NALread.h"
#include "TDecTop.h"
#include "TDecConformance.h"
#include "TMDCCommon/TMDCQPTable.hpp"
//! \ingroup TLibDecoder
//! \{

TDecTop::TDecTop()
    : m_iMaxRefPicNum(0), m_associatedIRAPType(NAL_UNIT_INVALID), m_pocCRA(0), m_pocRandomAccess(MAX_INT), m_cListPic(), m_parameterSetManager(), m_apcSlicePilot(NULL), m_SEIs(), m_cPrediction(), m_cTrQuant(), m_cGopDecoder(), m_cSliceDecoder(), m_cCuDecoder(), m_cEntropyDecoder(), m_cCavlcDecoder(), m_cSbacDecoder(), m_cBinCABAC(), m_seiReader(), m_cLoopFilter(), m_cSAO(), m_pcPic(NULL), m_prevPOC(MAX_INT), m_prevTid0POC(0), m_bFirstSliceInPicture(true)
#if FGS_RDD5_ENABLE
      ,
      m_bFirstPictureInSequence(true), m_grainCharacteristic(), m_grainBuf()
#endif
      ,
      m_bFirstSliceInSequence(true), m_prevSliceSkipped(false), m_skippedPOC(0), m_bFirstSliceInBitstream(true), m_lastPOCNoOutputPriorPics(-1), m_isNoOutputPriorPics(false), m_craNoRaslOutputFlag(false)
#if O0043_BEST_EFFORT_DECODING
      ,
      m_forceDecodeBitDepth(8)
#endif
      ,
      m_pDecodedSEIOutputStream(NULL), m_warningMessageSkipPicture(false)
#if MCTS_ENC_CHECK
      ,
      m_tmctsCheckEnabled(false)
#endif
      ,
      m_prefixSEINALUs()
{
#if ENC_DEC_TRACE
  if (g_hTrace == NULL)
  {
    g_hTrace = fopen("TraceDec.txt", "wb");
  }
  g_bJustDoIt = g_bEncDecTraceDisable;
  g_nSymbolCounter = 0;
#endif
}

TDecTop::~TDecTop()
{
#if ENC_DEC_TRACE
  if (g_hTrace != stdout)
  {
    fclose(g_hTrace);
  }
#endif
  while (!m_prefixSEINALUs.empty())
  {
    delete m_prefixSEINALUs.front();
    m_prefixSEINALUs.pop_front();
  }
}

Void TDecTop::create(Int DescriptionId)
{
  m_cGopDecoder.create();
  m_DecoderDescriptionId = DescriptionId;
  m_apcSlicePilot = new TComSlice;
  m_uiSliceIdx = 0;
}

Void TDecTop::destroy()
{
  m_cGopDecoder.destroy();

  delete m_apcSlicePilot;
  m_apcSlicePilot = NULL;

  m_cSliceDecoder.destroy();
}

Void TDecTop::init()
{
  // initialize ROM
  initROM();
  m_cGopDecoder.init(&m_cEntropyDecoder, &m_cSbacDecoder, &m_cBinCABAC, &m_cCavlcDecoder, &m_cSliceDecoder, &m_cLoopFilter, &m_cSAO);
  m_cSliceDecoder.init(&m_cEntropyDecoder, &m_cCuDecoder, &m_conformanceCheck);
#if MCTS_ENC_CHECK
  m_cEntropyDecoder.init(&m_cPrediction, &m_conformanceCheck);
#else
  m_cEntropyDecoder.init(&m_cPrediction);
#endif
}

Void TDecTop::deletePicBuffer()
{
  TComList<TComPic *>::iterator iterPic = m_cListPic.begin();
  Int iSize = Int(m_cListPic.size());

  for (Int i = 0; i < iSize; i++)
  {
    TComPic *pcPic = *(iterPic++);
    pcPic->destroy();

    delete pcPic;
    pcPic = NULL;
  }

  m_cSAO.destroy();

  m_cLoopFilter.destroy();

  // destroy ROM
}

Void TDecTop::xGetNewPicBuffer(const TComSPS &sps, const TComPPS &pps, TComPic *&rpcPic, const UInt temporalLayer)
{
  m_iMaxRefPicNum = sps.getMaxDecPicBuffering(temporalLayer); // m_uiMaxDecPicBuffering has the space for the picture currently being decoded
  if (m_cListPic.size() < (UInt)m_iMaxRefPicNum)
  {
    rpcPic = new TComPic();

#if REDUCED_ENCODER_MEMORY
#if SHUTTER_INTERVAL_SEI_PROCESSING
    rpcPic->create(sps, pps, false, true, getShutterFilterFlag());
#else
    rpcPic->create(sps, pps, false, true);
    rpcPic->setDescriptionId(m_DecoderDescriptionId);

    // Mention to the Pic that we are in which description respect to the decoder
#endif
#else
#if SHUTTER_INTERVAL_SEI_PROCESSING
    rpcPic->create(sps, pps, true, getShutterFilterFlag());
#else
    rpcPic->create(sps, pps, true);
#endif
#endif

    m_cListPic.pushBack(rpcPic);

    return;
  }

  Bool bBufferIsAvailable = false;
  TComList<TComPic *>::iterator iterPic = m_cListPic.begin();
  while (iterPic != m_cListPic.end())
  {
    rpcPic = *(iterPic++);
    if (rpcPic->getReconMark() == false && rpcPic->getOutputMark() == false)
    {
      rpcPic->setOutputMark(false);
      bBufferIsAvailable = true;
      break;
    }

    if (rpcPic->getSlice(0)->isReferenced() == false && rpcPic->getOutputMark() == false)
    {
      rpcPic->setOutputMark(false);
      rpcPic->setReconMark(false);
      rpcPic->getPicYUVRecC()->setBorderExtension(false);
      rpcPic->getPicYUVRec1()->setBorderExtension(false);
      rpcPic->getPicYUVRec2()->setBorderExtension(false);

      bBufferIsAvailable = true;
      break;
    }
  }

  if (!bBufferIsAvailable)
  {
    // There is no room for this picture, either because of faulty encoder or dropped NAL. Extend the buffer.
    m_iMaxRefPicNum++;
    rpcPic = new TComPic();
    m_cListPic.pushBack(rpcPic);
  }
  rpcPic->destroy();
#if REDUCED_ENCODER_MEMORY
#if SHUTTER_INTERVAL_SEI_PROCESSING
  rpcPic->create(sps, pps, false, true, getShutterFilterFlag());
#else
  rpcPic->create(sps, pps, false, true);
  //@tle: Put two time in two cases different, the code is not very organized
  rpcPic->setDescriptionId(m_DecoderDescriptionId);
#endif
#else
#if SHUTTER_INTERVAL_SEI_PROCESSING
  rpcPic->create(sps, pps, true, getShutterFilterFlag());
#else
  rpcPic->create(sps, pps, true);
#endif
#endif
}

Void TDecTop::xSelectCu(TComPic *pcPicRef,TComDataCU* &pcCUA,TComDataCU *&pcCUB, UInt uiAbsPartIdx, UInt uiDepth, const Int *QPTable1, const Int *QPTable2, Int &index)
{
  TComPic *pcPicA = pcCUA->getPic();
  TComPic *pcPicB = pcCUB->getPic();
  TComPic *pcPic = pcPicA ? pcPicA : pcPicB;
  TComSlice *const pcSliceA = pcCUA->getSlice();
  TComSlice *const pcSliceB = pcCUB->getSlice();
  // pcPicA and pcPicB derive from a picture, but in two independent descriptions
  // check if pcSliceA and pcSliceB are not null first, either pcSliceA or pcSliceB is null pcSlice will be the other one
  // if both are null, then pcSlice will be null and exit the function
  TComSlice *const pcSlice = pcSliceA ? pcSliceA : pcSliceB;
  const TComDataCU *pcCU = pcSliceA ? pcCUA : pcCUB;
  if (pcCUA->getIsCorrupted()){
    pcCU = pcCUB;
  }
  else if (pcCUB->getIsCorrupted()){
    pcCU = pcCUA;
  }
  if (pcSlice == NULL)
  {
    // in this case, pcCUA and pcCUB are both null, so pcCU is null
    // we don't have any choice left but to replace theses CU with preceding ones
    return;
  }
  // check which sps of which is null or not ?
  const TComSPS *sps = pcSlice->getSPS();
  if (sps == NULL)
  {
    // in this case, pcSliceA and pcSliceB are both null, so sps is null
    // we don't have any choice left but to replace theses CU with preceding ones
    return;
  }
  const TComPPS *pps = pcSlice->getPPS();
  if (pps == NULL)
  {
    // in this case, pcSliceA and pcSliceB are both null, so pps is null
    // we don't have any choice left but to replace theses CU with preceding ones
    return;
  }
  const UInt maxCUWidth = sps->getMaxCUWidth();
  const UInt maxCUHeight = sps->getMaxCUHeight();

  Bool bBoundary = false;
  UInt uiLPelX = pcCU->getCUPelX() + g_auiRasterToPelX[g_auiZscanToRaster[uiAbsPartIdx]];
  const UInt uiRPelX = uiLPelX + (maxCUWidth >> uiDepth) - 1;
  UInt uiTPelY = pcCU->getCUPelY() + g_auiRasterToPelY[g_auiZscanToRaster[uiAbsPartIdx]];
  const UInt uiBPelY = uiTPelY + (maxCUHeight >> uiDepth) - 1;

  // quadtree
  if (((uiDepth < pcCU->getDepth(uiAbsPartIdx)) && (uiDepth < sps->getLog2DiffMaxMinCodingBlockSize())) || bBoundary)
  {
    UInt uiQNumParts = (pcPic->getNumPartitionsInCtu() >> (uiDepth << 1)) >> 2;
    for (UInt uiPartUnitIdx = 0; uiPartUnitIdx < 4; uiPartUnitIdx++, uiAbsPartIdx += uiQNumParts)
    {
      uiLPelX = pcCU->getCUPelX() + g_auiRasterToPelX[g_auiZscanToRaster[uiAbsPartIdx]];
      uiTPelY = pcCU->getCUPelY() + g_auiRasterToPelY[g_auiZscanToRaster[uiAbsPartIdx]];
      if ((uiLPelX < sps->getPicWidthInLumaSamples()) && (uiTPelY < sps->getPicHeightInLumaSamples()))
      {
        index++;
        xSelectCu(pcPicRef,pcCUA, pcCUB, uiAbsPartIdx, uiDepth + 1,QPTable1,QPTable2,index);
      }
    }
    return;
  }
  // Check if two Pic has the same structure
  // assert(pcCUA->getHeight(uiAbsPartIdx) == pcCUB->getHeight(uiAbsPartIdx));
  // assert(pcCUA->getHeight(uiAbsPartIdx) == pcCUB->getHeight(uiAbsPartIdx));
  // for each CU in the quadtree, perform the merging process
  if (pcPicA==NULL && pcPicB==NULL){
    return;
  }
  else if(pcCUA==NULL||pcPicA==NULL)//pcSliceA!=NULL&pcSliceB!=NULL&&pcSliceA->getIsCorrupted()&&!pcSliceB->getIsCorrupted())
  {
    UChar width = Clip3 ((UChar)0,(UChar)(64 >> uiDepth),pcCUB->getWidth(uiAbsPartIdx));
    UChar height = Clip3 ((UChar)0,(UChar)(64 >> uiDepth),pcCUB->getHeight(uiAbsPartIdx));
    // std::cout << "B:"<< (Int)width <<"/"<<(Int)height<<std::endl;
    pcPicB->getPicYUVRec2()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUB->getCtuRsAddr(), uiAbsPartIdx, height, width);
    return;
  }
  else if(pcCUB==NULL||pcPicB==NULL)//pcSliceB!=NULL&pcSliceA!=NULL&&pcSliceB->getIsCorrupted()&&!pcSliceA->getIsCorrupted())
  {
    UChar width = Clip3 ((UChar)0,(UChar)(64 >> uiDepth),pcCUA->getWidth(uiAbsPartIdx));
    UChar height = Clip3 ((UChar)0,(UChar)(64 >> uiDepth),pcCUA->getHeight(uiAbsPartIdx));
    // std::cout << "A:"<< (Int)width <<"/"<<(Int)height<<std::endl;

    pcPicA->getPicYUVRec1()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUA->getCtuRsAddr(), uiAbsPartIdx, height, width);
    return;
  }

  if (!pcCUA->getIsCorrupted()&&pcCUB->getIsCorrupted()){
    std::cout<<"pcCUB->getIsCorrupted()"<<pcCUA->getCtuRsAddr()<<"/"<<pcCUB->getCtuRsAddr()<<std::endl;
    std::cout <<"H/W"<<(Int)pcCUB->getHeight(uiAbsPartIdx)<<"/"<<(Int)pcCUB->getWidth(uiAbsPartIdx)<<std::endl;
    pcPicA->getPicYUVRec1()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUA->getCtuRsAddr(), uiAbsPartIdx, pcCUA->getHeight(uiAbsPartIdx), pcCUA->getWidth(uiAbsPartIdx));
    return;
  }
  else if (pcCUA->getIsCorrupted()&&!pcCUB->getIsCorrupted()){
    std::cout<<"pcCUA->getIsCorrupted()"<<pcCUA->getCtuRsAddr()<<"/"<<pcCUB->getCtuRsAddr() << std::endl;
    std::cout <<"H/W"<<(Int) pcCUA->getHeight(uiAbsPartIdx)<<"/"<<(Int) pcCUB->getWidth(uiAbsPartIdx)<<std::endl;
    pcPicB->getPicYUVRec2()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUB->getCtuRsAddr(), uiAbsPartIdx, pcCUB->getHeight(uiAbsPartIdx), pcCUB->getWidth(uiAbsPartIdx));
    return;
  }

  if (pcCUA->getCbf(uiAbsPartIdx, COMPONENT_Y) && !pcCUB->getCbf(uiAbsPartIdx, COMPONENT_Y))
  {
    pcPicA->getPicYUVRec1()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUA->getCtuRsAddr(), uiAbsPartIdx, pcCUA->getHeight(uiAbsPartIdx), pcCUA->getWidth(uiAbsPartIdx));
    return;
  }
  else if (!pcCUA->getCbf(uiAbsPartIdx, COMPONENT_Y) && pcCUB->getCbf(uiAbsPartIdx, COMPONENT_Y))
  {
    pcPicB->getPicYUVRec2()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUB->getCtuRsAddr(), uiAbsPartIdx, pcCUB->getHeight(uiAbsPartIdx), pcCUB->getWidth(uiAbsPartIdx));
    return;
  }



  if (QPTable1[index]<QPTable2[index]){
    pcPicA->getPicYUVRec1()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUA->getCtuRsAddr(), uiAbsPartIdx, pcCUA->getHeight(uiAbsPartIdx), pcCUA->getWidth(uiAbsPartIdx));
  }else
  {
    pcPicB->getPicYUVRec2()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCUB->getCtuRsAddr(), uiAbsPartIdx, pcCUB->getHeight(uiAbsPartIdx), pcCUB->getWidth(uiAbsPartIdx));
  }
  // if (pcCUA->getCodedQP() < pcCUB->getCodedQP())
  // {
  //   pcPicA->getPicYUVRec1()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCU->getCtuRsAddr(), uiAbsPartIdx, pcCU->getHeight(uiAbsPartIdx), pcCU->getWidth(uiAbsPartIdx));
  // }
  // else
  // {
  //   pcPicB->getPicYUVRec2()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCU->getCtuRsAddr(), uiAbsPartIdx, pcCU->getHeight(uiAbsPartIdx), pcCU->getWidth(uiAbsPartIdx));
  // }
  // else
  // {
  //   if (pcCUA->getRefQP(uiAbsPartIdx) < pcCUB->getRefQP(uiAbsPartIdx))
  //   {
  //     pcPicA->getPicYUVRec1()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCU->getCtuRsAddr(), uiAbsPartIdx, pcCU->getHeight(uiAbsPartIdx), pcCU->getWidth(uiAbsPartIdx));
  //   }
  //   else if (pcCUA->getRefQP(uiAbsPartIdx) > pcCUB->getRefQP(uiAbsPartIdx))
  //   {
  //     pcPicB->getPicYUVRec2()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCU->getCtuRsAddr(), uiAbsPartIdx, pcCU->getHeight(uiAbsPartIdx), pcCU->getWidth(uiAbsPartIdx));
  //   }
  //   else
  //   {
  //     // all information which serve to decide is over, select any between two
  //     pcPicB->getPicYUVRec2()->copyCUToPic(pcPicRef->getPicYUVRecC(), pcCU->getCtuRsAddr(), uiAbsPartIdx, pcCU->getHeight(uiAbsPartIdx), pcCU->getWidth(uiAbsPartIdx));
  //   }
  // }
}

Void TDecTop::outputBuffer(TComPic *pcPicA, TComPic *pcPicB, BitDepths const &bitDepths, Int POC){
  if (POC > 0){
    // pcPicA->getPicYUVRec1()->dump("debugoutputA.yuv",bitDepths,true,true);
    // pcPicB->getPicYUVRec2()->dump("debugoutputB.yuv",bitDepths,true, true);
    pcPicA->getPicYUVRecC()->dump("debugoutputCentralA.yuv",bitDepths,true, true);
    // pcPicB->getPicYUVRecC()->dump("debugoutputCentralB.yuv",bitDepths,true, true);
  }
  else
  {
    // pcPicA->getPicYUVRec1()->dump("debugoutputA.yuv",bitDepths,false,true);
    // pcPicB->getPicYUVRec2()->dump("debugoutputB.yuv",bitDepths,false, true);
    pcPicA->getPicYUVRecC()->dump("debugoutputCentralA.yuv",bitDepths,false, true);
    // pcPicB->getPicYUVRecC()->dump("debugoutputCentralB.yuv",bitDepths,false, true);
  }
}

Void TDecTop::mergingMDC(TDecTop &rTdec2,TDecCtx &ctx1, TDecCtx &ctx2)
{
  TComPic *pcPicA = this->getPcPic();
  TComPic *pcPicB = rTdec2.getPcPic();
  TComSlice *pcSlice;
  Int POCD1 = 0;
  Int POCD2 = 0;
  if (pcPicA){
    pcSlice =pcPicA->getSlice(0);
  }
  if (pcPicB)
  {
    pcSlice = pcPicB->getSlice(0);
  }
  if (pcPicA&&pcPicB){
    POCD1 = pcPicA->getPOC();
    POCD2 = pcPicB->getPOC();
  }
  else if (pcPicA)
  {
    POCD2 = pcPicA->getPOC();
  }
  else if (pcPicB)
  {
    POCD1 = pcPicB->getPOC();
  }
  ctx1.POC = POCD1;
  ctx2.POC = POCD2;
  auto iterPic = m_cListPic.begin();
  // // prepare with previous frame in case of lost
  while (iterPic != m_cListPic.end())
  {
    TComPic *rpcPic = *(iterPic++);
    if (rpcPic->getPicSym()->getSlice(0)->getPOC() == POCD1-1)
    {
      rpcPic->getPicYUVRecC()->copyToPic(pcPicA->getPicYUVRecC());
      break;
    }
  }
  auto iterPic2 = rTdec2.m_cListPic.begin();
  while (iterPic2 != rTdec2.m_cListPic.end())
  {
    TComPic *rpcPic = *(iterPic2++);
    if (rpcPic->getPicSym()->getSlice(0)->getPOC() == POCD2-1)
    {
      rpcPic->getPicYUVRecC()->copyToPic(pcPicB->getPicYUVRecC());
      break;
    }
  }
  if (m_cListPic.size() > 0)
  {
    m_cListPic.back()->getPicYUVRecC()->copyToPic(pcPicA->getPicYUVRecC());
  }
  // the list m_cListPic contains the previous frame for references, it can happened that the 
  if (ctx1.LostPOC.size()>0&&ctx2.LostPOC.size()>0){
    ctx1.getMore = true;
    ctx2.getMore = true;
    ctx1.LostPOC.pop_back();
    ctx2.LostPOC.pop_back();
    outputBuffer(pcPicA,pcPicB,pcSlice->getSPS()->getBitDepths(),POCD1);
    return;
  }
  else if (ctx2.LostPOC.size()>0){
    // check if there is a lost in ctx 2
    if (ctx2.LostPOC.back()==POCD1){
      ctx2.LostPOC.pop_back();
      executeLoopFilters();
      pcPicA->getPicYUVRec1()->copyToPic(pcPicB->getPicYUVRecC());
      pcPicA->getPicYUVRec1()->copyToPic(pcPicB->getPicYUVRec2());
      pcPicA->getPicYUVRec1()->copyToPic(pcPicA->getPicYUVRecC());
      outputBuffer(pcPicA,pcPicB,pcSlice->getSPS()->getBitDepths(),POCD1);
    }else
    {
      if (POCD1>ctx2.LostPOC.back())
        // create lost pic by using the in description 1
      {
          Int count=0;
          Int lostPoc=0;
          while ((lostPoc = m_apcSlicePilot->checkThatAllRefPicsAreAvailable(m_cListPic, m_apcSlicePilot->getRPS(), true, m_pocRandomAccess)) > 0)
          {
            rTdec2.xCreateLostPicture(lostPoc - 1,pcPicB->getDescriptionId(),rTdec2.m_cListPic);
          }
          if (ctx2.LostPOC.back()==pcPicB->getPOC()){
            ctx2.LostPOC.pop_back();
            executeLoopFilters();
            pcPicA->getPicYUVRec1()->copyToPic(pcPicB->getPicYUVRecC());
            pcPicA->getPicYUVRec1()->copyToPic(pcPicB->getPicYUVRec2());
            pcPicA->getPicYUVRec1()->copyToPic(pcPicA->getPicYUVRecC());
            outputBuffer(pcPicA,pcPicB,pcSlice->getSPS()->getBitDepths(),POCD1);

          }
      }
    }
  return;
  }
  else if(ctx1.LostPOC.size()>0)
  {
    if (ctx1.LostPOC.back()==POCD2){
      // correct the frame in the past
      ctx1.LostPOC.pop_back();
      rTdec2.executeLoopFilters();
      pcPicB->getPicYUVRec2()->copyToPic(pcPicA->getPicYUVRecC());
      pcPicB->getPicYUVRec2()->copyToPic(pcPicA->getPicYUVRec1());
      pcPicB->getPicYUVRec2()->copyToPic(pcPicB->getPicYUVRecC());
      outputBuffer(pcPicA,pcPicB,pcSlice->getSPS()->getBitDepths(),POCD2);

    }
    else
    {
      if (POCD2>ctx1.LostPOC.back())
      {
        Int count=0;
        Int lostPoc=0;
        while ((lostPoc = m_apcSlicePilot->checkThatAllRefPicsAreAvailable(m_cListPic, m_apcSlicePilot->getRPS(), true, m_pocRandomAccess)) > 0)
        {
          xCreateLostPicture(lostPoc - 1,pcPicA->getDescriptionId(),m_cListPic);
        }
        if (ctx1.LostPOC.back()==pcPicA->getPOC()){
          ctx1.LostPOC.pop_back();
          rTdec2.executeLoopFilters();
          pcPicB->getPicYUVRec2()->copyToPic(pcPicA->getPicYUVRecC());
          pcPicB->getPicYUVRec2()->copyToPic(pcPicA->getPicYUVRec1());
          pcPicB->getPicYUVRec2()->copyToPic(pcPicB->getPicYUVRecC());
          outputBuffer(pcPicA,pcPicB,pcSlice->getSPS()->getBitDepths(),POCD2);

        }
      }
    }
    return;
  }
  if (POCD1!=POCD2){
    return;
  }

  Int index = 0;
  const UInt numberOfCtusInFrame = this->getPcPic()->getNumberOfCtusInFrame();
  TMDCQPTable *pqptable = TMDCQPTable::getInstance();
  Int *QPTable1 = new Int[1024];
  Int *QPTable2 = new Int[1024];
  pqptable->seekLine(DESCRIPTION1,numberOfCtusInFrame*POCD1);
  pqptable->seekLine(DESCRIPTION2,numberOfCtusInFrame*POCD2);
  for (int i = 0; i < numberOfCtusInFrame; i++)
  {
    index = 0;
    Int countQP1 = pqptable->readALine(DESCRIPTION1);
    memcpy(QPTable1,pqptable->getQPArray(),countQP1*sizeof(Int));
    Int countQP2 = pqptable->readALine(DESCRIPTION2);
    memcpy(QPTable2,pqptable->getQPArray(),countQP2*sizeof(Int));
    TComDataCU *pCtuD1 = pcPicA->getPicSym()->getCtu(i);
    TComDataCU *pCtuD2 = pcPicB->getPicSym()->getCtu(i);

    xSelectCu(pcPicA,pCtuD1, pCtuD2, 0, 0, QPTable1,QPTable2,index);
    // printf("QP array1[%d,%d]:",pCtuD1->getCtuRsAddr(),countQP1);
    // for (int i =0;i<countQP1;i++){
    //   printf("%d ",QPTable1[i]);
    // }
    // std::cout << std::endl;
    // printf("QP array2[%d,%d]:",pCtuD2->getCtuRsAddr(),countQP2);
    // for (int i =0;i<countQP2;i++){
    //   printf("%d ",QPTable2[i]);
    // }
    // std::cout << std::endl;
  }
  pcPicA->getPicYUVRecC()->copyToPic(pcPicB->getPicYUVRecC());
  executeLoopFilters();
  rTdec2.executeLoopFilters();
  outputBuffer(pcPicA,pcPicB,pcSlice->getSPS()->getBitDepths(),POCD1);

  // unifying the best picture
  ctx1.getMore = true;
  ctx2.getMore = true;
}
// this function sort the pic list and update the poc and rpcListPic
Void TDecTop::sortMoveToTheNext(Int &poc, TComList<TComPic *> *&rpcListPic){
  if (!m_pcPic)
  {
    /* nothing to deblock */
    return;
  }
  TComPic *pcPic = m_pcPic;
  // pcPic->setOutputMark(pcPic->getSlice(0)->getPicOutputFlag() ? true : false);
  // pcPic->setReconMark(true);
  TComSlice::sortPicList(m_cListPic); // sorting for application output
  poc = pcPic->getSlice((m_uiSliceIdx - 1) > 0 ? m_uiSliceIdx - 1:0)->getPOC();
  rpcListPic = &m_cListPic;
  m_cCuDecoder.destroy();
  m_bFirstSliceInPicture = true;
}
Void TDecTop::executeLoopFilters()
{
  if (!m_pcPic)
  {
    /* nothing to deblock */
    return;
  }
  TComPic *pcPic = m_pcPic;
  // Execute Deblock + Cleanup
  m_cGopDecoder.filterPicture(pcPic);
}

Void TDecTop::checkNoOutputPriorPics(TComList<TComPic *> *pcListPic)
{
  if (!pcListPic || !m_isNoOutputPriorPics)
  {
    return;
  }

  TComList<TComPic *>::iterator iterPic = pcListPic->begin();

  while (iterPic != pcListPic->end())
  {
    TComPic *pcPicTmp = *(iterPic++);
    if (m_lastPOCNoOutputPriorPics != pcPicTmp->getPOC())
    {
      pcPicTmp->setOutputMark(false);
    }
  }
}

Void TDecTop::xCreateLostPicture(Int iLostPoc, Int descriptionId, TComList<TComPic *> &rpcListPic)
{
  printf("\ninserting lost poc : %d of D%d\n", iLostPoc,m_DecoderDescriptionId);
  TComPic *cFillPic;
  xGetNewPicBuffer(*(m_parameterSetManager.getFirstSPS()), *(m_parameterSetManager.getFirstPPS()), cFillPic, 0);
  cFillPic->getSlice(0)->initSlice();
  cFillPic->setDescriptionId(descriptionId);
  TComList<TComPic *>::iterator iterPic = rpcListPic.begin();
  Int closestPoc = 1000000;
  while (iterPic != rpcListPic.end())
  {
    TComPic *rpcPic = *(iterPic++);
    if (abs(rpcPic->getPicSym()->getSlice(0)->getPOC() - iLostPoc) < closestPoc && abs(rpcPic->getPicSym()->getSlice(0)->getPOC() - iLostPoc) != 0 && rpcPic->getPicSym()->getSlice(0)->getPOC() != m_apcSlicePilot->getPOC())
    {
      closestPoc = abs(rpcPic->getPicSym()->getSlice(0)->getPOC() - iLostPoc);
    }
  }
  iterPic = rpcListPic.begin();
  while (iterPic != rpcListPic.end())
  {
    TComPic *rpcPic = *(iterPic++);
    if (abs(rpcPic->getPicSym()->getSlice(0)->getPOC() - iLostPoc) == closestPoc && rpcPic->getPicSym()->getSlice(0)->getPOC() != m_apcSlicePilot->getPOC())
    {
      printf("copying picture %d to %d (%d)\n", rpcPic->getPicSym()->getSlice(0)->getPOC(), iLostPoc, m_apcSlicePilot->getPOC());
      rpcPic->getPicYuvRec()->copyToPic(cFillPic->getPicYuvRec());
      rpcPic->getPicYUVRecC()->copyToPic(cFillPic->getPicYUVRecC());
      break;
    }
  }
  cFillPic->setCurrSliceIdx(0);
  for (Int ctuRsAddr = 0; ctuRsAddr < cFillPic->getNumberOfCtusInFrame(); ctuRsAddr++)
  {
    cFillPic->getCtu(ctuRsAddr)->initCtu(cFillPic, ctuRsAddr);
  }
  cFillPic->getSlice(0)->setReferenced(true);
  cFillPic->getSlice(0)->setPOC(iLostPoc);
  xUpdatePreviousTid0POC(cFillPic->getSlice(0));
  cFillPic->setReconMark(true);
  cFillPic->setOutputMark(false);
  if (m_pocRandomAccess == MAX_INT)
  {
    m_pocRandomAccess = iLostPoc;
  }
  m_pcPic = cFillPic;
}

#if MCTS_EXTRACTION
Void TDecTop::xActivateParameterSets(Bool bSkipCabacAndReconstruction)
#else
Void TDecTop::xActivateParameterSets()
#endif
{
  if (m_bFirstSliceInPicture)
  {
    const TComPPS *pps = m_parameterSetManager.getPPS(m_apcSlicePilot->getPPSId()); // this is a temporary PPS object. Do not store this value
    assert(pps != 0);

    const TComSPS *sps = m_parameterSetManager.getSPS(pps->getSPSId()); // this is a temporary SPS object. Do not store this value
    assert(sps != 0);

    m_parameterSetManager.clearSPSChangedFlag(sps->getSPSId());
    m_parameterSetManager.clearPPSChangedFlag(pps->getPPSId());

    if (false == m_parameterSetManager.activatePPS(m_apcSlicePilot->getPPSId(), m_apcSlicePilot->isIRAP()))
    {
      printf("Parameter set activation failed!");
      assert(0);
    }

    xParsePrefixSEImessages();
#if MCTS_ENC_CHECK
    xAnalysePrefixSEImessages();
#endif

#if RExt__HIGH_BIT_DEPTH_SUPPORT == 0
    if (sps->getSpsRangeExtension().getExtendedPrecisionProcessingFlag() || sps->getBitDepth(CHANNEL_TYPE_LUMA) > 12 || sps->getBitDepth(CHANNEL_TYPE_CHROMA) > 12)
    {
      printf("High bit depth support must be enabled at compile-time in order to decode this bitstream\n");
      assert(0);
      exit(1);
    }
#endif

    // NOTE: globals were set up here originally. You can now use:
    // g_uiMaxCUDepth = sps->getMaxTotalCUDepth();
    // g_uiAddCUDepth = sps->getMaxTotalCUDepth() - sps->getLog2DiffMaxMinCodingBlockSize()

    //  Get a new picture buffer. This will also set up m_pcPic, and therefore give us a SPS and PPS pointer that we can use.
    xGetNewPicBuffer(*(sps), *(pps), m_pcPic, m_apcSlicePilot->getTLayer());
    m_apcSlicePilot->applyReferencePictureSet(m_cListPic, m_apcSlicePilot->getRPS());
#if FGS_RDD5_ENABLE
    // Initialization of film grain synthesizer
    m_pcPic->createGrainSynthesizer(m_bFirstPictureInSequence, &m_grainCharacteristic, &m_grainBuf, sps);
    m_bFirstPictureInSequence = false;
#endif
    // make the slice-pilot a real slice, and set up the slice-pilot for the next slice
    assert(m_pcPic->getNumAllocatedSlice() == (m_uiSliceIdx + 1));
    m_apcSlicePilot = m_pcPic->getPicSym()->swapSliceObject(m_apcSlicePilot, m_uiSliceIdx);

    // we now have a real slice:
    TComSlice *pSlice = m_pcPic->getSlice(m_uiSliceIdx);

    // Update the PPS and SPS pointers with the ones of the picture.
    pps = pSlice->getPPS();
    sps = pSlice->getSPS();

    // Initialise the various objects for the new set of settings
    m_cSAO.create(sps->getPicWidthInLumaSamples(), sps->getPicHeightInLumaSamples(), sps->getChromaFormatIdc(), sps->getMaxCUWidth(), sps->getMaxCUHeight(), sps->getMaxTotalCUDepth(), pps->getPpsRangeExtension().getLog2SaoOffsetScale(CHANNEL_TYPE_LUMA), pps->getPpsRangeExtension().getLog2SaoOffsetScale(CHANNEL_TYPE_CHROMA));
    m_cLoopFilter.create(sps->getMaxTotalCUDepth());
    m_cPrediction.initTempBuff(sps->getChromaFormatIdc());

    Bool isField = false;
    Bool isTopField = false;

    if (!m_SEIs.empty())
    {
      // Check if any new Picture Timing SEI has arrived
      SEIMessages pictureTimingSEIs = getSeisByType(m_SEIs, SEI::PICTURE_TIMING);
      if (pictureTimingSEIs.size() > 0)
      {
        SEIPictureTiming *pictureTiming = (SEIPictureTiming *)*(pictureTimingSEIs.begin());
        isField = (pictureTiming->m_picStruct == 1) || (pictureTiming->m_picStruct == 2) || (pictureTiming->m_picStruct == 9) || (pictureTiming->m_picStruct == 10) || (pictureTiming->m_picStruct == 11) || (pictureTiming->m_picStruct == 12);
        isTopField = (pictureTiming->m_picStruct == 1) || (pictureTiming->m_picStruct == 9) || (pictureTiming->m_picStruct == 11);
      }
    }

    // Set Field/Frame coding mode
    m_pcPic->setField(isField);
    m_pcPic->setTopField(isTopField);

    // transfer any SEI messages that have been received to the picture
    m_pcPic->setSEIs(m_SEIs);
    m_SEIs.clear();
#if MCTS_EXTRACTION
    if (!bSkipCabacAndReconstruction)
    {
#endif
      // Recursive structure
      m_cCuDecoder.create(sps->getMaxTotalCUDepth(), sps->getMaxCUWidth(), sps->getMaxCUHeight(), sps->getChromaFormatIdc());
#if MCTS_ENC_CHECK
      m_cCuDecoder.init(&m_cEntropyDecoder, &m_cTrQuant, &m_cPrediction, &m_conformanceCheck);
#else
    m_cCuDecoder.init(&m_cEntropyDecoder, &m_cTrQuant, &m_cPrediction);
#endif
      m_cTrQuant.init(sps->getMaxTrSize());

      m_cSliceDecoder.create(sps);
    }
#if MCTS_EXTRACTION
  }
#endif
  else
  {
    // make the slice-pilot a real slice, and set up the slice-pilot for the next slice
    m_pcPic->allocateNewSlice();
    // assert(m_pcPic->getNumAllocatedSlice() == (m_uiSliceIdx + 1));
    m_apcSlicePilot = m_pcPic->getPicSym()->swapSliceObject(m_apcSlicePilot, m_uiSliceIdx);

    TComSlice *pSlice = m_pcPic->getSlice(m_uiSliceIdx); // we now have a real slice.

    const TComSPS *sps = pSlice->getSPS();
    const TComPPS *pps = pSlice->getPPS();

    // check that the current active PPS has not changed...
    if (m_parameterSetManager.getSPSChangedFlag(sps->getSPSId()))
    {
      printf("Error - a new SPS has been decoded while processing a picture\n");
      exit(1);
    }
    if (m_parameterSetManager.getPPSChangedFlag(pps->getPPSId()))
    {
      printf("Error - a new PPS has been decoded while processing a picture\n");
      exit(1);
    }

    xParsePrefixSEImessages();
#if MCTS_ENC_CHECK
    xAnalysePrefixSEImessages();
#endif
    // Check if any new SEI has arrived
    if (!m_SEIs.empty())
    {
      // Currently only decoding Unit SEI message occurring between VCL NALUs copied
      SEIMessages &picSEI = m_pcPic->getSEIs();
      SEIMessages decodingUnitInfos = extractSeisByType(m_SEIs, SEI::DECODING_UNIT_INFO);
      picSEI.insert(picSEI.end(), decodingUnitInfos.begin(), decodingUnitInfos.end());
      deleteSEIs(m_SEIs);
    }
  }
}

Void TDecTop::xParsePrefixSEIsForUnknownVCLNal()
{
  while (!m_prefixSEINALUs.empty())
  {
    // do nothing?
    printf("Discarding Prefix SEI associated with unknown VCL NAL unit.\n");
    delete m_prefixSEINALUs.front();
  }
  // TODO: discard following suffix SEIs as well?
}

Void TDecTop::xParsePrefixSEImessages()
{
  while (!m_prefixSEINALUs.empty())
  {
    InputNALUnit &nalu = *m_prefixSEINALUs.front();
    m_seiReader.parseSEImessage(&(nalu.getBitstream()), m_SEIs, nalu.m_nalUnitType, m_parameterSetManager.getActiveSPS(), m_pDecodedSEIOutputStream);
    delete m_prefixSEINALUs.front();
    m_prefixSEINALUs.pop_front();
  }
}

#if MCTS_ENC_CHECK
Void TDecTop::xAnalysePrefixSEImessages()
{
  if (m_tmctsCheckEnabled)
  {
    SEIMessages mctsSEIs = getSeisByType(m_SEIs, SEI::TEMP_MOTION_CONSTRAINED_TILE_SETS);
    for (SEIMessages::iterator it = mctsSEIs.begin(); it != mctsSEIs.end(); it++)
    {
      SEITempMotionConstrainedTileSets *mcts = (SEITempMotionConstrainedTileSets *)(*it);
      if (!mcts->m_each_tile_one_tile_set_flag)
      {
        printf("cannot (yet) check Temporal constrained MCTS if each_tile_one_tile_set_flag is not enabled\n");
        exit(1);
      }
      else
      {
        printf("MCTS check enabled!\n");
        m_conformanceCheck.enableTMctsCheck(true);
      }
    }
  }
}
#endif

#if MCTS_EXTRACTION
Bool TDecTop::xDecodeSlice(InputNALUnit &nalu, Int &iSkipFrame, Int iPOCLastDisplay, Bool bSkipCabacAndReconstruction)
#else
Bool TDecTop::xDecodeSlice(InputNALUnit &nalu, Int &iSkipFrame, Int iPOCLastDisplay, TDecCtx *pDecCtx)
#endif
{
  m_apcSlicePilot->initSlice(); // the slice pilot is an object to prepare for a new slice
                                // it is not associated with picture, sps or pps structures.

  if (m_bFirstSliceInPicture)
  {
    m_uiSliceIdx = 0;
  }
  else
  {
    m_apcSlicePilot->copySliceInfo(m_pcPic->getPicSym()->getSlice(m_uiSliceIdx - 1));
  }
  m_apcSlicePilot->setSliceIdx(m_uiSliceIdx);

  m_apcSlicePilot->setNalUnitType(nalu.m_nalUnitType);
  Bool nonReferenceFlag = (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_TRAIL_N ||
                           m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_TSA_N ||
                           m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_STSA_N ||
                           m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RADL_N ||
                           m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL_N);
  m_apcSlicePilot->setTemporalLayerNonReferenceFlag(nonReferenceFlag);
  m_apcSlicePilot->setReferenced(true); // Putting this as true ensures that picture is referenced the first time it is in an RPS
  m_apcSlicePilot->setTLayerInfo(nalu.m_temporalId);

#if ENC_DEC_TRACE
  const UInt64 originalSymbolCount = g_nSymbolCounter;
#endif

  // Step 1: Decode the header of the segment, if ok continue otherwise return false
  try {m_cEntropyDecoder.decodeSliceHeader(m_apcSlicePilot, &m_parameterSetManager, m_prevTid0POC);}
  catch (HeaderSyntaxException e)
  {
    std::cerr << e.what() << std::endl;
    return false;
  }
  // set POC for dependent slices in skipped pictures
  if (m_apcSlicePilot->getDependentSliceSegmentFlag() && m_prevSliceSkipped)
  {
    m_apcSlicePilot->setPOC(m_skippedPOC);
  }

  xUpdatePreviousTid0POC(m_apcSlicePilot);

  m_apcSlicePilot->setAssociatedIRAPPOC(m_pocCRA);
  m_apcSlicePilot->setAssociatedIRAPType(m_associatedIRAPType);

  // For inference of NoOutputOfPriorPicsFlag
  if (m_apcSlicePilot->getRapPicFlag())
  {
    if ((m_apcSlicePilot->getNalUnitType() >= NAL_UNIT_CODED_SLICE_BLA_W_LP && m_apcSlicePilot->getNalUnitType() <= NAL_UNIT_CODED_SLICE_IDR_N_LP) ||
        (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_CRA && m_bFirstSliceInSequence) ||
        (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_CRA && m_apcSlicePilot->getHandleCraAsBlaFlag()))
    {
      m_apcSlicePilot->setNoRaslOutputFlag(true);
    }
    // the inference for NoOutputPriorPicsFlag
    if (!m_bFirstSliceInBitstream && m_apcSlicePilot->getRapPicFlag() && m_apcSlicePilot->getNoRaslOutputFlag())
    {
      if (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_CRA)
      {
        m_apcSlicePilot->setNoOutputPriorPicsFlag(true);
      }
    }
    else
    {
      m_apcSlicePilot->setNoOutputPriorPicsFlag(false);
    }

    if (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_CRA)
    {
      m_craNoRaslOutputFlag = m_apcSlicePilot->getNoRaslOutputFlag();
    }
  }
  if (m_apcSlicePilot->getRapPicFlag() && m_apcSlicePilot->getNoOutputPriorPicsFlag())
  {
    m_lastPOCNoOutputPriorPics = m_apcSlicePilot->getPOC();
    m_isNoOutputPriorPics = true;
  }
  else
  {
    m_isNoOutputPriorPics = false;
  }

  // For inference of PicOutputFlag
  if (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL_N || m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL_R)
  {
    if (m_craNoRaslOutputFlag)
    {
      m_apcSlicePilot->setPicOutputFlag(false);
    }
  }

  if (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_CRA && m_craNoRaslOutputFlag) // Reset POC MSB when CRA has NoRaslOutputFlag equal to 1
  {
    TComPPS *pps = m_parameterSetManager.getPPS(m_apcSlicePilot->getPPSId());
    assert(pps != 0);
    TComSPS *sps = m_parameterSetManager.getSPS(pps->getSPSId());
    assert(sps != 0);
    Int iMaxPOClsb = 1 << sps->getBitsForPOC();
    m_apcSlicePilot->setPOC(m_apcSlicePilot->getPOC() & (iMaxPOClsb - 1));
    xUpdatePreviousTid0POC(m_apcSlicePilot);
  }

  // Skip pictures due to random access

  if (isRandomAccessSkipPicture(iSkipFrame, iPOCLastDisplay))
  {
    m_prevSliceSkipped = true;
    m_skippedPOC = m_apcSlicePilot->getPOC();
    return false;
  }
  // Skip TFD pictures associated with BLA/BLANT pictures
  if (isSkipPictureForBLA(iPOCLastDisplay))
  {
    m_prevSliceSkipped = true;
    m_skippedPOC = m_apcSlicePilot->getPOC();
    return false;
  }

  // clear previous slice skipped flag
  m_prevSliceSkipped = false;
  // we should only get a different poc for a new picture (with CTU address==0)
  if (!m_apcSlicePilot->getDependentSliceSegmentFlag() && m_apcSlicePilot->getPOC() != m_prevPOC && !m_bFirstSliceInSequence && (m_apcSlicePilot->getSliceCurStartCtuTsAddr() != 0))
  {
    printf("[TDecTop %d] Warning, the first slice of a picture might have been lost!\n",m_DecoderDescriptionId);
    // it already error at the next slice, so we need at least to complete the merging of last POC
    m_prevPOC = m_apcSlicePilot->getPOC();
    return true;
  }

  // exit when a new picture is found
  if (!m_apcSlicePilot->getDependentSliceSegmentFlag() && (m_apcSlicePilot->getSliceCurStartCtuTsAddr() == 0 && !m_bFirstSliceInPicture))
  {
    if (m_prevPOC >= m_pocRandomAccess)
    {
      m_prevPOC = m_apcSlicePilot->getPOC();
#if ENC_DEC_TRACE
      // rewind the trace counter since we didn't actually decode the slice
      g_nSymbolCounter = originalSymbolCount;
#endif
      return true;
    }
    m_prevPOC = m_apcSlicePilot->getPOC();
  }

  // detect lost reference picture and insert copy of earlier frame.
  // In this case the entire frame is lost definitively.
  {
    Int count=0;
    Int lostPoc=0;
    while ((lostPoc = m_apcSlicePilot->checkThatAllRefPicsAreAvailable(m_cListPic, m_apcSlicePilot->getRPS(), true, m_pocRandomAccess)) > 0)
    {
      xCreateLostPicture(lostPoc - 1,this->getDescriptionId(),this->m_cListPic);
      pDecCtx->LostPOC.push_back(lostPoc-1);
    }
  }
  if (!m_apcSlicePilot->getDependentSliceSegmentFlag())
  {
    m_prevPOC = m_apcSlicePilot->getPOC();
  }
  m_apcSlicePilot->applyReferencePictureSet(m_cListPic, m_apcSlicePilot->getRPS());
  if (!pDecCtx->LostPOC.empty()){
    pDecCtx->reRecodedMisMatchSlice=true;
    return false;
  }
  xActivateParameterSets();

  // actual decoding starts here
#if MCTS_EXTRACTION
  xActivateParameterSets(bSkipCabacAndReconstruction);
#else
#endif

  TComSlice *pcSlice = m_pcPic->getPicSym()->getSlice(m_uiSliceIdx);

  if (TDecConformanceCheck::doChecking())
  {
    m_conformanceCheck.checkSliceActivation(*pcSlice, nalu, *m_pcPic, m_bFirstSliceInBitstream, m_bFirstSliceInSequence, m_bFirstSliceInPicture);
  }

  m_bFirstSliceInSequence = false;
  m_bFirstSliceInBitstream = false;

  // When decoding the slice header, the stored start and end addresses were actually RS addresses, not TS addresses.
  // Now, having set up the maps, convert them to the correct form.
  pcSlice->setSliceSegmentCurStartCtuTsAddr(m_pcPic->getPicSym()->getCtuRsToTsAddrMap(pcSlice->getSliceSegmentCurStartCtuTsAddr()));
  pcSlice->setSliceSegmentCurEndCtuTsAddr(m_pcPic->getPicSym()->getCtuRsToTsAddrMap(pcSlice->getSliceSegmentCurEndCtuTsAddr()));
  if (!pcSlice->getDependentSliceSegmentFlag())
  {
    pcSlice->setSliceCurStartCtuTsAddr(m_pcPic->getPicSym()->getCtuRsToTsAddrMap(pcSlice->getSliceCurStartCtuTsAddr()));
    pcSlice->setSliceCurEndCtuTsAddr(m_pcPic->getPicSym()->getCtuRsToTsAddrMap(pcSlice->getSliceCurEndCtuTsAddr()));
  }

  m_pcPic->setTLayer(nalu.m_temporalId);

  if (!pcSlice->getDependentSliceSegmentFlag())
  {
    pcSlice->checkCRA(pcSlice->getRPS(), m_pocCRA, m_associatedIRAPType, m_cListPic);
#if MCTS_EXTRACTION
    if (bSkipCabacAndReconstruction)
    {
      m_bFirstSliceInPicture = false;
      m_uiSliceIdx++;
      return false;
    }
#endif
    // Set reference list
    pcSlice->setRefPicList(m_cListPic, true);

    // For generalized B
    // note: maybe not existed case (always L0 is copied to L1 if L1 is empty)
    if (pcSlice->isInterB() && pcSlice->getNumRefIdx(REF_PIC_LIST_1) == 0)
    {
      Int iNumRefIdx = pcSlice->getNumRefIdx(REF_PIC_LIST_0);
      pcSlice->setNumRefIdx(REF_PIC_LIST_1, iNumRefIdx);

      for (Int iRefIdx = 0; iRefIdx < iNumRefIdx; iRefIdx++)
      {
        pcSlice->setRefPic(pcSlice->getRefPic(REF_PIC_LIST_0, iRefIdx), REF_PIC_LIST_1, iRefIdx);
      }
    }
    if (!pcSlice->isIntra())
    {
      Bool bLowDelay = true;
      Int iCurrPOC = pcSlice->getPOC();
      Int iRefIdx = 0;

      for (iRefIdx = 0; iRefIdx < pcSlice->getNumRefIdx(REF_PIC_LIST_0) && bLowDelay; iRefIdx++)
      {
        if (pcSlice->getRefPic(REF_PIC_LIST_0, iRefIdx)->getPOC() > iCurrPOC)
        {
          bLowDelay = false;
        }
      }
      if (pcSlice->isInterB())
      {
        for (iRefIdx = 0; iRefIdx < pcSlice->getNumRefIdx(REF_PIC_LIST_1) && bLowDelay; iRefIdx++)
        {
          if (pcSlice->getRefPic(REF_PIC_LIST_1, iRefIdx)->getPOC() > iCurrPOC)
          {
            bLowDelay = false;
          }
        }
      }

      pcSlice->setCheckLDC(bLowDelay);
    }

    //---------------
    pcSlice->setRefPOCList();
  }

  m_pcPic->setCurrSliceIdx(m_uiSliceIdx);
  if (pcSlice->getSPS()->getScalingListFlag())
  {
    TComScalingList scalingList;
    if (pcSlice->getPPS()->getScalingListPresentFlag())
    {
      scalingList = pcSlice->getPPS()->getScalingList();
    }
    else if (pcSlice->getSPS()->getScalingListPresentFlag())
    {
      scalingList = pcSlice->getSPS()->getScalingList();
    }
    else
    {
      scalingList.setDefaultScalingList();
    }
    m_cTrQuant.setScalingListDec(scalingList);
    m_cTrQuant.setUseScalingList(true);
  }
  else
  {
    const Int maxLog2TrDynamicRange[MAX_NUM_CHANNEL_TYPE] =
        {
            pcSlice->getSPS()->getMaxLog2TrDynamicRange(CHANNEL_TYPE_LUMA),
            pcSlice->getSPS()->getMaxLog2TrDynamicRange(CHANNEL_TYPE_CHROMA)};
    m_cTrQuant.setFlatScalingList(maxLog2TrDynamicRange, pcSlice->getSPS()->getBitDepths());
    m_cTrQuant.setUseScalingList(false);
  }

  //  Decode a picture
  m_cGopDecoder.decompressSlice(&(nalu.getBitstream()), m_pcPic);

  m_bFirstSliceInPicture = false;
  m_uiSliceIdx++;

  return false;
}

Void TDecTop::xDecodeVPS(const std::vector<UChar> &naluData)
{
  TComVPS *vps = new TComVPS();

  m_cEntropyDecoder.decodeVPS(vps);
  m_parameterSetManager.storeVPS(vps, naluData);
}

Void TDecTop::xDecodeSPS(const std::vector<UChar> &naluData)
{
  TComSPS *sps = new TComSPS();
#if O0043_BEST_EFFORT_DECODING
  sps->setForceDecodeBitDepth(m_forceDecodeBitDepth);
#endif
  m_cEntropyDecoder.decodeSPS(sps);
  m_parameterSetManager.storeSPS(sps, naluData);
}

Void TDecTop::xDecodePPS(const std::vector<UChar> &naluData)
{
  TComPPS *pps = new TComPPS();
  m_cEntropyDecoder.decodePPS(pps);
  m_parameterSetManager.storePPS(pps, naluData);
}

#if MCTS_EXTRACTION
Bool TDecTop::decode(InputNALUnit &nalu, Int &iSkipFrame, Int &iPOCLastDisplay, Bool bSkipCabacAndReconstruction)
#else
Bool TDecTop::decode(InputNALUnit &nalu, Int &iSkipFrame, Int &iPOCLastDisplay,TDecCtx *pDecCtx)
#endif
{
  // ignore all NAL units of layers > 0
  if (nalu.m_nuhLayerId > 0)
  {
    fprintf(stderr, "Warning: found NAL unit with nuh_layer_id equal to %d. Ignoring.\n", nalu.m_nuhLayerId);
    return false;
  }
  // Initialize entropy decoder
  m_cEntropyDecoder.setEntropyDecoder(&m_cCavlcDecoder);
  m_cEntropyDecoder.setBitstream(&(nalu.getBitstream()));

  switch (nalu.m_nalUnitType)
  {
  case NAL_UNIT_VPS:
    xDecodeVPS(nalu.getBitstream().getFifo());
    return false;

  case NAL_UNIT_SPS:
    xDecodeSPS(nalu.getBitstream().getFifo());
    return false;

  case NAL_UNIT_PPS:
    xDecodePPS(nalu.getBitstream().getFifo());
    return false;

  case NAL_UNIT_PREFIX_SEI:
    // Buffer up prefix SEI messages until SPS of associated VCL is known.
    m_prefixSEINALUs.push_back(new InputNALUnit(nalu));
    return false;

  case NAL_UNIT_SUFFIX_SEI:
    if (m_pcPic)
    {
      m_seiReader.parseSEImessage(&(nalu.getBitstream()), m_pcPic->getSEIs(), nalu.m_nalUnitType, m_parameterSetManager.getActiveSPS(), m_pDecodedSEIOutputStream);
    }
    else
    {
      printf("Note: received suffix SEI but no picture currently active.\n");
    }
    return false;

  case NAL_UNIT_CODED_SLICE_TRAIL_R:
  case NAL_UNIT_CODED_SLICE_TRAIL_N:
  case NAL_UNIT_CODED_SLICE_TSA_R:
  case NAL_UNIT_CODED_SLICE_TSA_N:
  case NAL_UNIT_CODED_SLICE_STSA_R:
  case NAL_UNIT_CODED_SLICE_STSA_N:
  case NAL_UNIT_CODED_SLICE_BLA_W_LP:
  case NAL_UNIT_CODED_SLICE_BLA_W_RADL:
  case NAL_UNIT_CODED_SLICE_BLA_N_LP:
  case NAL_UNIT_CODED_SLICE_IDR_W_RADL:
  case NAL_UNIT_CODED_SLICE_IDR_N_LP:
  case NAL_UNIT_CODED_SLICE_CRA:
  case NAL_UNIT_CODED_SLICE_RADL_N:
  case NAL_UNIT_CODED_SLICE_RADL_R:
  case NAL_UNIT_CODED_SLICE_RASL_N:
  case NAL_UNIT_CODED_SLICE_RASL_R:
#if MCTS_EXTRACTION
    return xDecodeSlice(nalu, iSkipFrame, iPOCLastDisplay, bSkipCabacAndReconstruction);
#else
    return xDecodeSlice(nalu, iSkipFrame, iPOCLastDisplay, pDecCtx);
#endif
    break;

  case NAL_UNIT_EOS:
    m_associatedIRAPType = NAL_UNIT_INVALID;
    m_pocCRA = 0;
    m_pocRandomAccess = MAX_INT;
    m_prevPOC = MAX_INT;
    m_prevSliceSkipped = false;
    m_skippedPOC = 0;
    return false;

  case NAL_UNIT_ACCESS_UNIT_DELIMITER:
  {
    AUDReader audReader;
    UInt picType;
    try {audReader.parseAccessUnitDelimiter(&(nalu.getBitstream()), picType);}
    catch(...)
    {
      return true;
    }
    printf("Note: found NAL_UNIT_ACCESS_UNIT_DELIMITER\n");
    return false;
  }

  case NAL_UNIT_EOB:
    return false;

  case NAL_UNIT_FILLER_DATA:
  {
    FDReader fdReader;
    UInt size;
    fdReader.parseFillerData(&(nalu.getBitstream()), size);
    printf("Note: found NAL_UNIT_FILLER_DATA with %u bytes payload.\n", size);
    return false;
  }

  case NAL_UNIT_RESERVED_VCL_N10:
  case NAL_UNIT_RESERVED_VCL_R11:
  case NAL_UNIT_RESERVED_VCL_N12:
  case NAL_UNIT_RESERVED_VCL_R13:
  case NAL_UNIT_RESERVED_VCL_N14:
  case NAL_UNIT_RESERVED_VCL_R15:

  case NAL_UNIT_RESERVED_IRAP_VCL22:
  case NAL_UNIT_RESERVED_IRAP_VCL23:

  case NAL_UNIT_RESERVED_VCL24:
  case NAL_UNIT_RESERVED_VCL25:
  case NAL_UNIT_RESERVED_VCL26:
  case NAL_UNIT_RESERVED_VCL27:
  case NAL_UNIT_RESERVED_VCL28:
  case NAL_UNIT_RESERVED_VCL29:
  case NAL_UNIT_RESERVED_VCL30:
  case NAL_UNIT_RESERVED_VCL31:
    printf("Note: found reserved VCL NAL unit.\n");
    xParsePrefixSEIsForUnknownVCLNal();
    return false;

  case NAL_UNIT_RESERVED_NVCL41:
  case NAL_UNIT_RESERVED_NVCL42:
  case NAL_UNIT_RESERVED_NVCL43:
  case NAL_UNIT_RESERVED_NVCL44:
  case NAL_UNIT_RESERVED_NVCL45:
  case NAL_UNIT_RESERVED_NVCL46:
  case NAL_UNIT_RESERVED_NVCL47:
    printf("Note: found reserved NAL unit.\n");
    return false;
  case NAL_UNIT_UNSPECIFIED_48:
  case NAL_UNIT_UNSPECIFIED_49:
  case NAL_UNIT_UNSPECIFIED_50:
  case NAL_UNIT_UNSPECIFIED_51:
  case NAL_UNIT_UNSPECIFIED_52:
  case NAL_UNIT_UNSPECIFIED_53:
  case NAL_UNIT_UNSPECIFIED_54:
  case NAL_UNIT_UNSPECIFIED_55:
  case NAL_UNIT_UNSPECIFIED_56:
  case NAL_UNIT_UNSPECIFIED_57:
  case NAL_UNIT_UNSPECIFIED_58:
  case NAL_UNIT_UNSPECIFIED_59:
  case NAL_UNIT_UNSPECIFIED_60:
  case NAL_UNIT_UNSPECIFIED_61:
  case NAL_UNIT_UNSPECIFIED_62:
  case NAL_UNIT_UNSPECIFIED_63:
    printf("Note: found unspecified NAL unit.\n");
    return false;
  default:
    assert(0);
    break;
  }

  return false;
}

/** Function for checking if picture should be skipped because of association with a previous BLA picture
 * \param iPOCLastDisplay POC of last picture displayed
 * \returns true if the picture should be skipped
 * This function skips all TFD pictures that follow a BLA picture
 * in decoding order and precede it in output order.
 */
Bool TDecTop::isSkipPictureForBLA(Int &iPOCLastDisplay)
{
  if ((m_associatedIRAPType == NAL_UNIT_CODED_SLICE_BLA_N_LP || m_associatedIRAPType == NAL_UNIT_CODED_SLICE_BLA_W_LP || m_associatedIRAPType == NAL_UNIT_CODED_SLICE_BLA_W_RADL) &&
      m_apcSlicePilot->getPOC() < m_pocCRA && (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL_R || m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL_N))
  {
    iPOCLastDisplay++;
    return true;
  }
  return false;
}

/** Function for checking if picture should be skipped because of random access
 * \param iSkipFrame skip frame counter
 * \param iPOCLastDisplay POC of last picture displayed
 * \returns true if the picture shold be skipped in the random access.
 * This function checks the skipping of pictures in the case of -s option random access.
 * All pictures prior to the random access point indicated by the counter iSkipFrame are skipped.
 * It also checks the type of Nal unit type at the random access point.
 * If the random access point is CRA/CRANT/BLA/BLANT, TFD pictures with POC less than the POC of the random access point are skipped.
 * If the random access point is IDR all pictures after the random access point are decoded.
 * If the random access point is none of the above, a warning is issues, and decoding of pictures with POC
 * equal to or greater than the random access point POC is attempted. For non IDR/CRA/BLA random
 * access point there is no guarantee that the decoder will not crash.
 */
Bool TDecTop::isRandomAccessSkipPicture(Int &iSkipFrame, Int &iPOCLastDisplay)
{
  if (iSkipFrame)
  {
    iSkipFrame--; // decrement the counter
    return true;
  }
  else if (m_pocRandomAccess == MAX_INT) // start of random access point, m_pocRandomAccess has not been set yet.
  {
    if (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_CRA || m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_BLA_W_LP || m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_BLA_N_LP || m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_BLA_W_RADL)
    {
      // set the POC random access since we need to skip the reordered pictures in the case of CRA/CRANT/BLA/BLANT.
      m_pocRandomAccess = m_apcSlicePilot->getPOC();
    }
    else if (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_IDR_W_RADL || m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_IDR_N_LP)
    {
      m_pocRandomAccess = -MAX_INT; // no need to skip the reordered pictures in IDR, they are decodable.
    }
    else
    {
      if (!m_warningMessageSkipPicture)
      {
        printf("\nWarning: this is not a valid random access point and the data is discarded until the first CRA picture");
        m_warningMessageSkipPicture = true;
      }
      return true;
    }
  }
  // skip the reordered pictures, if necessary
  else if (m_apcSlicePilot->getPOC() < m_pocRandomAccess && (m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL_R || m_apcSlicePilot->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL_N))
  {
    iPOCLastDisplay++;
    return true;
  }
  // if we reach here, then the picture is not skipped.
  return false;
}

//! \}
