#pragma once
#include <fstream>
#include <string>
#include <vector>


#include "TLibCommon/TComDataCU.h"
#include "TLibCommon/TComSlice.h"
#include "TLibEncoder/TEncCu.h"



class TSysuAnalyzerOutput
{
public:
  TSysuAnalyzerOutput(void);
  void writeNewLineQP();
  void writeNewLineDqp();
  void writeQP(Int QP, const char* delimiter);
  void writeDqp(Int dQP, const char *delimiter);
  TSysuAnalyzerOutput(const char* cupu_name);
  TSysuAnalyzerOutput(const char* cupu_name, const char* qpDecFile);
  ~TSysuAnalyzerOutput(void);

#if 0
  /// write out residual
  Void writeOutResiYUV (TDecCu* pcDecCU, TComDataCU* pcCU, UInt uiZorderIdx, UInt uiDepth);
#endif
 

  /// splitting mode
  Void writeOutCUInfo   ( TComDataCU* pcCU );
  //void writeOutResidual(TComDataCU* pcCU);
  Void xWriteOutCUInfo  ( TComDataCU* pcCU, Int iLength, Int iOffset, UInt iDepth);
  //Void xWriteOutTUInfo  ( TComDataCU* pcCU, Int iLength, Int iOffset, UInt iDepth);
  Void writeResidual   (TComDataCU* pcCU);
  
  /// Sequence parameter set output
  //Void writeOutSps         ( TComSPS* pcSPS );


  ///write out tile info
  Void writeOutTileInfo(TComPic * pcPic);



  std::vector<int> aiCUBits;
  /*
  Void setAllDepthTo       ( TComDataCU* pcCU, UInt uiDepth );
  Bool compareSplitMode    ( TComDataCU* pcRecursive, TComDataCU* pcFast );

  Void printCUModeForLCU   ( TComDataCU* pcCU, Char* phMessage );
  Void xPrintCUModeForLCU  ( UChar* pcCU, Int iLength, UInt iDepth );
  */



  /// SINGLETON 
  static TSysuAnalyzerOutput* getInstance() { return m_instance;}
  static TSysuAnalyzerOutput* initInstanceEncoder() { if (m_instance == NULL) m_instance = new TSysuAnalyzerOutput(); return m_instance; }
  static TSysuAnalyzerOutput* initInstanceEncoder(const char *cupu_file) { if (m_instance == NULL) m_instance = new TSysuAnalyzerOutput(cupu_file); return m_instance; }
  static TSysuAnalyzerOutput* initInstanceDecoder(const char* cupu_file,const char *qpDecFile) { if (m_instance == NULL) m_instance = new TSysuAnalyzerOutput(cupu_file,qpDecFile); return m_instance; }

private:

  /// Decoder output ( extracted from bitstream )
  //std::ofstream m_cSpsOut;          ///< SPS info
  //std::ofstream m_cPredOutput;      ///< Prediction mode info output
  std::ofstream m_cCUPUOutput;      ///< CU info output
  std::ofstream m_QPFile;
  std::ofstream m_CbfOutput;
  std::ofstream m_QPFile2;
  //std::ofstream m_cMVOutput;        ///< MV info output
  //std::ofstream m_cMergeOutput;     ///< Merge info output
  //std::ofstream m_cIntraOutput;     ///< Intra info output
  //std::ofstream m_cTUOutput;        ///< TU info output
  //std::ofstream m_cBitOutputLCU;    ///< LCU bit consumption info output
  //std::ofstream m_cBitOutputSCU;    ///< SCU bit consumption info output
  //std::ofstream m_cTileOutPut;
  //std::ofstream m_cCUOutputResi;
  /// Encoder output ( extracted in the encoding process
  //std::ofstream m_cMEOutput;    ///< ME info (search point number, SAD, cost, etc)

  static TSysuAnalyzerOutput* m_instance;

};
