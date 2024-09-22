#include "TSysuAnalyzerOutput.h"
#include "Utilities/TVideoIOYuv.h"
TSysuAnalyzerOutput* TSysuAnalyzerOutput::m_instance = NULL;

TSysuAnalyzerOutput::TSysuAnalyzerOutput()
{
  const Int bitdepth[3] = {8,8,8};
  m_cCUPUOutput.open ("decoder_cupu.txt", ios::out);
#if ( HM_VERSION > 40)
  m_cTileOutPut.open("decoder_tile.txt", ios::out);
#endif


}

void TSysuAnalyzerOutput::writeNewLineQP() {
    m_QPFile << std::endl;
}


void TSysuAnalyzerOutput::writeQP(Int QP, const char *delimiter) {
    m_QPFile << QP << delimiter;
}


void TSysuAnalyzerOutput::writeDqp(Int dQP, const char *delimiter){
    m_CbfOutput<<dQP<<delimiter;
}

void TSysuAnalyzerOutput::writeNewLineDqp() {
    m_CbfOutput << std::endl;
}

TSysuAnalyzerOutput::TSysuAnalyzerOutput(const char* cupu_name)
{

    m_cCUPUOutput.open(cupu_name, ios::out);
#if ( HM_VERSION > 40)
    m_cTileOutPut.open("decoder_tile.txt", ios::out);
#endif


}
Void TSysuAnalyzerOutput::writeResidual   (TComDataCU* pcCU){
  
}

TSysuAnalyzerOutput::TSysuAnalyzerOutput(const char* cupu_name, const char* qpDecFile)
{
    char buffer[20];
    memset(buffer,0,20*sizeof(char));
    m_cCUPUOutput.open(cupu_name, ios::out);
}

#if ( HM_VERSION > 40)
//write out tile info
void TSysuAnalyzerOutput::writeOutTileInfo(TComPic * pcPic)
{
    int iPoc = pcPic ->getPOC();
    int iTileNumRows =  pcPic ->getPicSym() ->getNumRowsMinus1()+1;
    int iTileNumCols = pcPic ->getPicSym() ->getNumColumnsMinus1()+1;


    for( int uiRowIdx=0; uiRowIdx < iTileNumRows; uiRowIdx++ )
           for( int uiColumnIdx=0; uiColumnIdx <iTileNumCols ; uiColumnIdx++ )
           {
             int uiTileIdx = uiRowIdx * (iTileNumCols) + uiColumnIdx;

             //information for each tile
             int uiTileWidth = 0;
             int uiTileHeight = 0;
             int uiFirstCUAddr = 0;

             uiTileWidth = pcPic ->getPicSym()->getTComTile(uiTileIdx)->getTileWidth();

             uiTileHeight = pcPic ->getPicSym()->getTComTile(uiTileIdx)->getTileHeight();

             uiFirstCUAddr = pcPic ->getPicSym()->getTComTile(uiTileIdx) ->getFirstCUAddr();

             m_cTileOutPut << "<" << iPoc << "," << (iTileNumCols )* (iTileNumRows )<< ">"
                  << " " << uiFirstCUAddr << " " << uiTileWidth << " " << uiTileHeight << endl;


           }
}

#endif




void TSysuAnalyzerOutput::writeOutCUInfo   ( TComDataCU* pcCU )
{
  Int iPoc = pcCU->getSlice()->getPOC();
  Int iAddr = pcCU->getCtuRsAddr();
  Int iTotalNumPart = pcCU->getTotalNumPart();
  m_cCUPUOutput << "<" << iPoc << "," << iAddr << ">" << " ";  ///< Write out CU & PU splitting info
  xWriteOutCUInfo  ( pcCU, iTotalNumPart, 0, 0 );   ///< Recursive write Prediction, CU, PU, Merge, Intra, ME
  m_cCUPUOutput << endl;

}


void TSysuAnalyzerOutput::xWriteOutCUInfo  ( TComDataCU* pcCU, Int iLength, Int iOffset, UInt iDepth )
{
  
  UChar* puhDepth    = pcCU->getDepth();
  SChar * puhPartSize = pcCU->getPartitionSize();
  UChar * puhCbf = pcCU->getCbf(COMPONENT_Y);
  SChar* puhQP = pcCU->getQP();
  TComMv rcMV;

  if( puhDepth[iOffset] <= iDepth )
  {
    ///< CU PU info
    m_cCUPUOutput << (Int)(puhPartSize[iOffset]) << " "; 
    ///< TU info
    //xWriteOutTUInfo  ( pcCU, iLength, iOffset, 0 );   ///< Recursive write TU
    /// PU number in this leaf CU
    int iNumPart = 0;           

    switch ( puhPartSize[iOffset] )
    {
    case SIZE_2Nx2N:    iNumPart = 1; break;
    case SIZE_2NxN:     iNumPart = 2; break;
    case SIZE_Nx2N:     iNumPart = 2; break;
    case SIZE_NxN:      iNumPart = 4; break;
    case SIZE_2NxnU:    iNumPart = 2; break;
    case SIZE_2NxnD:    iNumPart = 2; break;
    case SIZE_nLx2N:    iNumPart = 2; break;
    case SIZE_nRx2N:    iNumPart = 2; break;
    default:    iNumPart = 0;  /*assert(0);*/  break;  ///< out of boundery
    }

    /// Traverse every PU
    int iPartAddOffset = 0;   ///< PU offset
    for( int i = 0; i < iNumPart; i++ )
    {

      switch ( puhPartSize[iOffset] )
      {
      case SIZE_2NxN:
        iPartAddOffset = ( i == 0 )? 0 : iLength >> 1;
        break;
      case SIZE_Nx2N:
        iPartAddOffset = ( i == 0 )? 0 : iLength >> 2;
        break;
      case SIZE_NxN:
        iPartAddOffset = ( iLength >> 2 ) * i;
        break;
      case SIZE_2NxnU:    
        iPartAddOffset = ( i == 0 ) ? 0 : iLength >> 3;
        break;
      case SIZE_2NxnD:    
        iPartAddOffset = ( i == 0 ) ? 0 : (iLength >> 1) + (iLength >> 3);
        break;
      case SIZE_nLx2N:    
        iPartAddOffset = ( i == 0 ) ? 0 : iLength >> 4;
        break;
      case SIZE_nRx2N:   
        iPartAddOffset = ( i == 0 ) ? 0 : (iLength >> 2) + (iLength >> 4);
        break;
      default:
        assert ( puhPartSize[iOffset] == SIZE_2Nx2N );
        iPartAddOffset = 0;
        break;
      }

      m_CbfOutput << (Int) puhCbf[iOffset] << " ";
      m_QPFile2 << (Int) puhQP[iOffset] <<",";
    } /// PU end
  }
  else
  {
    m_cCUPUOutput << "99" << " ";     ///< CU info
    for( UInt i = 0; i < 4; i++ )
    {
      xWriteOutCUInfo  ( pcCU, iLength/4, iOffset+iLength/4*i, iDepth+1);
    }
  }
}




TSysuAnalyzerOutput::~TSysuAnalyzerOutput()
{
  m_cCUPUOutput.close();
}
