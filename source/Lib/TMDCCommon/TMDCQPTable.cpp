#include "TMDCQPTable.hpp"
#include <stdexcept>
#include <cstring>
#define BUFFER_READ_SIZE 1024
TMDCQPTable* TMDCQPTable::m_instance = NULL;
Void TMDCQPTable::init(Int nbElementReadLine,const char * QPFileName1,
const char *QPFileName2,const char * QtreeFileName){
    qpArray = new Int [nbElementReadLine];
    qtreeArray = new Int[nbElementReadLine];
    memset(qpArray, 0, nbElementReadLine * sizeof(Int));
    memset(qtreeArray, 0, nbElementReadLine * sizeof(Int));
    m_filename[QTREE]=QtreeFileName;
    m_filename[DESCRIPTION1] = QPFileName1;
    m_filename[DESCRIPTION2] = QPFileName2;
}

Void TMDCQPTable::openFile(FileType description, std::ios::openmode openmode){
    m_ios[description].open(m_filename[description],openmode);
    if (m_ios[description].fail()){
        std::cerr << "Cannot open file : " << m_filename[description] <<std::endl;
        throw std::runtime_error("");
    }
}
Void TMDCQPTable::closeFile(FileType description){
    m_ios[description].close();
}

Int TMDCQPTable::convertStringToIntArrayQP(Int *bufferDest, char *str,int nbElement){
    int j = 0;
    for (int i = 0; i < nbElement; i++) {
        bufferDest[i] = 0;
    }
    for (int i = 0; i < nbElement&&str[i]!='\n'&&str[i]!='\0';i++){

        if (str[i] == ','){
            j++;
        }
        else
        {
            bufferDest[j] = bufferDest[j]*10 + (str[i] - 48);
        }
    }
    return j;
}

// TODO: See if there's a better way to organize this file
Int TMDCQPTable::convertStringtoIntArrayQtree(Int *bufferDest, char *str,int nbElement){
    int j = -1;
    bool skip = false;
    for (int i = 0; i < nbElement; i++) {
        bufferDest[i] = 0;
    }
    for (int i = 0; i < nbElement&&str[i]!='\n'&&str[i]!='\0';i++){
        //special character skip
        if (str[i]=='<'){
            skip = true;
        }
        if (str[i]=='>')
         {   //skip first space
            i++;
            skip = false;
            }
        if (skip)
        {
            continue;
        }
        if (str[i] == ' '){
            j++;
        }
        else
        {
            bufferDest[j] = bufferDest[j]*10 + (str[i] - 48);
        }
    }
    return j;
}
Int TMDCQPTable::seekLine(FileType description, Int line){
    if (m_ios[description].is_open()){
        m_ios[description].seekg(0, std::ios::beg);
        for (int i = 0; i < line; i++)
        {
            m_ios[description].ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
    else throw std::runtime_error("seekLine: Unexpected closed file");
    return 0;
}
Int TMDCQPTable::readALine(FileType description){
    char* buffer = new char[BUFFER_READ_SIZE];
    Int count = 0;
    // get the buffer from the file
    if (m_ios[description].is_open()) m_ios[description].getline(buffer, BUFFER_READ_SIZE);
    else throw std::runtime_error("readALineQp: Unexpected closed file");
    //parsing this buffer
    m_counters[description]++;

    if (description==QTREE){
        m_counter_rs++;
        count = convertStringtoIntArrayQtree(qtreeArray,buffer, BUFFER_READ_SIZE);
    }else{
        count =  convertStringToIntArrayQP(qpArray,buffer,BUFFER_READ_SIZE);
    }
    delete[] buffer;
    return count;
}
void TMDCQPTable::writeOutCUInfo   ( TComDataCU* pcCU )
{
  if (m_ios[QTREE].is_open()){
  Int iPoc = pcCU->getSlice()->getPOC();
  Int iAddr = pcCU->getCtuRsAddr();
  Int iTotalNumPart = pcCU->getTotalNumPart();
  m_ios[QTREE] << "<" << iPoc << "," << iAddr << ">" << " ";  ///< Write out CU & PU splitting info
  xWriteOutCUInfo  ( pcCU, iTotalNumPart, 0, 0 );   ///< Recursive write Prediction, CU, PU, Merge, Intra, ME
  m_ios[QTREE] << std::endl;}
  else throw std::runtime_error("writeOutCUInfo: Unexpected closed file");
}


void TMDCQPTable::xWriteOutCUInfo  ( TComDataCU* pcCU, Int iLength, Int iOffset, UInt iDepth )
{
  
  UChar* puhDepth    = pcCU->getDepth();
  SChar * puhPartSize = pcCU->getPartitionSize();
  UChar * puhCbf = pcCU->getCbf(COMPONENT_Y);
  SChar* puhQP = pcCU->getQP();
  TComMv rcMV;

  if( puhDepth[iOffset] <= iDepth )
  {
    ///< CU PU info
    m_ios[QTREE] << (Int)(puhPartSize[iOffset]) << " "; 
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

      } /// PU end
  }
  else
  {
    m_ios[QTREE] << "99" << " ";     ///< CU info
    for( UInt i = 0; i < 4; i++ )
    {
      xWriteOutCUInfo  ( pcCU, iLength/4, iOffset+iLength/4*i, iDepth+1);
    }
  }
}


TMDCQPTable::~TMDCQPTable(){
    delete [] qpArray;
    delete [] qtreeArray;
    for (auto &i:m_ios){
        i.close();
    }
}