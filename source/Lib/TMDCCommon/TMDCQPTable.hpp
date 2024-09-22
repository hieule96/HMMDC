#ifndef _TMDCQPTable_
#define _TMDCQPTable_
#pragma once

#include <string>
#include <fstream>
#include <array>
#include <iostream>
#include <TLibCommon/CommonDef.h>
#include <TLibCommon/TComDataCU.h>
class TMDCQPTable{

    public:
        TMDCQPTable():qpArray(NULL),qtreeArray(NULL),m_counters({0,0,0}),m_counter_rs(0){};
        void init(Int nbElementReadLine,const char * QPFileName1,const char *QPFileName2,const char * QtreeFileName);
        Int readALine(FileType description);
        Int* getQPArray(){return qpArray;}
        Int* getQtreeArray(){return qtreeArray;}
        Int convertStringToIntArrayQP(Int *bufferDest, char *str,int nbElement);
        Int convertStringtoIntArrayQtree(Int *bufferDest, char *str,int nbElement);
        Int getCount(FileType description) {return m_counter_rs;};
        Void resetCount() {
            m_counter_rs = 0;
            m_counters[DESCRIPTION1] = 0;
            m_counters[DESCRIPTION2] = 0;
            m_ios[QTREE].seekg(ios::beg);
            m_ios[DESCRIPTION1].seekg(ios::beg);
            m_ios[DESCRIPTION2].seekg(ios::beg);

        }
        ~TMDCQPTable();
        static TMDCQPTable* getInstance() {return m_instance;}
        static TMDCQPTable* initInstance(Int nbElementReadLine,const char* QPFileName1,const char* QPFileName2, const char* QtreeFileName){
            if (m_instance==NULL){
                m_instance = new TMDCQPTable();
                m_instance->init(nbElementReadLine,QPFileName1,QPFileName2,QtreeFileName);
            }
            return m_instance;
        }
        Void closeFile(FileType description);
        Void incrementCounterRs(){m_counter_rs++;};
        Void openFile(FileType description, std::ios::openmode openmode);
        Void writeOutCUInfo   ( TComDataCU* pcCU );
        Int  seekLine(FileType description, Int line);
    private:
        Void xWriteOutCUInfo  ( TComDataCU* pcCU, Int iLength, Int iOffset, UInt iDepth );
        Int* qpArray;
        Int* qtreeArray;
        std::array<const char *,3> m_filename;
        std::array<Int,3> m_counters;
        Int m_counter_rs;
        std::array<std::fstream,3> m_ios;
        static TMDCQPTable* m_instance;
};


#endif
