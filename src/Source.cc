#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "Source.h"

namespace {
  FEDRawDataCollection readRaw(std::ifstream &is, unsigned int nfeds) {
    FEDRawDataCollection rawCollection;
    for (unsigned int ifed = 0; ifed < nfeds; ++ifed) {
      unsigned int fedId;
      is.read(reinterpret_cast<char *>(&fedId), sizeof(unsigned int));
      unsigned int fedSize;
      is.read(reinterpret_cast<char *>(&fedSize), sizeof(unsigned int));
      FEDRawData &rawData = rawCollection.FEDData(fedId);
      rawData.resize(fedSize);
      is.read(reinterpret_cast<char *>(rawData.data()), fedSize);
    }
    return rawCollection;
  }

  FEDRawDataCollection readRawBuff(const void* input_buffer) { //, unsigned int nfeds) {
    FEDRawDataCollection rawCollection;
    unsigned iter = 0; 
    const uint32_t * test_buffer = reinterpret_cast<const uint32_t *>(input_buffer);
    unsigned int nfeds = test_buffer[iter];
    iter++;
    nfeds = 108;
    /*
    unsigned int feds[nfeds];
    unsigned pId = 0; 
    for(unsigned int i0 = 0; i0 < 138; i0++) { 
      if(i0 != 10  && i0 != 11  && i0 != 22  && i0 != 23  && i0 != 34  && i0 != 35  && i0 != 46  && i0 != 47  && i0 != 58  && i0 != 59  &&
	 i0 != 70  && i0 != 71  && i0 != 82  && i0 != 83  && i0 != 94  && i0 != 95  && i0 != 103 && i0 != 104 && i0 != 105 && i0 != 106 &&
	 i0 != 107 && i0 != 115 && i0 != 116 && i0 != 117 && i0 != 118 && i0 != 119 && i0 != 127 && i0 != 128 && i0 != 129 && i0 != 130 && i0 != 131) {feds[pId]=1200+i0; pId++;}
    }
    */
    for (unsigned int ifed = 0; ifed < nfeds; ++ifed) {
      unsigned int fedId   = (unsigned int) test_buffer[iter]; iter++;
      unsigned int fedSize = (unsigned int) test_buffer[iter]; iter++;
      fedId = 1200+ifed;
      fedSize = 20;
      FEDRawData &rawData = rawCollection.FEDData(fedId);
      rawData.resize(fedSize*4);
      //std::memcpy(rawData.data(),&(test_buffer[iter]),fedSize*4);
      iter += fedSize;
    }
    return rawCollection;
  }

}  // namespace

namespace edm {
  Source::Source(int maxEvents, ProductRegistry &reg, std::string const &datadir)
    : maxEvents_(maxEvents), numEvents_(0), iterEvents_(1),fBase_(0), rawToken_(reg.produces<FEDRawDataCollection>()) {
    std::ifstream in_raw((datadir + "/raw.bin").c_str(), std::ios::binary);

    unsigned int nfeds;
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    /*
    while (not in_raw.eof()) {
      in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      raw_.emplace_back(readRaw(in_raw, nfeds));

      // next event
      in_raw.exceptions(std::ifstream::badbit);
      in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    }
    */
    fNFeds = nfeds;
    //fNFeds = 1;
    if (maxEvents_ < 0) {
      maxEvents_ = raw_.size();
    }
  }

  std::shared_ptr<Event> Source::produce(int streamId, ProductRegistry const &reg) {
    const int old = numEvents_.fetch_add(1);
    if (old >= int(raw_.size())) {
      //fBase_.fetch_add(1);
      numEvents_ = 0;//fetch_add(1)
      return nullptr;
    }
    lastEvent_ = std::make_unique<Event>(streamId, old, reg);
    const int index = old;// % raw_.size();
    lastEvent_->emplace(rawToken_, raw_[index]);
    return lastEvent_;
  }

  void  Source::fill(const void* input_buffer,bool iClear)  {
    if(raw_.size() > 0 && iClear) raw_.clear();
    raw_.emplace_back(readRawBuff(input_buffer));
  }
  /*
  void Source::fill_fromstream(int streamId, ProductRegistry const &reg,char* iRaw) {
    FEDRawDataCollection rawCollection;
    unsigned int pId = 0;
    for (unsigned int ifed = 0; ifed < fNFeds; ++ifed) {
      std::stringstream strValue;
      strValue << iRaw[pId]; pId++;
      unsigned int fedId;   strValue >> fedId;
      strValue << iRaw[pId]; pId++;
      unsigned int fedSize; strValue >> fedSize;
      //FEDRawData &rawData = rawCollection.FEDData(fedId);
      //rawData.data()      = &(iRaw[pId]);
      pId+=fedSize;
    }
    raw_.emplace_back(rawCollection);
  }
  */
}  // namespace edm
