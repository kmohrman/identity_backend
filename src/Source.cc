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
      std::cout << "----> Fed " << fedId << " -- " << fedSize << " -- " << std::endl;
      is.read(reinterpret_cast<char *>(rawData.data()), fedSize);
    }
    return rawCollection;
  }

  FEDRawDataCollection readRawBuff(void* input_buffer, unsigned int nfeds) {
    FEDRawDataCollection rawCollection;
    unsigned iter = 0; 
    for (unsigned int ifed = 0; ifed < nfeds; ++ifed) {
      unsigned int fedId   = (unsigned int) input_buffer[iter];
      unsigned int fedSize = (unsigned int) input_buffer[iter+1];
      FEDRawData &rawData = rawCollection.FEDData(fedId);
      rawData.resize(fedSize);
      std::memcpy(rawData.data(),input_buffer[iter+2],fedSize);
      iter += fedSize;
    }
    return rawCollection;
  }

}  // namespace

namespace edm {
  Source::Source(int maxEvents, ProductRegistry &reg, std::string const &datadir)
    : maxEvents_(maxEvents), numEvents_(0), iterEvents_(1),fBase_(0), rawToken_(reg.produces<FEDRawDataCollection>()) {
    std::cout << "---> source " << datadir << std::endl;
    std::ifstream in_raw((datadir + "/raw.bin").c_str(), std::ios::binary);

    unsigned int nfeds;
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    std::cout << "---> N feds " << nfeds << " -- " << datadir << std::endl;
    while (not in_raw.eof()) {
      in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      raw_.emplace_back(readRaw(in_raw, nfeds));

      // next event
      in_raw.exceptions(std::ifstream::badbit);
      in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    }
    fNFeds = nfeds;
    if (maxEvents_ < 0) {
      maxEvents_ = raw_.size();
    }
  }

  std::shared_ptr<Event> Source::produce(int streamId, ProductRegistry const &reg) {
    const int old = numEvents_.fetch_add(1);
    const int iev = old + 1 + fBase_;
    std::cout << "Calling Source : " << old << " -- " << iev << std::endl;
    if (old >= iterEvents_) {
      fBase_.fetch_add(1);
      numEvents_ = 0;//fetch_add(1)
      return nullptr;
    }
    lastEvent_ = std::make_unique<Event>(streamId, iev, reg);
    const int index = iev % raw_.size();
    lastEvent_->emplace(rawToken_, raw_[index]);
    return lastEvent_;
  }

  void  Source::fill(void* input_buffer)  {  
    raw_.pop_back();
    raw_.emplace_back(readRawBuff(input_buffer,fNFeds));
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
