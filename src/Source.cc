#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "Source.h"

namespace {
  //FEDRawDataCollection
  std::pair<FEDRawDataCollection,BeamSpotPOD> readRawBuff(const void* input_buffer) { //, unsigned int nfeds) {
    BeamSpotPOD bs;
    FEDRawDataCollection rawCollection;
    unsigned iter = 0; 
    const uint32_t * test_buffer = reinterpret_cast<const uint32_t *>(input_buffer);
    unsigned int pBSSize = 11;
    std::memcpy(&bs,&(test_buffer[iter]),sizeof(float)*pBSSize); iter+=pBSSize;
    unsigned int nfeds = test_buffer[iter]; iter++;
    for (unsigned int ifed = 0; ifed < nfeds; ++ifed) {
      unsigned int fedId   = (unsigned int) test_buffer[iter]; iter++;
      unsigned int fedSize = (unsigned int) test_buffer[iter]; iter++;
      FEDRawData &rawData = rawCollection.FEDData(fedId);
      rawData.resize(fedSize*4);
      std::memcpy(rawData.data(),&(test_buffer[iter]),fedSize*4);
      iter += fedSize;
    }
    return std::pair<FEDRawDataCollection,BeamSpotPOD>(rawCollection,bs);
  }

}  // namespace

namespace edm {
  Source::Source(int maxEvents, ProductRegistry &reg, std::string const &datadir)
    : maxEvents_(maxEvents), numEvents_(0), iterEvents_(1),fBase_(0),
      rawToken_(reg.produces<FEDRawDataCollection>()),
      beamSpotPODToken_(reg.produces<BeamSpotPOD>()) {
    std::ifstream in_raw((datadir + "/raw.bin").c_str(), std::ios::binary);

    unsigned int nfeds;
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
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
    lastEvent_->emplace(rawToken_, raw_[index].first);
    lastEvent_->emplace(beamSpotPODToken_, raw_[index].second);
    return lastEvent_;
  }

  void  Source::fill(const void* input_buffer,bool iClear)  {
    if(raw_.size() > 0 && iClear) raw_.clear();
    raw_.emplace_back(readRawBuff(input_buffer));
  }
}  // namespace edm
