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
}  // namespace

namespace edm {
  Source::Source(int maxEvents, ProductRegistry &reg, std::string const &datadir)
    : maxEvents_(maxEvents), numEvents_(0), fCount_(0) , iterEvents_(1),fBase_(0), rawToken_(reg.produces<FEDRawDataCollection>()) {
    std::ifstream in_raw((datadir + "/raw.bin").c_str(), std::ios::binary);

    unsigned int nfeds;
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    while (not in_raw.eof()) {
      in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      raw_.emplace_back(readRaw(in_raw, nfeds));
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
    const int count = fCount_.fetch_sub(1);
    if(count < 1) {
      fCount_=0;
      return nullptr;
    }
    if (old >= int(raw_.size())) {
      numEvents_ = 0;
    }
    lastEvent_ = std::make_unique<Event>(streamId, old, reg);
    const int index = old  % raw_.size();
    lastEvent_->emplace(rawToken_, raw_[index]);
    return lastEvent_;
  }

  void  Source::fill(const void* input_buffer,bool iClear)  {
    fCount_.fetch_add(1);
  }
}  // namespace edm
