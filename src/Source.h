#ifndef Source_h
#define Source_h

#include <atomic>
#include <filesystem>
#include <string>
#include <memory>

//#include "Framework/Event.h"
//#include "DataFormats/FEDRawDataCollection.h"

namespace edm {
  class Source {
  public:
    explicit Source(int maxEvents, ProductRegistry& reg, std::string const& datadir);

    int maxEvents() const { return maxEvents_; }
    // thread safe
    std::shared_ptr<Event> produce(int streamId, ProductRegistry const& reg);
    void  fill(const void* input_buffer,bool iClear);
    void clear();
    std::vector<std::shared_ptr<Event>> lastEvents_;

  private:
    int maxEvents_;
    std::atomic<int> numEvents_;
    std::atomic<int> fCount_;
    int iterEvents_;
    std::atomic<int> fBase_;
    unsigned int fNFeds;
    EDPutTokenT<FEDRawDataCollection> const rawToken_;
    std::vector<FEDRawDataCollection> raw_;
    //EDPutTokenT<BeamSpotPOD> beamSpotPODToken_;
    //std::vector<std::pair<FEDRawDataCollection,BeamSpotPOD> > raw_;
  };
}  // namespace edm

#endif
