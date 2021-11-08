#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/host_noncached_unique_ptr.h"
#include "CUDADataFormats/BeamSpotCUDA.h"
#include "Framework/ProductRegistry.h"
#include "DataFormats/FEDRawDataCollection.h"

#include "Framework/ESPluginFactory.h"
#include "Framework/PluginFactory.h"
#include "Framework/EventSetup.h"
#include "DataFormats/BeamSpotPOD.h"
#include "Source.h"
#include "Source.cc"
#include "StreamSchedule.h"
#include "StreamSchedule.cc"
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

class BSTest  {
public:
  explicit BSTest(std::string const& datadir) : data_(datadir) {
    fEvent = std::make_unique<edm::Event>(0, 0, reg_);
  }
  void setItAll(unsigned int iId,std::vector<std::string> const& esproducers,std::vector<std::string> runs);  
  const void** getOutput();
  void fillSource(const void* input_buffer,bool iClear);
  uint64_t* getSize();
  using OutputStorage = HostProduct<int8_t[]>;
  using SizeStorage   = HostProduct<uint64_t[]>;

private:
  std::string data_;
  edm::EventSetup fSetup;
  std::unique_ptr<edm::Event>      fEvent;
  std::vector<edm::StreamSchedule> fStream;
  edm::ProductRegistry reg_;
  edmplugin::PluginManager fPluginManager;
  edm::Source *fSource;
  std::vector<uint64_t> fSizes;
  edm::EDGetTokenT<OutputStorage>   *outputToken_;
  edm::EDGetTokenT<SizeStorage> *sizeToken_;
};

void BSTest::setItAll(unsigned int iId,std::vector<std::string> const& esproducers,std::vector<std::string> runs) { 
  std::string datadir = "/models/identity_fp32/1/data";
  fSource = new edm::Source(1, reg_, datadir);
  for (auto const& name : esproducers) {
    fPluginManager.load(name);
    auto esp = edm::ESPluginFactory::create(name, datadir);
    esp->produce(fSetup);
  }
  fStream.emplace_back(reg_,fPluginManager,fSource,&fSetup,iId,runs);
  std::cout << "---> check " << reg_.size() << std::endl;
  outputToken_ = new edm::EDGetTokenT<OutputStorage>(fStream[0].registry_.consumes<OutputStorage>());
  sizeToken_   = new edm::EDGetTokenT<SizeStorage>(fStream[0].registry_.consumes<SizeStorage>());
}
void BSTest::fillSource(const void* input_buffer,bool iClear) {
  fSource->fill(input_buffer,iClear);
}
const void** BSTest::getOutput() { 
  auto globalWaitTask = edm::make_empty_waiting_task();
  globalWaitTask->increment_ref_count();
  for (auto& s : fStream) {
    auto pTask = edm::WaitingTaskHolder(globalWaitTask.get());
    s.runToCompletionAsync(pTask);
  }
  globalWaitTask->wait_for_all();
  if (globalWaitTask->exceptionPtr()) {
    std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
  }
  const void** iOutputs = new const void*[fSource->lastEvents_.size()];
  fSizes.clear();
  for(unsigned i0 = 0; i0 < fSource->lastEvents_.size(); i0++) { 
    auto eventPtr = fSource->lastEvents_[i0].get();
    auto const& output   = eventPtr->get(*outputToken_);
    iOutputs[i0]  =  reinterpret_cast<const void*>(output.get());
    auto const& pSize = eventPtr->get(*sizeToken_);
    fSizes.push_back(pSize.get()[0]);
  }
  return iOutputs;
}
uint64_t* BSTest::getSize()  { 
  return  fSizes.data();
}
