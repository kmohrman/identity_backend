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
  void* getOutput();
  void fillSource(const void* input_buffer,bool iClear);
  uint64_t getSize();

private:
  std::string data_;
  edm::EventSetup fSetup;
  std::unique_ptr<edm::Event>      fEvent;
  std::vector<edm::StreamSchedule> fStream;
  edm::ProductRegistry reg_;
  edmplugin::PluginManager fPluginManager;
  edm::Source *fSource;
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
}
void BSTest::fillSource(const void* input_buffer,bool iClear) {
  fSource->fill(input_buffer,iClear);
}
void* BSTest::getOutput() { 
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
  //auto eventPtr = fSource->lastEvent_.get();
  //fStream[0].fOutput->produce(*eventPtr,fSetup);
  void* iInput = new void*[500000];//reinterpret_cast<void*>(fStream[0].fOutput->getOutput());
  return iInput;
}
uint64_t BSTest::getSize()  { 
  return  fStream[0].fOutput->getSize();
}
