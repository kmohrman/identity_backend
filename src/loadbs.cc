#include "CUDACore/Product.h"
#include "CUDACore/ScopedContext.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/host_noncached_unique_ptr.h"
#include "CUDADataFormats/BeamSpotCUDA.h"
#include "Framework/ProductRegistry.h"
//#include "Framework/EmptyWaitingTask.h"
//#include "Framework/WaitingTask.h"
//#include "Framework/WaitingTaskHolder.h"
#include "DataFormats/FEDRawDataCollection.h"
//#include "plugin-SiPixelClusterizer/SiPixelRawToClusterCUDA.cc"

#include "Framework/ESPluginFactory.h"
#include "Framework/PluginFactory.h"
#include "Framework/EventSetup.h"
#include "DataFormats/BeamSpotPOD.h"
//#include "PluginManager.h"
//#include "PluginManager.cc"
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
//#include <filesystem>


/*
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
*/

class BSTest  {
public:
  explicit BSTest(std::string const& datadir) : data_(datadir),
						//bsPutToken_{reg_.produces<cms::cuda::Product<BeamSpotCUDA>>()},
						bsHost{cms::cuda::make_host_noncached_unique<BeamSpotPOD>(cudaHostAllocWriteCombined)} {
  						//fEvent(0,0,reg_),
						//fSiPixelRawToClusterCUDA(reg_) {
						  fEvent = std::make_unique<edm::Event>(0, 0, reg_);
						}
  void loadBS(int iId);//edm::EventSetup& eventSetup);
  void readDummy();
  void Event();
  void setItAll(unsigned int iId,std::vector<std::string> const& esproducers,std::vector<std::string> runs);  
  void runToCompletion();
  //cms::cuda::host::unique_ptr<uint32_t[]> getOutput();
  void* getOutput();
  void fillSource(const void* input_buffer,bool iClear);
  uint32_t getSize();

private:
  std::string data_;
  const edm::EDPutTokenT<cms::cuda::Product<BeamSpotCUDA> > bsPutToken_;
  cms::cuda::host::noncached::unique_ptr<BeamSpotPOD> bsHost;
  edm::EventSetup fSetup;
  std::unique_ptr<edm::Event>      fEvent;
  std::vector<edm::StreamSchedule> fStream;
  std::vector<FEDRawDataCollection> raw_;
  edm::ProductRegistry reg_;
  edmplugin::PluginManager fPluginManager;
  //SiPixelRawToClusterCUDA fSiPixelRawToClusterCUDA;
  edm::Source *fSource;
  CountValidatorSimple* fOutput;
};

// explicit ~BSTest() { }
void BSTest::readDummy(){
  std::string fileRaw="/models/identity_fp32/1/data/raw.bin";
  std::ifstream in_raw(fileRaw.c_str(), std::ios::binary);
  unsigned int nfeds;
  in_raw.exceptions(std::ifstream::badbit);
  in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int)); 
  while (not in_raw.eof()) {
    in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    raw_.emplace_back(readRaw(in_raw, nfeds));
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
  }
}
void BSTest::loadBS(int iId){ //,void* iContainer) { 
  auto bs = std::make_unique<BeamSpotPOD>();
  std::string file="/models/identity_fp32/1/data/beamspot.bin";
  std::ifstream in(file.c_str(), std::ios::binary);
  in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
  in.read(reinterpret_cast<char*>(bs.get()), sizeof(BeamSpotPOD));
  //bsHost = bs;
  //bsHost = static_cast<cms::cuda::host::noncached::unique_ptr<BeamSpotPOD> >(bs);
  fSetup.put(std::move(bs));
  *bsHost = fSetup.get<BeamSpotPOD>();
  //auto bsHost2 = std::move(bs);
  //bsHost = *(dynamic_cast<cms::cuda::host::noncached::unique_ptr<BeamSpotPOD> >(std::move(bs)));
  cms::cuda::ScopedContextProduce ctx{iId};
  BeamSpotCUDA bsDevice(ctx.stream());
  cms::cuda::copyAsync(bsDevice.ptr(), bsHost, ctx.stream());
  auto lEventPtr = fEvent.get();
  ctx.emplace(*lEventPtr, bsPutToken_, std::move(bsDevice));  
}
void BSTest::Event() {
  auto nextEventTaskHolder = edm::WaitingTaskWithArenaHolder();
  //edm::EventProcessor processor(
  //maxEvents, numberOfStreams, std::move(edmodules), std::move(esmodules), datadir, validation);
  //auto lEventPtr = fEvent.get();
  //fSiPixelRawToClusterCUDA.doProduce(*lEventPtr,fSetup);//produce(fEvent,fSetup);
  //fSiPixelRawToClusterCUDA.doAcquire(*lEventPtr,fSetup,nextEventTaskHolder);
}
void BSTest::setItAll(unsigned int iId,std::vector<std::string> const& esproducers,std::vector<std::string> runs) { 
  std::string datadir = "/models/identity_fp32/1/data";
  fSource = new edm::Source(1, reg_, datadir);
  for (auto const& name : esproducers) {
    fPluginManager.load(name);
    //std::filesystem::path datadir = "/models/identity_fp32/1/data";
    auto esp = edm::ESPluginFactory::create(name, datadir);
    esp->produce(fSetup);
  }
  fStream.emplace_back(reg_,fPluginManager,fSource,&fSetup,iId,runs);
  //fPluginManager.load("CountValidatorSimple");
  //int modInd = 4;
  //reg_.beginModuleConstruction(modInd);
  //fOutput = new CountValidatorSimple(reg_);
}
void BSTest::runToCompletion() {
    // The task that waits for all other work
    auto globalWaitTask = edm::make_empty_waiting_task();
    globalWaitTask->increment_ref_count();
    for (auto& s : fStream) {
      std::cout << " Running CMSSW 1 " <<std::endl;
      auto start = std::chrono::high_resolution_clock::now();
      auto pTask = edm::WaitingTaskHolder(globalWaitTask.get());
      //s.processOneEvent2(globalWaitTask.get());
      //s.processOneEventAsync(pTask);
      s.runToCompletionAsync(pTask);
      auto finish = std::chrono::high_resolution_clock::now();
      auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start);
      std::cout << " Running CMSSW 2  time in mus " <<  microseconds.count() << "µs" << std::endl;
    }
    globalWaitTask->wait_for_all();
    if (globalWaitTask->exceptionPtr()) {
      std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
    }
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
uint32_t BSTest::getSize()  { 
  return  fStream[0].fOutput->getSize();
}
