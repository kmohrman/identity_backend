//#include <iostream>

#include <tbb/task.h>

#include "Framework/PluginFactory.h"
#include "Framework/FunctorTask.h"
#include "Framework/WaitingTask.h"
#include "Framework/Worker.h"

#include "PluginManager.h"
#include "PluginManager.cc"
#include "StreamSchedule.h"

namespace edm {
  StreamSchedule::StreamSchedule(ProductRegistry reg,
                                 edmplugin::PluginManager& pluginManager,
                                 Source* source,
                                 EventSetup const* eventSetup,
                                 int streamId,
                                 std::vector<std::string> const& path)
    : registry_(std::move(reg)), source_(source), eventSetup_(eventSetup), streamId_(streamId)
   {
    path_.reserve(path.size());
    int modInd = 1;
    for (auto const& name : path) {
      pluginManager.load(name);
      registry_.beginModuleConstruction(modInd);
      path_.emplace_back(PluginFactory::create(name, registry_));
      std::vector<Worker*> consumes;
      for (unsigned int depInd : registry_.consumedModules()) {
        if (depInd != ProductRegistry::kSourceIndex) {
          consumes.push_back(path_[depInd - 1].get());
        }
      }
      path_.back()->setItemsToGet(std::move(consumes));
      ++modInd;
    }
    //pluginManager.load("CountValidatorSimple");
    //registry_.beginModuleConstruction(modInd);
    //fOutput = new CountValidatorSimple(registry_);
  }

  StreamSchedule::~StreamSchedule() = default;
  StreamSchedule::StreamSchedule(StreamSchedule&&) = default;
  StreamSchedule& StreamSchedule::operator=(StreamSchedule&&) = default;

  void StreamSchedule::runToCompletionAsync(WaitingTaskHolder h) {
    auto task =
      make_functor_task(tbb::task::allocate_root(), [this, h]() mutable { processOneEventAsync(std::move(h)); });
    if (streamId_ == 0) {
      tbb::task::spawn(*task);
    } else {
      tbb::task::enqueue(*task);
    }
  }

  void StreamSchedule::processOneEventAsync(WaitingTaskHolder h) {
    auto event = source_->produce(streamId_, registry_);
    if (event) {
      // Pass the event object ownership to the "end-of-event" task
      // Pass a non-owning pointer to the event to preceding tasks
      //std::cout << "Begin processing event " << event->eventID() << std::endl;
      auto eventPtr = event.get();
      auto nextEventTask =
          make_waiting_task(tbb::task::allocate_root(),
                            [this, h = std::move(h), ev = std::move(event)](std::exception_ptr const* iPtr) mutable {
                              ev.reset();
                              if (iPtr) {
                                h.doneWaiting(*iPtr);
                              } else {
                                for (auto const& worker : path_) {
                                  worker->reset();
                                }
                                processOneEventAsync(std::move(h));
                              }
                            });
      // To guarantee that the nextEventTask is spawned also in
      // absence of Workers, and also to prevent spawning it before
      // all workers have been processed (should not happen though)
      auto nextEventTaskHolder = WaitingTaskHolder(nextEventTask);

      for (auto iWorker = path_.rbegin(); iWorker != path_.rend(); ++iWorker) {
        (*iWorker)->doWorkAsync(*eventPtr, *eventSetup_, nextEventTask);
      }
    } else {
      h.doneWaiting(std::exception_ptr{});
    }
  }

  void StreamSchedule::processOneEvent2(WaitingTask* h) {
    auto event = source_->produce(streamId_, registry_);
    if (event) {
      auto eventPtr = event.get();
      //auto nextEventTaskHolder = WaitingTaskHolder(nextEventTask);
      for (auto iWorker = path_.rbegin(); iWorker != path_.rend(); ++iWorker) {
        //std::cout << "calling doWorkAsync for " << iWorker->get() << " with nextEventTask " << h << std::endl;
        (*iWorker)->doWorkAsync(*eventPtr, *eventSetup_, h);
      }
    }
    else {
      //h->dependentTaskFailed(std::exception_ptr{});
      auto task = h;
      h = nullptr;
      if (0 == task->decrement_ref_count()) {
        tbb::task::spawn(*task);
      }
    }
  }
  
  uint32_t StreamSchedule::processOneEvent(WaitingTaskHolder h) {
    auto event = source_->produce(streamId_, registry_);
    if (event) {
      // Pass the event object ownership to the "end-of-event" task
      // Pass a non-owning pointer to the event to preceding tasks
      //std::cout << "Begin processing event " << event->eventID() << std::endl;
      auto eventPtr = event.get();
      auto nextEventTask =
          make_waiting_task(tbb::task::allocate_root(),
                            [this, h = std::move(h), ev = std::move(event)](std::exception_ptr const* iPtr) mutable {
                              ev.reset();
                              if (iPtr) {
                                h.doneWaiting(*iPtr);
                              } else {
                                for (auto const& worker : path_) {
                                  worker->reset();
                                }
                                processOneEventAsync(std::move(h));
                              }
                            });
      // To guarantee that the nextEventTask is spawned also in
      // absence of Workers, and also to prevent spawning it before
      // all workers have been processed (should not happen though)
      auto nextEventTaskHolder = WaitingTaskHolder(nextEventTask);
      for (auto iWorker = path_.rbegin(); iWorker != path_.rend(); ++iWorker) {
        //std::cout << "calling doWorkAsync for " << iWorker->get() << " with nextEventTask " << nextEventTask << std::endl;
        (*iWorker)->doWorkAsync(*eventPtr, *eventSetup_, nextEventTask);
      }
    } else {
      h.doneWaiting(std::exception_ptr{});
      uint32_t out = 0;
      return out;
    }
    //auto eventPtr = event.get();
    //fOutput->produce(*eventPtr, *eventSetup_);
    uint32_t  output = 0;//fOutput->getOutput();
    return output;
  }

  /*
  cms::cuda::host::unique_ptr<uint32_t[]> StreamSchedule::processOneEvent(WaitingTaskHolder h) {
    edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiToken_(registry_.consumes<cms::cuda::Product<SiPixelDigisCUDA>>());
    edm::EDGetTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterToken_(registry_.consumes<cms::cuda::Product<SiPixelClustersCUDA>>());
    auto event = source_->produce(streamId_, registry_);
    if (event) {
      // Pass the event object ownership to the "end-of-event" task
      // Pass a non-owning pointer to the event to preceding tasks
      //std::cout << "Begin processing event " << event->eventID() << std::endl;
      auto eventPtr = event.get();
      auto nextEventTask =
          make_waiting_task(tbb::task::allocate_root(),
                            [this, h = std::move(h), ev = std::move(event)](std::exception_ptr const* iPtr) mutable {
                              ev.reset();
                              if (iPtr) {
                                h.doneWaiting(*iPtr);
                              } else {
                                for (auto const& worker : path_) {
                                  worker->reset();
                                }
                                processOneEventAsync(std::move(h));
                              }
                            });
      // To guarantee that the nextEventTask is spawned also in
      // absence of Workers, and also to prevent spawning it before
      // all workers have been processed (should not happen though)
      auto nextEventTaskHolder = WaitingTaskWithArenaHolder(nextEventTask); ///Phil: This is a guess!
      //auto nextEventTaskHolder = WaitingTaskHolder(nextEventTask);

      for (auto iWorker = path_.rbegin(); iWorker != path_.rend(); ++iWorker) {
        std::cout << "calling doWorkAsync for " << iWorker->get() << " with nextEventTask " << nextEventTask << std::endl;
        (*iWorker)->doWorkAsync(*eventPtr, *eventSetup_, nextEventTask);
      }
      std::cout << " --> getting Digis 1 " << std::endl;
      auto const& pdigis = eventPtr->get(digiToken_);
      std::cout << " --> getting Digis 1.5 " << std::endl;
      cms::cuda::ScopedContextProduce ctx{pdigis};
      //cms::cuda::ScopedContextAcquire ctx{pdigis, std::move(nextEventTaskHolder)};
      std::cout << " --> getting Digis 2 " << std::endl;
      auto const& digis = ctx.get(*eventPtr, digiToken_);
      std::cout << " --> getting Digis 3 " << std::endl;      
      auto const& clusters = ctx.get(*eventPtr, clusterToken_);
      std::cout << " --> getting Cluster " << std::endl;      
      uint32_t nModules = digis.nModules();
      std::cout << "---> " << nModules << std::endl;
      cms::cuda::host::unique_ptr<uint32_t[]> h_clusInModule = cms::cuda::make_host_unique<uint32_t[]>(nModules, ctx.stream());
      cudaCheck(cudaMemcpyAsync(
				h_clusInModule.get(), clusters.clusInModule(), sizeof(uint32_t) * nModules, cudaMemcpyDefault, ctx.stream()));
      return h_clusInModule;
    } else {
      h.doneWaiting(std::exception_ptr{});
      cms::cuda::host::unique_ptr<uint32_t[]> h_clusInModule;
      return h_clusInModule;
    }
  }
  */
  void StreamSchedule::endJob() {
    for (auto& w : path_) {
      w->doEndJob();
    }
  }
}  // namespace edm
