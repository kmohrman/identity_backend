// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <memory>
#include <thread>
#include "triton/backend/backend_common.h"
#include <tbb/task_scheduler_init.h>
//#include "loadlst.cc"
#include "../SDL/LST.h"
//#include "vector_add.cu"

namespace triton { namespace backend { namespace identity {
 

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Does this model support batching in the first dimension. This
  // function should not be called until after the model is completely
  // loaded.
  TRITONSERVER_Error* SupportsFirstDimBatching(bool* supports);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Block the thread for seconds specified in 'creation_delay_sec' parameter.
  // This function is used for testing.
  TRITONSERVER_Error* CreationDelay();
  ////BSTest* fBSTest;
  SDL::LST* fLST;
  
 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version,
      common::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  

  const std::string name_;
  const uint64_t version_;
  common::TritonJson::Value model_config_;

  bool supports_batching_initialized_;
  bool supports_batching_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version,
      std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version,
    common::TritonJson::Value&& model_config)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), model_config_(std::move(model_config)),
      supports_batching_initialized_(false), supports_batching_(false)
{
}

TRITONSERVER_Error*
ModelState::SupportsFirstDimBatching(bool* supports)
{
  // We can't determine this during model initialization because
  // TRITONSERVER_ServerModelBatchProperties can't be called until the
  // model is loaded. So we just cache it here.
  if (!supports_batching_initialized_) {
    uint32_t flags = 0;
    RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
        triton_server_, name_.c_str(), version_, &flags, nullptr /* voidp */));
    supports_batching_ = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
    supports_batching_initialized_ = true;
  }

  *supports = supports_batching_;
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreationDelay()
{
  // Feature for testing purpose...
  // look for parameter 'creation_delay_sec' in model config
  // and sleep for the value specified
  common::TritonJson::Value parameters;
  if (model_config_.Find("parameters", &parameters)) {
    common::TritonJson::Value creation_delay_sec;
    if (parameters.Find("creation_delay_sec", &creation_delay_sec)) {
      std::string creation_delay_sec_str;
      RETURN_IF_ERROR(creation_delay_sec.MemberAsString(
          "string_value", &creation_delay_sec_str));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Creation delay is set to : ") + creation_delay_sec_str)
              .c_str());
      std::this_thread::sleep_for(
          std::chrono::seconds(std::stoi(creation_delay_sec_str)));
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 1 input and 1 output.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 input, got ") +
          std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 output, got ") +
          std::to_string(outputs.ArraySize()));

  common::TritonJson::Value input, output;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

  // Input and output must have same datatype
  std::string input_dtype, output_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance()
  {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      instance_id);
  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id)
{
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);
  std::cout << " --> start " << cname << std::endl;
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }
  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  // std::string* state = new std::string("backend state");
  //RETURN_IF_ERROR(
  //     TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));
  std::cout << "--> Done " << std::endl;
  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  //void* vbackendstate;
  //RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  //std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  //LOG_MESSAGE(
  //    TRITONSERVER_LOG_INFO,
  //    (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  // For testing.. Block the thread for certain time period before returning.
  RETURN_IF_ERROR(model_state->CreationDelay());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  //Now initialize Patatrack modules
  ////model_state->fBSTest = new BSTest("data/beamspot.bin");
  model_state->fLST = new SDL::LST();
  model_state->fLST->eventSetup();
  //std::vector<std::string> lESModules;
  //std::vector<std::string> lEDModules;
  //lESModules = {"BeamSpotESProducer",
  //  	"SiPixelGainCalibrationForHLTGPUESProducer",
  //  	"SiPixelROCsStatusAndMappingWrapperESProducer",
  //  	"PixelCPEFastESProducer"};
  //lEDModules = {"BeamSpotToCUDA","SiPixelRawToClusterCUDA","SiPixelRecHitCUDA","SiPixelDigiErrorsSoAFromCUDA", "SiPixelRecHitFromCUDA","CAHitNtupletCUDA", "PixelTrackSoAFromCUDA", "PixelVertexProducerCUDA","PixelVertexSoAFromCUDA","CountValidatorSimple"};
  //model_state->fBSTest->setItAll(0,lESModules,lEDModules);
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  //  LOG_MESSAGE(
  //    TRITONSERVER_LOG_INFO,
  //    (std::string("model ") + model_state->Name() + ", instance " +
  //     instance_state->Name() + ", executing " + std::to_string(request_count) +
  //     " requests")
  //        .c_str());
  //bool supports_batching = false;
  // RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));
  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  std::cout << "THIS IS request_count " << request_count << std::endl;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.
  //uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  //uint64_t max_exec_end_ns = 0;
  //uint64_t total_batch_size = 0;

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  tbb::task_scheduler_init tsi(1);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    // Triton ensures that there is only a single input since that is
    // what is specified in the model configuration, so normally there
    // would be no reason to check it but we do here to demonstate the
    // API.
    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // If an error response was sent for the above then display an
    // error message and move on to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    //LOG_MESSAGE(
    //    TRITONSERVER_LOG_INFO,
    //    (std::string("request ") + std::to_string(r) + ": id = \"" +
    //     request_id + "\", correlation_id = " + std::to_string(correlation_id) +
    //     ", input_count = " + std::to_string(input_count) +
    //     ", requested_output_count = " + std::to_string(requested_output_count))
    //        .c_str());

    const char* input_name;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name));

    TRITONBACKEND_Input* input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, input_name, &input));

    // We also validated that the model configuration specifies only a
    // single output, but the request is not required to request any
    // output at all so we only produce an output if requested.
    const char* requested_output_name = nullptr;
    if (requested_output_count > 0) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(
              request, 0 /* index */, &requested_output_name));
    }
    
    // If an error response was sent while getting the input or
    // requested output name then display an error message and move on
    // to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input or requested output name, error response "
           "sent")
              .c_str());
      continue;
    }

    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            input, nullptr /* input_name */, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input properties, error response sent")
              .c_str());
      continue;
    }
    const void* input_buffer = nullptr;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;
    GUARDED_RESPOND_IF_ERROR(
			     responses, r,
			     TRITONBACKEND_InputBuffer(
						       input, 0, &input_buffer, &buffer_byte_size, &input_memory_type,
						       &input_memory_type_id));
    if ((responses[r] == nullptr) ||
	(input_memory_type == TRITONSERVER_MEMORY_GPU)) {
      GUARDED_RESPOND_IF_ERROR(
			       responses, r,
			       TRITONSERVER_ErrorNew(
						     TRITONSERVER_ERROR_UNSUPPORTED,
						     "failed to get input buffer in CPU memory"));
    }
    //model_state->fBSTest->fillSource(input_buffer,true);
    //for(unsigned int i0 = 1; i0 < input_shape[0]; i0++) model_state->fBSTest->fillSource(input_buffer,false);
    std::cout << "The input_count: " << input_count << std::endl;
    //std::vector<int> *lst_output = model_state->fLST->readRawBuff(input_buffer);
    //model_state->fLST->readRawBuff(input_buffer);
    std::vector<int> *lst_output = model_state->fLST->readRawBuff(input_buffer);
    std::cout << "What is this?: " << lst_output->at(0) << std::endl;

    std::vector<int> lst_output_direct = model_state->fLST->lst_output;
    std::vector<int> *lst_output_direct_p = &(model_state->fLST->lst_output);

    std::cout << "lst_output_direct[0]" << lst_output_direct.at(0) << std::endl;
    std::cout << "lst_output_direct_p[0]" << lst_output_direct_p->at(0) << std::endl;

    int first_item_0;
    std::memcpy(&first_item_0, lst_output_direct.data(), sizeof(int));
    std::cout << "THIS IS first_item_0: " << first_item_0 << std::endl;

    int first_item_1;
    std::memcpy(&first_item_1, lst_output_direct_p->data(), sizeof(int));
    std::cout << "THIS IS first_item_1: " << first_item_1 << std::endl;


    // Run LST on a hard coded event
    bool run_hardcoded = false;
    if (run_hardcoded) {
      cudaStream_t stream = 0;
      bool verbose;
      std::vector<float> see_px;
      std::vector<float> see_py;
      std::vector<float> see_pz;
      std::vector<float> see_dxy;
      std::vector<float> see_dz;
      std::vector<float> see_ptErr;
      std::vector<float> see_etaErr;
      std::vector<float> see_stateTrajGlbX;
      std::vector<float> see_stateTrajGlbY;
      std::vector<float> see_stateTrajGlbZ;
      std::vector<float> see_stateTrajGlbPx;
      std::vector<float> see_stateTrajGlbPy;
      std::vector<float> see_stateTrajGlbPz;
      std::vector<int> see_q;
      std::vector<std::vector<int>> see_hitIdx;
      std::vector<unsigned int> ph2_detId;
      std::vector<float> ph2_x;
      std::vector<float> ph2_y;
      std::vector<float> ph2_z;

      verbose = true;
      // From /blue/p.chang/p.chang/data/lst/CMSSW_12_2_0_pre2/trackingNtuple_10mu_pt_0p5_50.root
      see_px = { 39.87165069580078,  20.210636138916016,  -8.118329048156738,  -11.748462677001953,  -5.885512351989746,  -7.5683417320251465,  -24.35856819152832,  -28.708385467529297,  24.15365219116211,  28.33498191833496,  34.74605178833008,  11.07693862915039,  28.408700942993164,  11.69003677368164,  -8.877352714538574,  -15.017558097839355,  -14.734334945678711,  -6.289051055908203,  -9.415787696838379,  -9.00146484375,  -7.3984375,  -10.853144645690918,  -10.840180397033691,  -5.767538070678711,  -7.751054763793945,  -7.574886798858643,  -31.677433013916016,  -46.835140228271484,  49.23747253417969,  33.985809326171875,  7.509385108947754,  7.647968769073486,  7.126499176025391,  8.755658149719238,  9.43903636932373,  8.159379005432129,  7.250615119934082,  7.555498123168945,  6.9113993644714355,  9.083843231201172,  9.829330444335938,  8.49206829071045,  -12.002909660339355,  -11.417337417602539,  -8.56479549407959,  -8.056528091430664,  -8.269803047180176,  -7.984747886657715,  -16.60462760925293,  -15.573132514953613,  -10.416543006896973,  -9.746770858764648,  -9.49640941619873,  -9.155677795410156,  -10.680980682373047,  -10.178755760192871,  -7.697154521942139,  -7.294583797454834,  -8.021846771240234,  -8.043277740478516,  -14.199366569519043,  -13.39987850189209,  -9.278435707092285,  -8.68097972869873,  -9.200770378112793,  -9.240825653076172,  7.04472541809082,  6.3965535163879395,  7.028075218200684,  6.92043924331665,  6.15958309173584,  6.947277545928955,  8.032379150390625,  7.942108154296875,  8.064872741699219,  7.24937105178833,  6.824822902679443,  7.290019989013672,  -8.635575294494629,  -9.368552207946777,  -10.31214714050293,  -11.057748794555664,  -7.5622453689575195,  -8.167137145996094,  -8.361503601074219,  -8.967562675476074,  -7.784886837005615,  -8.124894142150879,  -9.068493843078613,  -9.366728782653809,  11.954262733459473,  10.118915557861328,  11.446704864501953,  9.412618637084961,  11.844738006591797,  10.045613288879395,  -2.7255911827087402,  -2.764690399169922,  2.8147103786468506,  2.911376953125,  3.0747435092926025,  0.8161746263504028,  0.8343629240989685,  0.8590767979621887,  0.8757950067520142,  0.8165073394775391,  0.843512237071991,  0.8685556054115295,  0.8945918083190918,  -2.1008756160736084,  -2.0811822414398193,  -2.1071536540985107,  -2.0862956047058105,  -39.39778518676758,  -0.9653070569038391,  -1.0562235116958618,  -1.0126218795776367,  -1.1015400886535645,  -1.0375455617904663,  -1.007827877998352,  48.15880584716797,  31.1234188079834,  -8.655336380004883,  -30.867149353027344,  8.988533973693848,  -47.76204299926758,  48.09022903442383,  48.046722412109375,  48.14298629760742,  48.242149353027344,  49.33876419067383,  49.3233642578125,  35.851016998291016,  35.77186584472656,  35.13629913330078,  35.16637420654297,  34.8143310546875,  35.56747055053711,  34.728336334228516,  -25.713376998901367,  -25.724712371826172,  -26.38947296142578,  -26.378326416015625,  -25.952926635742188,  -26.615175247192383,  -53.02956008911133,  -53.26742172241211,  -60.210227966308594,  -58.97599411010742,  -50.293087005615234,  -50.319271087646484,  -51.67558670043945,  -51.119407653808594};
      see_py = { -32.373538970947266,  -15.42562198638916,  -14.806164741516113,  -21.386451721191406,  -10.765464782714844,  -13.817461013793945,  18.65734100341797,  21.9686336517334,  -18.450763702392578,  -21.6581974029541,  63.67839813232422,  20.233976364135742,  52.04657745361328,  21.360755920410156,  -16.174135208129883,  -27.275808334350586,  -26.76331901550293,  -11.501237869262695,  -17.15812873840332,  -16.408414840698242,  -13.502241134643555,  -19.749841690063477,  -19.72626304626465,  -10.558887481689453,  -14.148332595825195,  -13.82935619354248,  24.224666595458984,  35.74595260620117,  -37.75737380981445,  -26.010225296020508,  13.668986320495605,  13.92324161529541,  12.964860916137695,  15.964341163635254,  17.22241973876953,  14.867188453674316,  13.195935249328613,  13.75601577758789,  12.572153091430664,  16.571308135986328,  17.943363189697266,  15.482563972473145,  -21.818143844604492,  -20.76190757751465,  -15.625120162963867,  -14.707406997680664,  -15.092673301696777,  -14.577938079833984,  -30.10068130493164,  -28.241657257080078,  -18.956865310668945,  -17.7482852935791,  -17.29837417602539,  -16.683475494384766,  -19.432729721069336,  -18.527856826782227,  -14.059626579284668,  -13.332680702209473,  -14.642753601074219,  -14.681307792663574,  -25.76390266418457,  -24.324295043945312,  -16.90457534790039,  -15.826533317565918,  -16.762508392333984,  -16.834627151489258,  12.810295104980469,  11.614461898803711,  12.779552459716797,  12.580023765563965,  11.17542552947998,  12.629655838012695,  14.63287353515625,  14.466714859008789,  14.692583084106445,  13.185650825500488,  12.401373863220215,  13.260648727416992,  -15.75167179107666,  -17.06856918334961,  -18.76017951965332,  -20.09942054748535,  -13.821703910827637,  -14.908095359802246,  -15.253927230834961,  -16.34210777282715,  -14.221921920776367,  -14.83201789855957,  -16.524450302124023,  -17.05946159362793,  21.8929443359375,  18.489749908447266,  20.95414161682129,  17.182172775268555,  21.690227508544922,  18.354177474975586,  2.5754382610321045,  2.6136200428009033,  -2.649737596511841,  -2.737015724182129,  -2.884345054626465,  0.6451150178909302,  0.6609541773796082,  0.6831649541854858,  0.6976981163024902,  0.6454364657402039,  0.6692317128181458,  0.6917650103569031,  0.7146821022033691,  1.9627729654312134,  1.9435056447982788,  1.968960165977478,  1.9485478401184082,  31.943132400512695,  -0.7729750871658325,  -0.8384506106376648,  -0.8080663681030273,  -0.8721997141838074,  -0.8250266313552856,  -0.8037440776824951,  -39.07398986816406,  -23.802772521972656,  -15.7833251953125,  23.610769271850586,  16.396451950073242,  38.763153076171875,  -38.99638366699219,  -38.9619255065918,  -39.03169250488281,  -39.106719970703125,  -39.98252868652344,  -39.96965026855469,  -27.553516387939453,  -27.490478515625,  -26.937564849853516,  -26.960424423217773,  -26.686492919921875,  -27.27593421936035,  -26.61949920654297,  19.762243270874023,  19.77130699157715,  20.258014678955078,  20.251384735107422,  19.956314086914062,  20.449495315551758,  43.28916549682617,  43.4815788269043,  49.4258918762207,  48.37759017944336,  40.940879821777344,  40.96249008178711,  42.1453857421875,  41.62306594848633};
      see_pz = { 102.9832763671875,  -88.19041442871094,  104.47129821777344,  150.95123291015625,  75.90970611572266,  97.46934509277344,  106.4661865234375,  125.39421081542969,  -105.42803192138672,  -123.71235656738281,  -449.241943359375,  -142.85501098632812,  -367.2718200683594,  -150.8206024169922,  114.1419677734375,  192.6598663330078,  189.0045623779297,  81.0625228881836,  121.0994873046875,  115.73025512695312,  95.24909973144531,  139.4384002685547,  139.2493133544922,  74.41570281982422,  99.82036590576172,  97.52598571777344,  138.23590087890625,  204.2759552001953,  -215.14793395996094,  -148.41177368164062,  -96.61841583251953,  -98.30241394042969,  -91.681640625,  -112.79459381103516,  -121.54533386230469,  -105.08798217773438,  -93.24540710449219,  -97.09212493896484,  -88.87865447998047,  -117.02555847167969,  -126.57382202148438,  -109.38743591308594,  154.05982971191406,  146.60020446777344,  110.25863647460938,  103.75537872314453,  106.47212982177734,  102.8404541015625,  212.67169189453125,  199.54002380371094,  133.8407745361328,  125.27928924560547,  122.08378601074219,  117.745361328125,  137.1685791015625,  130.79685974121094,  99.17840576171875,  94.03207397460938,  103.27254486083984,  103.57693481445312,  181.96490478515625,  171.82228088378906,  119.31793212890625,  111.68363189697266,  118.2731704711914,  118.81879425048828,  -90.56780242919922,  -82.11650848388672,  -90.3547592163086,  -88.94398498535156,  -79.0400619506836,  -89.29547119140625,  -103.41069030761719,  -102.20538330078125,  -103.84146118164062,  -93.21522521972656,  -87.67705535888672,  -93.74784851074219,  111.17901611328125,  120.4735107421875,  132.57223510742188,  142.05775451660156,  97.52521514892578,  105.17664337158203,  107.71450805664062,  115.4034194946289,  100.34893035888672,  104.64054107666016,  116.74583435058594,  120.533203125,  -154.4860076904297,  -130.54074096679688,  -147.94781494140625,  -121.37741088867188,  -153.03733825683594,  -129.57017517089844,  -53.014060974121094,  -53.782630920410156,  54.62260818481445,  56.451847076416016,  59.570072174072266,  -19.61507225036621,  -20.09921646118164,  -20.69693374633789,  -21.143489837646484,  -19.63486671447754,  -20.343360900878906,  -20.95399284362793,  -21.63762855529785,  -40.71340560913086,  -40.33354187011719,  -40.836280822753906,  -40.432559967041016,  -101.6891860961914,  23.446203231811523,  25.540164947509766,  24.531314849853516,  26.57099151611328,  25.074169158935547,  24.38584327697754,  124.35140228271484,  -135.91949462890625,  111.37667846679688,  134.80416870117188,  -115.8166732788086,  -123.3208236694336,  123.74313354492188,  123.60462951660156,  123.75849914550781,  123.97946166992188,  127.70701599121094,  127.63870239257812,  -155.8209991455078,  -155.42160034179688,  -153.8428955078125,  -154.02609252929688,  -152.0731964111328,  -155.8533477783203,  -151.6287384033203,  113.05592346191406,  113.14877319335938,  115.50179290771484,  115.50643920898438,  113.44928741455078,  117.13196563720703,  -136.9005584716797,  -137.80079650878906,  -155.91758728027344,  -152.8220977783203,  -129.35536193847656,  -129.4439697265625,  -132.60162353515625,  -132.25082397460938};
      see_dxy = { 0.0014435771154239774,  -0.0024830386973917484,  0.0018283006502315402,  0.0032594013027846813,  -0.0006701125530526042,  0.0007711647776886821,  0.001624779193662107,  0.0004742662131320685,  -0.0014338938053697348,  -0.0007530140574090183,  -0.0051228804513812065,  -0.002480496885254979,  -0.004894590936601162,  -0.0028069978579878807,  0.0032507407013326883,  0.006582696922123432,  0.006522062234580517,  -0.0011756805470213294,  0.0025918083265423775,  0.0022608451545238495,  0.0016175227938219905,  0.00467539532110095,  0.004682864528149366,  -0.0021474037785083055,  0.0009508490329608321,  0.0007645381847396493,  -0.0016062131617218256,  -0.005727509502321482,  0.007118320558220148,  0.0023215741384774446,  0.0019701451528817415,  0.001840501558035612,  0.0026427211705595255,  -0.00026930077001452446,  -0.0011760166380554438,  0.0005677614826709032,  0.0016446377849206328,  0.00124691694509238,  0.002237741369754076,  -0.0012482027523219585,  -0.002068885834887624,  -0.0005433885380625725,  0.006012029480189085,  0.005454722326248884,  0.0003934030537493527,  -0.0004950519069097936,  -0.0001379105815431103,  -0.0006500954041257501,  0.011094988323748112,  0.010528258979320526,  0.004968132358044386,  0.004171508364379406,  0.0035891958978027105,  0.0031108863186091185,  0.005006629042327404,  0.004276178311556578,  -0.0011513205245137215,  -0.002013488905504346,  -2.0195084289298393e-05,  3.7520803743973374e-05,  0.010026042349636555,  0.009316051378846169,  0.0035491385497152805,  0.0026273916009813547,  0.003704957664012909,  0.003782422048971057,  0.003557331394404173,  0.005724829155951738,  0.0036094028037041426,  0.004267339129000902,  0.007315834052860737,  0.00414270767942071,  0.0009048061911016703,  0.0010630922624841332,  0.0008646014612168074,  0.003716743318364024,  0.005432487465441227,  0.0035783331841230392,  0.0003616071480792016,  0.002607310190796852,  0.006143821403384209,  0.00777426129207015,  -0.0031018629670143127,  -0.0005721115157939494,  0.00134738115593791,  0.0034523149952292442,  -0.0022580446675419807,  -0.0006597914034500718,  0.0034071202389895916,  0.004468217026442289,  -0.009798719547688961,  -0.004506868310272694,  -0.009078984148800373,  -0.0025526683311909437,  -0.009629474952816963,  -0.004366330336779356,  0.0026255021803081036,  0.0032129271421581507,  -0.000673716829624027,  -0.0027948494534939528,  -0.006341614294797182,  0.01015761960297823,  0.008068922907114029,  0.0038693295791745186,  0.002204414689913392,  0.010005277581512928,  0.005810809321701527,  0.0016074837185442448,  -0.001897907699458301,  -0.014634008519351482,  -0.01527541782706976,  -0.01440214179456234,  -0.015079601667821407,  -0.002348099136725068,  -0.0009468270000070333,  0.010802694596350193,  0.0023805107921361923,  0.012923728674650192,  0.008137203752994537,  0.004261254332959652,  -0.00016276691167149693,  0.00028632392059080303,  0.0016032969579100609,  -0.0005256824661046267,  -0.0011378808412700891,  -0.00035809617838822305,  -0.017547067254781723,  -0.018014540895819664,  -0.024632545188069344,  -0.024538744240999222,  -0.015311772003769875,  -0.01564750261604786,  -0.010072581470012665,  -0.008302661590278149,  0.0005539977573789656,  -0.0003690477751661092,  0.011142549104988575,  -0.011111638508737087,  0.01398569718003273,  0.029665375128388405,  0.029872460290789604,  0.043360475450754166,  0.04326995089650154,  0.0406651645898819,  0.054417334496974945,  0.12982383370399475,  0.12458178400993347,  0.074856698513031,  0.11727132648229599,  0.0829092413187027,  0.08263138681650162,  0.0978020504117012,  0.08850549161434174};
      see_dz = { 0.5306985974311829,  0.5379186272621155,  0.5484537482261658,  0.5507892370223999,  0.5483153462409973,  0.5491196513175964,  0.5332756042480469,  0.536558985710144,  0.5371520519256592,  0.5377475619316101,  0.538401186466217,  0.5381228923797607,  0.5413116812705994,  0.540708065032959,  0.5506625175476074,  0.5451840758323669,  0.5507078766822815,  0.561480700969696,  0.545707106590271,  0.5628185272216797,  0.5509226322174072,  0.5455765128135681,  0.5507786273956299,  0.5574732422828674,  0.5453033447265625,  0.557796835899353,  0.5641169548034668,  0.5510123372077942,  0.5251953601837158,  0.5284637808799744,  0.5494840145111084,  0.5138992667198181,  0.559454619884491,  0.5483866333961487,  0.5173068046569824,  0.557158350944519,  0.5302144885063171,  0.49777865409851074,  0.5408999919891357,  0.5288329124450684,  0.49818670749664307,  0.5394079685211182,  0.5496417880058289,  0.5465924739837646,  0.543201744556427,  0.5473402142524719,  0.5495222210884094,  0.5464977025985718,  0.5510504245758057,  0.5473150014877319,  0.5446963906288147,  0.5479593873023987,  0.550368070602417,  0.5472261309623718,  0.553259551525116,  0.5458762645721436,  0.5446239709854126,  0.5463380813598633,  0.5552672743797302,  0.5454753041267395,  0.55565345287323,  0.5475645065307617,  0.5452711582183838,  0.5477084517478943,  0.5567442774772644,  0.5470816493034363,  0.5488709211349487,  0.5391759276390076,  0.5500729084014893,  0.5467268228530884,  0.5493376851081848,  0.5468612313270569,  0.5492763519287109,  0.5387774109840393,  0.5522828698158264,  0.548048734664917,  0.5476728677749634,  0.5488652586936951,  0.5333482027053833,  0.5439490675926208,  0.4896846115589142,  0.4893946945667267,  0.5282347202301025,  0.5453301668167114,  0.5025352835655212,  0.5069913268089294,  0.5319708585739136,  0.5439207553863525,  0.47960424423217773,  0.480433851480484,  0.5350872874259949,  0.5356789827346802,  0.5753836035728455,  0.5637020468711853,  0.5229751467704773,  0.5249722599983215,  0.4294787645339966,  0.43445441126823425,  0.634148359298706,  0.6603400111198425,  0.6324892640113831,  0.1719820499420166,  0.3033774793148041,  0.15022149682044983,  0.2711695730686188,  0.2363121062517166,  0.3740379810333252,  0.2444494366645813,  0.3671005070209503,  0.5739944577217102,  0.5964694023132324,  0.5683115720748901,  0.5875035524368286,  0.5283555388450623,  0.4794953763484955,  0.5940616130828857,  0.5709834694862366,  0.7061455845832825,  0.7265850901603699,  0.6996791362762451,  0.5305230617523193,  0.5396535992622375,  0.545448362827301,  0.5363614559173584,  0.5422065258026123,  0.5275655388832092,  1.1413565874099731,  1.3109358549118042,  1.7543213367462158,  1.919954776763916,  -3.5643651485443115,  -3.4001991748809814,  -4.67738676071167,  -4.966463565826416,  2.422884702682495,  2.7048704624176025,  0.5258795619010925,  2.9649927616119385,  0.17082011699676514,  -0.9504974484443665,  -1.2218213081359863,  2.5677473545074463,  2.280766487121582,  4.04732608795166,  -0.5658145546913147,  -1.0115752220153809,  0.6412320733070374,  -0.39829933643341064,  0.3512372672557831,  -3.030770778656006,  -2.9022514820098877,  -5.480184555053711,  1.5309590101242065};
      see_ptErr = { 0.18760524690151215,  0.19318126142024994,  0.5871739983558655,  1.3122867345809937,  0.3144877254962921,  0.5571578741073608,  0.21720357239246368,  0.2880745828151703,  0.1396036595106125,  0.18636657297611237,  11.454681396484375,  0.7822906970977783,  10.133988380432129,  1.3189784288406372,  0.5958247184753418,  1.7756963968276978,  1.6167353391647339,  0.3023035228252411,  0.701651394367218,  0.6077187061309814,  0.40219980478286743,  0.9016399383544922,  0.847623348236084,  0.24569013714790344,  0.4617851674556732,  0.41697731614112854,  0.5021517872810364,  1.1564005613327026,  1.0336153507232666,  0.5040810704231262,  0.4749266505241394,  0.44049662351608276,  0.4050142168998718,  0.4867732524871826,  0.4540836811065674,  0.3965107500553131,  0.4565838873386383,  0.4375058114528656,  0.3893609642982483,  0.5277464389801025,  0.5083855390548706,  0.43432849645614624,  0.6185556054115295,  0.576867938041687,  0.32340753078460693,  0.29009318351745605,  0.20754596590995789,  0.19574715197086334,  1.2417305707931519,  1.1260656118392944,  0.5012829303741455,  0.446125328540802,  0.28683263063430786,  0.2708331346511841,  0.4936889410018921,  0.4624258279800415,  0.2705520987510681,  0.24040403962135315,  0.19728395342826843,  0.1949029117822647,  0.9135509729385376,  0.8394654393196106,  0.4124123454093933,  0.35710999369621277,  0.2716531455516815,  0.26958921551704407,  0.17951080203056335,  0.1484318971633911,  0.1851922571659088,  0.2309120148420334,  0.1800607591867447,  0.23951219022274017,  0.11514155566692352,  0.11157823354005814,  0.1151357963681221,  0.21246865391731262,  0.18614277243614197,  0.21606771647930145,  0.2301146388053894,  0.26080673933029175,  0.3226873576641083,  0.35544660687446594,  0.17446711659431458,  0.19869065284729004,  0.20758314430713654,  0.23401547968387604,  0.16266845166683197,  0.17482906579971313,  0.21253009140491486,  0.22436600923538208,  0.4696238040924072,  0.3132462203502655,  0.41678741574287415,  0.25893014669418335,  0.4503155052661896,  0.30214768648147583,  0.06447294354438782,  0.06907770037651062,  0.0466463677585125,  0.05067018046975136,  0.0581519678235054,  0.013431071303784847,  0.014161947183310986,  0.014918382279574871,  0.015593167394399643,  0.014329979196190834,  0.015534847043454647,  0.016387062147259712,  0.017587244510650635,  0.029326561838388443,  0.028661327436566353,  0.028821241110563278,  0.028166769072413445,  0.4715159237384796,  0.03616654872894287,  0.04620860517024994,  0.03895372152328491,  0.049761176109313965,  0.028897440060973167,  0.02426392398774624,  0.014778939075767994,  0.015905536711215973,  0.01438144315034151,  0.01559689361602068,  0.015231295488774776,  0.015352202579379082,  0.369840532541275,  0.3693244457244873,  0.35683581233024597,  0.3576928377151489,  0.36183062195777893,  0.3615289628505707,  0.6114336252212524,  0.6091896295547485,  0.5594491958618164,  0.560326337814331,  0.49082037806510925,  0.5208179354667664,  0.487885445356369,  0.5369307994842529,  0.5373145341873169,  0.5383020639419556,  0.5381396412849426,  0.46510568261146545,  0.47390225529670715,  0.28855085372924805,  0.27168306708335876,  0.47722798585891724,  0.4264363944530487,  0.2583577036857605,  0.25862881541252136,  0.2836274206638336,  0.28268638253211975};
      see_etaErr = { 6.457656854763627e-05,  0.00010636897786753252,  0.00021065532928332686,  0.0002725214581005275,  0.00015472630911972374,  0.00018409214681014419,  0.00020241712627466768,  0.0001036732064676471,  0.00011041451944038272,  0.00011320347402943298,  0.00023483468976337463,  0.00023280658933799714,  0.0006276516942307353,  0.000489475904032588,  0.0002470552281010896,  0.0002478386741131544,  0.00024887113249860704,  0.0003986189258284867,  0.0004036372993141413,  0.0004096018965356052,  0.0002626478672027588,  0.0002633517433423549,  0.0002641002065502107,  0.00035259476862847805,  0.00035599939292296767,  0.00035905386903323233,  0.00031689656316302717,  0.00028236969956196845,  0.0002856109640561044,  0.0002943731960840523,  0.000654339964967221,  0.0006037827697582543,  0.0006137864547781646,  0.0006051237578503788,  0.0005639624432660639,  0.0005653160624206066,  0.0007593373884446919,  0.0007056128233671188,  0.0007176374783739448,  0.0007393392734229565,  0.0006905739428475499,  0.000700087402947247,  0.00029964075656607747,  0.00012822031567338854,  0.000288220850052312,  0.00010128405119758099,  0.0002905844303313643,  0.00010064487287309021,  0.0002706825325731188,  0.00010402646148577332,  0.00026243223692290485,  8.359044295502827e-05,  0.0002642055624164641,  8.158724813256413e-05,  0.0003757151134777814,  0.00018357386579737067,  0.0003565276856534183,  0.00014285460929386318,  0.0003633523010648787,  0.00014544122677762061,  0.0003654731553979218,  0.00014929691678844392,  0.00034982923534698784,  0.00010877697786781937,  0.0003561003250069916,  0.00010960255895042792,  0.0003707936848513782,  0.0003585538943298161,  0.0003598226758185774,  0.0005814431933686137,  0.0005736841121688485,  0.0005784351960755885,  0.00034221733221784234,  0.00033365681883879006,  0.0003221595543436706,  0.0005550097557716072,  0.0005481625557877123,  0.0005502061685547233,  0.00029459575307555497,  0.00014886484132148325,  0.00043721433030441403,  0.000413670320995152,  0.0002959157864097506,  0.00015853455988690257,  0.0004453343863133341,  0.0004201193805783987,  0.0002956942480523139,  0.000154652472701855,  0.0004401983751449734,  0.00041690730722621083,  0.00032644715975038707,  0.00027474958915263414,  0.00033714930759742856,  0.0002854687918443233,  0.0003126582014374435,  0.00026466473354958,  0.0009230987634509802,  0.0009231135481968522,  0.0007147723808884621,  0.0007639983668923378,  0.0007424339419230819,  0.0017349586123600602,  0.001724369009025395,  0.001829619170166552,  0.00181943632196635,  0.001722576911561191,  0.001714779413305223,  0.001797390985302627,  0.0017901407554745674,  0.0007801480824127793,  0.0005713902646675706,  0.0007876434247009456,  0.0005512847565114498,  0.0001840846671257168,  0.0018424312584102154,  0.0019792455714195967,  0.0018944081384688616,  0.0020208428613841534,  0.0017271583201363683,  0.0017095598159357905,  8.16941901575774e-05,  8.94474724191241e-05,  9.411648352397606e-05,  8.469070598948747e-05,  7.839938916731626e-05,  0.0001479167549405247,  0.005363330245018005,  0.005363104399293661,  0.005152696743607521,  0.0051524885930120945,  0.005037629511207342,  0.005037416238337755,  0.009204903617501259,  0.009204760193824768,  0.008893216960132122,  0.008893484249711037,  0.0064644357189536095,  0.007239502854645252,  0.00644022086635232,  0.012872133404016495,  0.012872692197561264,  0.012374921701848507,  0.012375611811876297,  0.010288485325872898,  0.009396676905453205,  0.0024629898834973574,  0.0016520884819328785,  0.0021649322006851435,  0.0015257826307788491,  0.003634221153333783,  0.0036343366373330355,  0.0035270294174551964,  0.003565189428627491};
      see_stateTrajGlbX = { 11.959503173828125,  7.59503173828125,  -3.1978402137756348,  -3.1425681114196777,  -3.197739362716675,  -3.1423516273498535,  -9.528078079223633,  -9.530559539794922,  9.474441528320312,  9.634696960449219,  3.1633830070495605,  3.301332473754883,  3.1630241870880127,  3.3010685443878174,  -4.135122299194336,  -4.090638637542725,  -4.036077499389648,  -4.136107921600342,  -4.0906243324279785,  -4.0369133949279785,  -4.135175704956055,  -4.090709209442139,  -4.036061763763428,  -4.1356201171875,  -4.090639591217041,  -4.036446571350098,  -11.736021995544434,  -11.736507415771484,  12.078226089477539,  12.077754974365234,  4.039442539215088,  4.179738998413086,  4.134854793548584,  4.039501190185547,  4.179569721221924,  4.135022163391113,  4.039305686950684,  4.179661750793457,  4.134652137756348,  4.039450168609619,  4.179556846618652,  4.134936332702637,  -5.277444362640381,  -5.234116077423096,  -5.2767333984375,  -5.233985424041748,  -5.277344226837158,  -5.234011173248291,  -5.277492523193359,  -5.234084606170654,  -5.276762962341309,  -5.23396110534668,  -5.277346611022949,  -5.23397970199585,  -5.278066158294678,  -5.234193325042725,  -5.2769775390625,  -5.234065055847168,  -5.2782440185546875,  -5.234116554260254,  -5.278092384338379,  -5.234152317047119,  -5.276876449584961,  -5.234033107757568,  -5.278167247772217,  -5.234073638916016,  5.293721675872803,  5.2944512367248535,  5.293613433837891,  5.293736934661865,  5.294356822967529,  5.293634414672852,  5.293774127960205,  5.294644355773926,  5.293550968170166,  5.2937750816345215,  5.294480800628662,  5.293612957000732,  -6.5228376388549805,  -6.523784160614014,  -6.522416114807129,  -6.522758483886719,  -6.522377014160156,  -6.523830413818359,  -6.522140026092529,  -6.523122310638428,  -6.522790908813477,  -6.523813247680664,  -6.522331237792969,  -6.522630214691162,  6.463504314422607,  6.463461875915527,  6.463441371917725,  6.463399410247803,  6.463505268096924,  6.4634623527526855,  -5.646063804626465,  -5.646759510040283,  7.308294296264648,  7.308235168457031,  7.308289527893066,  5.587844371795654,  5.634014129638672,  5.586319446563721,  5.6327009201049805,  5.587444305419922,  5.633621692657471,  5.585808753967285,  5.632223606109619,  -7.140595436096191,  -7.112336158752441,  -7.1405768394470215,  -7.112340450286865,  -9.865790367126465,  -4.681307315826416,  -4.681473255157471,  -4.629383087158203,  -4.629641532897949,  -5.868945121765137,  -5.869597911834717,  20.69538688659668,  9.634734153747559,  -4.134171009063721,  -11.73445987701416,  4.17788553237915,  -28.841459274291992,  72.92900848388672,  72.9489974975586,  60.216209411621094,  60.237754821777344,  51.934120178222656,  51.95176696777344,  59.58340835571289,  59.62984848022461,  50.34939956665039,  50.30570983886719,  42.56801223754883,  42.60612869262695,  42.045310974121094,  -61.70746612548828,  -61.653907775878906,  -50.35239028930664,  -50.30293273925781,  -42.363704681396484,  -43.04840087890625,  -40.97694396972656,  -41.490596771240234,  -19.989578247070312,  -20.368371963500977,  -70.83287811279297,  -70.8016128540039,  -58.895835876464844,  -61.16007614135742};
      see_stateTrajGlbY = { -9.674687385559082,  -5.825852870941162,  -5.80513858795166,  -5.706777572631836,  -5.805153846740723,  -5.706580638885498,  7.262317180633545,  7.263968467712402,  -7.272950649261475,  -7.395053863525391,  5.793915271759033,  6.049752712249756,  5.793384075164795,  6.049236297607422,  -7.493706226348877,  -7.4160847663879395,  -7.3175950050354,  -7.495087623596191,  -7.416173934936523,  -7.319167137145996,  -7.493647575378418,  -7.416027069091797,  -7.31755256652832,  -7.494329452514648,  -7.41602087020874,  -7.3182806968688965,  8.937713623046875,  8.938345909118652,  -9.279993057250977,  -9.279425621032715,  7.4109978675842285,  7.66998291015625,  7.587550640106201,  7.41112756729126,  7.669658184051514,  7.587777614593506,  7.410874366760254,  7.669923782348633,  7.58736515045166,  7.411137104034424,  7.669693470001221,  7.587737083435059,  -9.548979759216309,  -9.470887184143066,  -9.548210144042969,  -9.470928192138672,  -9.549029350280762,  -9.47093391418457,  -9.54910659790039,  -9.470904350280762,  -9.548295021057129,  -9.470939636230469,  -9.549091339111328,  -9.470949172973633,  -9.549676895141602,  -9.470858573913574,  -9.548419952392578,  -9.470900535583496,  -9.550073623657227,  -9.470895767211914,  -9.549785614013672,  -9.470877647399902,  -9.548337936401367,  -9.470913887023926,  -9.550043106079102,  -9.470913887023926,  9.732877731323242,  9.734561920166016,  9.73266887664795,  9.732895851135254,  9.734354019165039,  9.732701301574707,  9.732797622680664,  9.73459243774414,  9.732284545898438,  9.732856750488281,  9.734354972839355,  9.732502937316895,  -11.779211044311523,  -11.780938148498535,  -11.778491973876953,  -11.779091835021973,  -11.77822494506836,  -11.780876159667969,  -11.777795791625977,  -11.779566764831543,  -11.77900505065918,  -11.780831336975098,  -11.778200149536133,  -11.778690338134766,  11.904093742370605,  11.9041166305542,  11.90412712097168,  11.90415096282959,  11.904092788696289,  11.904114723205566,  5.459832668304443,  5.460421085357666,  -6.683858394622803,  -6.683907985687256,  -6.684068202972412,  4.80575704574585,  4.847445487976074,  4.804513454437256,  4.846315383911133,  4.805428504943848,  4.847031116485596,  4.804092884063721,  4.845802307128906,  6.958586692810059,  6.930490493774414,  6.9585490226745605,  6.930484771728516,  8.02550983428955,  -3.544660806655884,  -3.544747829437256,  -3.5081465244293213,  -3.5083556175231934,  -4.384707450866699,  -4.385272026062012,  -16.707740783691406,  -7.395198822021484,  -7.493884563446045,  8.93632984161377,  7.667088508605957,  23.573389053344727,  -58.14378356933594,  -58.16038513183594,  -48.154090881347656,  -48.165748596191406,  -41.59673309326172,  -41.610557556152344,  -46.685245513916016,  -46.718196868896484,  -39.24609375,  -39.21186828613281,  -33.08420181274414,  -33.147403717041016,  -32.668434143066406,  46.117820739746094,  46.0798225402832,  37.76224136352539,  37.72921371459961,  31.91575050354004,  32.39519119262695,  33.583457946777344,  34.014461517333984,  16.375646591186523,  16.62322235107422,  58.49716567993164,  58.47120666503906,  48.544158935546875,  50.37771224975586};
      see_stateTrajGlbZ = { 31.395000457763672,  -32.64500045776367,  41.573001861572266,  40.87300109863281,  41.573001861572266,  40.87300109863281,  42.12300109863281,  42.12300109863281,  -40.87300109863281,  -41.573001861572266,  -40.323001861572266,  -42.12300109863281,  -40.323001861572266,  -42.12300109863281,  53.52000045776367,  52.970001220703125,  52.27000045776367,  53.52000045776367,  52.970001220703125,  52.27000045776367,  53.52000045776367,  52.970001220703125,  52.27000045776367,  53.52000045776367,  52.970001220703125,  52.27000045776367,  51.720001220703125,  51.720001220703125,  -52.27000045776367,  -52.27000045776367,  -51.720001220703125,  -53.52000045776367,  -52.970001220703125,  -51.720001220703125,  -53.52000045776367,  -52.970001220703125,  -51.720001220703125,  -53.52000045776367,  -52.970001220703125,  -51.720001220703125,  -53.52000045776367,  -52.970001220703125,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  68.06800079345703,  67.51799774169922,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  -68.06800079345703,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  83.88800048828125,  -83.33799743652344,  -83.33799743652344,  -83.33799743652344,  -83.33799743652344,  -83.33799743652344,  -83.33799743652344,  -110.59200286865234,  -110.59200286865234,  140.60000610351562,  140.60000610351562,  140.60000610351562,  -138.8000030517578,  -140.0500030517578,  -138.8000030517578,  -140.0500030517578,  -138.8000030517578,  -140.0500030517578,  -138.8000030517578,  -140.0500030517578,  -140.60000610351562,  -140.0500030517578,  -140.60000610351562,  -140.0500030517578,  -24.950000762939453,  111.84200286865234,  111.84200286865234,  110.59200286865234,  110.59200286865234,  139.35000610351562,  139.35000610351562,  53.88248825073242,  -41.573001861572266,  53.52000045776367,  51.720001220703125,  -53.52000045776367,  -74.1313247680664,  187.5749969482422,  187.7550048828125,  155.73599243164062,  155.91600036621094,  130.26400756835938,  130.44400024414062,  -265.5159912109375,  -265.9159851074219,  -219.38400268554688,  -218.98399353027344,  -186.36050415039062,  -184.71949768066406,  -184.3195037841797,  267.635009765625,  267.2349853515625,  221.10299682617188,  220.7030029296875,  187.87049865722656,  187.4705047607422,  -106.94734191894531,  -106.86043548583984,  -52.10027313232422,  -52.30128860473633,  -186.25599670410156,  -186.0760040283203,  -157.23500061035156,  -157.4149932861328};
      see_stateTrajGlbPx = { 39.982154846191406,  20.144136428833008,  -8.184589385986328,  -11.813593864440918,  -5.951789379119873,  -7.633485794067383,  -24.441490173339844,  -28.791311264038086,  24.070646286010742,  28.250579833984375,  34.67988586425781,  11.00787353515625,  28.34254264831543,  11.620975494384766,  -8.962834358215332,  -15.10213565826416,  -14.817795753479004,  -6.374571800231934,  -9.500391006469727,  -9.084966659545898,  -7.483925819396973,  -10.93773078918457,  -10.9236478805542,  -5.853057861328125,  -7.835663795471191,  -7.658384799957275,  -31.779409408569336,  -46.93708419799805,  49.13154220581055,  33.87992477416992,  7.424851417541504,  7.560487747192383,  7.039959907531738,  8.671110153198242,  9.351542472839355,  8.07282543182373,  7.166080474853516,  7.468014717102051,  6.824861526489258,  8.999288558959961,  9.74183177947998,  8.405508995056152,  -12.111724853515625,  -11.525263786315918,  -8.673627853393555,  -8.164493560791016,  -8.378652572631836,  -8.0927152633667,  -16.713415145874023,  -15.681035995483398,  -10.52535343170166,  -9.854707717895508,  -9.60523796081543,  -9.26362133026123,  -10.789804458618164,  -10.28669548034668,  -7.806000709533691,  -7.402554035186768,  -8.130708694458008,  -8.151240348815918,  -14.308171272277832,  -13.507787704467773,  -9.387253761291504,  -8.78892707824707,  -9.309610366821289,  -9.348767280578613,  6.933831691741943,  6.285653114318848,  6.917183876037598,  6.809549808502197,  6.0486931800842285,  6.83638858795166,  7.92147159576416,  7.831181526184082,  7.953971862792969,  7.1384782791137695,  6.713923931121826,  7.179131031036377,  -8.769672393798828,  -9.502655029296875,  -10.446206092834473,  -11.191804885864258,  -7.696351051330566,  -8.301258087158203,  -8.49558162689209,  -9.101651191711426,  -7.918994903564453,  -8.259015083312988,  -9.202559471130371,  -9.500797271728516,  11.8187255859375,  9.983407974243164,  11.311172485351562,  9.277118682861328,  11.709202766418457,  9.910106658935547,  -2.663613796234131,  -2.7027008533477783,  2.8902781009674072,  2.9869282245635986,  3.1502671241760254,  0.7619412541389465,  0.7796505689620972,  0.8048022389411926,  0.8210440278053284,  0.7622781991958618,  0.7887845635414124,  0.8142665028572083,  0.8398105502128601,  -2.022359848022461,  -2.0029819011688232,  -2.0286357402801514,  -2.00809383392334,  -39.3061408996582,  -1.0055506229400635,  -1.0963633060455322,  -1.0524275302886963,  -1.1412529945373535,  -1.0870596170425415,  -1.057382345199585,  48.34945297241211,  31.039005279541016,  -8.740829467773438,  -30.969114303588867,  8.901070594787598,  -47.49345016479492,  48.74382019042969,  48.700469970703125,  48.686859130859375,  48.78614044189453,  49.81023025512695,  49.79497528076172,  35.33736038208008,  35.25788879394531,  34.69913864135742,  34.72956085205078,  34.443111419677734,  35.195613861083984,  34.36160659790039,  -26.221410751342773,  -26.232385635375977,  -26.810657501220703,  -26.79918098449707,  -26.311445236206055,  -26.979236602783203,  -52.646873474121094,  -52.879878997802734,  -60.02273178100586,  -58.78529357910156,  -49.63499069213867,  -49.661441802978516,  -51.126708984375,  -50.54985427856445};
      see_stateTrajGlbPy = { -32.236961364746094,  -15.512362480163574,  -14.769639015197754,  -21.350543975830078,  -10.728964805603027,  -13.78157901763916,  18.548580169677734,  21.85984230041504,  -18.558923721313477,  -21.768177032470703,  63.714454650878906,  20.271631240844727,  52.08263397216797,  21.398405075073242,  -16.126922607421875,  -27.22907066345215,  -26.717201232910156,  -11.454058647155762,  -17.11142921447754,  -16.362329483032227,  -13.455045700073242,  -19.703123092651367,  -19.680164337158203,  -10.511722564697266,  -14.10165023803711,  -13.783292770385742,  24.090734481811523,  35.61199188232422,  -37.895111083984375,  -26.147994995117188,  13.715087890625,  13.970937728881836,  13.012054443359375,  16.010421752929688,  17.270084381103516,  14.9143648147583,  13.242032051086426,  13.803705215454102,  12.619339942932129,  16.61737823486328,  17.991018295288086,  15.529726028442383,  -21.75792694091797,  -20.702190399169922,  -15.564969062805176,  -14.647746086120605,  -15.03251838684082,  -14.518280029296875,  -30.04041290283203,  -28.1818904876709,  -18.896665573120117,  -17.688581466674805,  -17.238182067871094,  -16.623781204223633,  -19.37251853942871,  -18.4681453704834,  -13.999486923217773,  -13.27303695678711,  -14.582587242126465,  -14.621642112731934,  -25.703638076782227,  -24.264537811279297,  -16.844390869140625,  -15.76684284210205,  -16.70230484008789,  -16.774925231933594,  12.870656967163086,  11.67485237121582,  12.839913368225098,  12.640392303466797,  11.235832214355469,  12.690022468566895,  14.693208694458008,  14.527061462402344,  14.752914428710938,  13.246013641357422,  12.461762428283691,  13.321008682250977,  -15.677408218383789,  -16.99427604675293,  -18.685863494873047,  -20.025083541870117,  -13.747481346130371,  -14.833830833435059,  -15.179661750793457,  -16.26780891418457,  -14.147686958312988,  -14.757756233215332,  -16.450166702270508,  -16.98516082763672,  21.96640968322754,  18.56326675415039,  21.027610778808594,  17.255708694458008,  21.763694763183594,  18.42769432067871,  2.639486789703369,  2.6776726245880127,  -2.567098617553711,  -2.654360771179199,  -2.8016622066497803,  0.7083502411842346,  0.7246838808059692,  0.7463382482528687,  0.7613713145256042,  0.708663821220398,  0.732941210269928,  0.7549158334732056,  0.7783207297325134,  2.043579339981079,  2.024005174636841,  2.0497653484344482,  2.029046058654785,  32.05583190917969,  -0.7198446393013,  -0.7852356433868408,  -0.7554936408996582,  -0.8195514678955078,  -0.7585979700088501,  -0.7373358011245728,  -38.83784103393555,  -23.912744522094727,  -15.736139297485352,  23.476871490478516,  16.4440975189209,  39.09177780151367,  -38.17627716064453,  -38.14161682128906,  -38.351139068603516,  -38.425941467285156,  -39.39361572265625,  -39.380550384521484,  -28.20926856994629,  -28.146648406982422,  -27.49839210510254,  -27.52082633972168,  -27.163925170898438,  -27.754106521606445,  -27.091236114501953,  19.083017349243164,  19.09259605407715,  19.69721031188965,  19.691076278686523,  19.48118782043457,  19.966732025146484,  43.7537727355957,  43.95207595825195,  49.653419494628906,  48.60913848876953,  41.73628616333008,  41.7575798034668,  42.809574127197266,  42.312950134277344};
      see_stateTrajGlbPz = { 102.9832763671875,  -88.19041442871094,  104.47129821777344,  150.95123291015625,  75.90970611572266,  97.46934509277344,  106.4661865234375,  125.39421081542969,  -105.42803192138672,  -123.71235656738281,  -449.241943359375,  -142.85501098632812,  -367.2718200683594,  -150.8206024169922,  114.1419677734375,  192.6598663330078,  189.0045623779297,  81.0625228881836,  121.0994873046875,  115.73025512695312,  95.24909973144531,  139.4384002685547,  139.2493133544922,  74.41570281982422,  99.82036590576172,  97.52598571777344,  138.23590087890625,  204.2759552001953,  -215.14793395996094,  -148.41177368164062,  -96.61841583251953,  -98.30241394042969,  -91.681640625,  -112.79459381103516,  -121.54533386230469,  -105.08798217773438,  -93.24540710449219,  -97.09212493896484,  -88.87865447998047,  -117.02555847167969,  -126.57382202148438,  -109.38743591308594,  154.05982971191406,  146.60020446777344,  110.25863647460938,  103.75537872314453,  106.47212982177734,  102.8404541015625,  212.67169189453125,  199.54002380371094,  133.8407745361328,  125.27928924560547,  122.08378601074219,  117.745361328125,  137.1685791015625,  130.79685974121094,  99.17840576171875,  94.03207397460938,  103.27254486083984,  103.57693481445312,  181.96490478515625,  171.82228088378906,  119.31793212890625,  111.68363189697266,  118.2731704711914,  118.81879425048828,  -90.56780242919922,  -82.11650848388672,  -90.3547592163086,  -88.94398498535156,  -79.0400619506836,  -89.29547119140625,  -103.41069030761719,  -102.20538330078125,  -103.84146118164062,  -93.21522521972656,  -87.67705535888672,  -93.74784851074219,  111.17901611328125,  120.4735107421875,  132.57223510742188,  142.05775451660156,  97.52521514892578,  105.17664337158203,  107.71450805664062,  115.4034194946289,  100.34893035888672,  104.64054107666016,  116.74583435058594,  120.533203125,  -154.4860076904297,  -130.54074096679688,  -147.94781494140625,  -121.37741088867188,  -153.03733825683594,  -129.57017517089844,  -53.014060974121094,  -53.782630920410156,  54.62260818481445,  56.451847076416016,  59.570072174072266,  -19.61507225036621,  -20.09921646118164,  -20.69693374633789,  -21.143489837646484,  -19.63486671447754,  -20.343360900878906,  -20.95399284362793,  -21.63762855529785,  -40.71340560913086,  -40.33354187011719,  -40.836280822753906,  -40.432559967041016,  -101.6891860961914,  23.446203231811523,  25.540164947509766,  24.531314849853516,  26.57099151611328,  25.074169158935547,  24.38584327697754,  124.35140228271484,  -135.91949462890625,  111.37667846679688,  134.80416870117188,  -115.8166732788086,  -123.3208236694336,  123.74313354492188,  123.60462951660156,  123.75849914550781,  123.97946166992188,  127.70701599121094,  127.63870239257812,  -155.8209991455078,  -155.42160034179688,  -153.8428955078125,  -154.02609252929688,  -152.0731964111328,  -155.8533477783203,  -151.6287384033203,  113.05592346191406,  113.14877319335938,  115.50179290771484,  115.50643920898438,  113.44928741455078,  117.13196563720703,  -136.9005584716797,  -137.80079650878906,  -155.91758728027344,  -152.8220977783203,  -129.35536193847656,  -129.4439697265625,  -132.60162353515625,  -132.25082397460938};
      see_q = { -1,  1,  1,  1,  1,  1,  -1,  -1,  1,  1,  -1,  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1,  -1,  1,  1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1,  -1,  -1,  -1,  -1,  -1,  1,  1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  -1,  1,  1,  -1,  -1,  1,  -1,  -1,  -1,  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  -1,  -1,  -1,  -1,  -1,  -1,  1,  1,  1,  1,  1,  1,  1,  1};
      see_hitIdx = { {5, 8, 63, 68},  {4, 7, 10, 14},  {3, 61, 66, 69},  {3, 61, 66, 70},  {3, 61, 64, 69},  {3, 61, 64, 70},  {2, 62, 67, 71},  {2, 62, 65, 71},  {4, 10, 14, 21},  {4, 10, 14, 18},  {0, 9, 13, 17},  {0, 9, 13, 16},  {0, 9, 12, 17},  {0, 9, 12, 16},  {61, 66, 69, 74},  {61, 66, 69, 75},  {61, 66, 69, 73},  {61, 66, 70, 74},  {61, 66, 70, 75},  {61, 66, 70, 73},  {61, 64, 69, 74},  {61, 64, 69, 75},  {61, 64, 69, 73},  {61, 64, 70, 74},  {61, 64, 70, 75},  {61, 64, 70, 73},  {62, 67, 71, 76},  {62, 65, 71, 76},  {10, 14, 21, 27},  {10, 14, 18, 27},  {9, 13, 17, 23},  {9, 13, 17, 25},  {9, 13, 17, 24},  {9, 13, 16, 23},  {9, 13, 16, 25},  {9, 13, 16, 24},  {9, 12, 17, 23},  {9, 12, 17, 25},  {9, 12, 17, 24},  {9, 12, 16, 23},  {9, 12, 16, 25},  {9, 12, 16, 24},  {66, 69, 74, 80},  {66, 69, 74, 81},  {66, 69, 75, 80},  {66, 69, 75, 81},  {66, 69, 73, 80},  {66, 69, 73, 81},  {64, 69, 74, 80},  {64, 69, 74, 81},  {64, 69, 75, 80},  {64, 69, 75, 81},  {64, 69, 73, 80},  {64, 69, 73, 81},  {66, 70, 74, 80},  {66, 70, 74, 81},  {66, 70, 75, 80},  {66, 70, 75, 81},  {66, 70, 73, 80},  {66, 70, 73, 81},  {64, 70, 74, 80},  {64, 70, 74, 81},  {64, 70, 75, 80},  {64, 70, 75, 81},  {64, 70, 73, 80},  {64, 70, 73, 81},  {13, 17, 23, 31},  {13, 17, 25, 31},  {13, 17, 24, 31},  {12, 17, 23, 31},  {12, 17, 25, 31},  {12, 17, 24, 31},  {13, 16, 23, 31},  {13, 16, 25, 31},  {13, 16, 24, 31},  {12, 16, 23, 31},  {12, 16, 25, 31},  {12, 16, 24, 31},  {69, 74, 80, 87},  {69, 74, 81, 87},  {70, 74, 80, 87},  {70, 74, 81, 87},  {69, 75, 80, 87},  {69, 75, 81, 87},  {70, 75, 80, 87},  {70, 75, 81, 87},  {69, 73, 80, 87},  {69, 73, 81, 87},  {70, 73, 80, 87},  {70, 73, 81, 87},  {17, 23, 31, 37},  {16, 23, 31, 37},  {17, 25, 31, 37},  {16, 25, 31, 37},  {17, 24, 31, 37},  {16, 24, 31, 37},  {22, 30, 36, 41},  {22, 30, 35, 41},  {79, 86, 90, 92},  {79, 85, 90, 92},  {79, 84, 90, 92},  {29, 33, 40, 43},  {29, 33, 40, 42},  {29, 33, 39, 43},  {29, 33, 39, 42},  {28, 33, 40, 43},  {28, 33, 40, 42},  {28, 33, 39, 43},  {28, 33, 39, 42},  {30, 36, 41, 45},  {30, 36, 41, 46},  {30, 35, 41, 45},  {30, 35, 41, 46},  {1, 6, 11},  {77, 82, 88},  {78, 82, 88},  {77, 82, 89},  {78, 82, 89},  {82, 88, 91},  {82, 89, 91},  {5, 8, 63, 68, 56, 58},  {4, 7, 10, 14, 21, 18},  {3, 61, 66, 64, 70, 69, 73, 75, 74},  {2, 62, 65, 67, 71, 76},  {0, 9, 13, 12, 17, 16, 23, 24, 25},  {1, 6, 11, 55, 54, 62, 63},  {41},  {42},  {38},  {37},  {29},  {30},  {25},  {24},  {20},  {21},  {12},  {15},  {14},  {51},  {50},  {46},  {45},  {40},  {39},  {67},  {66},  {55},  {54},  {16},  {17},  {10},  {11}};
      ph2_detId = { 411321449,  411321450,  411350101,  411350102,  411571217,  411571218,  411587701,  411587702,  411591797,  411591798,  411616353,  411616354,  411849861,  411849862,  411853969,  411853970,  411874413,  411874414,  412095509,  412095510,  412124349,  412124350,  412365849,  412365850,  412390605,  412390606,  419710001,  419710002,  419734690,  419738813,  419738814,  419959865,  419959866,  419976245,  419976246,  419980341,  419980342,  420005081,  420005082,  420238397,  420238398,  420263157,  420263158,  420484173,  420484174,  420512853,  420512854,  420750413,  420754521,  420754522,  420779101,  420779102,  437521477,  437521478,  437524513,  437524514,  437787717,  437787717,  437787718,  437790753,  437790754,  438572077,  438572077,  438572078,  438837345,  438837346,  439617597,  439617598,  439889029,  439889030};
      ph2_x = { 30.51020050048828,  30.06665802001953,  -51.36579132080078,  -51.36856460571289,  11.797688484191895,  11.762527465820312,  34.9076042175293,  34.17367172241211,  35.65303421020508,  36.27356719970703,  -58.85078430175781,  -62.75593948364258,  42.56789779663086,  42.03622817993164,  42.045082092285156,  42.65549850463867,  -71.30575561523438,  -71.30575561523438,  17.002138137817383,  17.359773635864258,  50.45943069458008,  50.45943069458008,  20.277393341064453,  19.964269638061523,  58.48500061035156,  58.48500061035156,  -30.268720626831055,  -30.33267593383789,  50.212772369384766,  51.84865188598633,  51.84865188598633,  -11.994145393371582,  -12.092185020446777,  -35.266021728515625,  -34.53525924682617,  -36.027305603027344,  -36.6478385925293,  59.56729507446289,  59.5644645690918,  -43.049678802490234,  -42.51227569580078,  72.33489227294922,  72.33489227294922,  -17.679553985595703,  -18.028722763061523,  -51.178165435791016,  -51.18094253540039,  -20.110904693603516,  -21.14443016052246,  -20.96961212158203,  -63.562259674072266,  -63.562259674072266,  19.83856773376465,  19.63016700744629,  -20.369857788085938,  -20.267240524291992,  20.56698226928711,  20.594482421875,  21.33839988708496,  -19.54481315612793,  -19.14232063293457,  -28.476301193237305,  -28.72524070739746,  -28.557058334350586,  28.23151397705078,  28.0576171875,  -41.4915885925293,  -41.15141677856445,  41.439762115478516,  41.48566436767578};
      ph2_y = { -23.605579376220703,  -23.338693618774414,  42.14610290527344,  42.14256286621094,  21.90972900390625,  21.840721130371094,  -27.050264358520508,  -26.447938919067383,  -27.636144638061523,  -28.145404815673828,  48.50764846801758,  51.66998291015625,  -33.0841178894043,  -32.69963073730469,  -32.66823196411133,  -33.1895751953125,  58.889801025390625,  58.889801025390625,  31.82354736328125,  32.383323669433594,  -39.332298278808594,  -39.332298278808594,  38.159019470214844,  37.573204040527344,  -45.90589141845703,  -45.90589141845703,  22.886947631835938,  22.91532325744629,  -40.28700637817383,  -41.52977752685547,  -41.52977752685547,  -21.48033905029297,  -21.672752380371094,  26.61353302001953,  26.007343292236328,  27.180091857910156,  27.68935203552246,  -47.62282943725586,  -47.62632751464844,  32.396060943603516,  32.01976776123047,  -57.65045166015625,  -57.65045166015625,  -31.397897720336914,  -31.962995529174805,  38.414894104003906,  38.411354064941406,  -35.4271240234375,  -37.36289596557617,  -37.03583526611328,  47.43385314941406,  47.43385314941406,  -15.280265808105469,  -15.183039665222168,  16.624059677124023,  16.535945892333984,  -16.612171173095703,  -16.564538955688477,  -17.080642700195312,  14.836277961730957,  14.580804824829102,  22.880088806152344,  23.47719955444336,  23.37933921813965,  -22.7451171875,  -22.606861114501953,  34.0152473449707,  33.72981262207031,  -33.28642272949219,  -33.33147048950195};
      ph2_z = { -133.3105010986328,  -133.7104949951172,  -128.94500732421875,  -128.76499938964844,  -153.97950744628906,  -154.37950134277344,  -152.8695068359375,  -152.46949768066406,  -156.02049255371094,  -155.62049865722656,  -157.23500061035156,  -157.4149932861328,  -186.36050415039062,  -185.9604949951172,  -184.3195037841797,  -184.71949768066406,  -186.25599670410156,  -186.0760040283203,  -222.63949584960938,  -222.239501953125,  -219.38400268554688,  -218.98399353027344,  -267.1304931640625,  -267.5304870605469,  -265.9159851074219,  -265.5159912109375,  132.20050048828125,  131.80050659179688,  131.80050659179688,  130.26400756835938,  130.44400024414062,  152.8695068359375,  152.46949768066406,  153.97950744628906,  154.37950134277344,  157.1304931640625,  157.53050231933594,  155.91600036621094,  155.73599243164062,  187.4705047607422,  187.87049865722656,  187.5749969482422,  187.7550048828125,  223.74949645996094,  224.14950561523438,  220.7030029296875,  221.10299682617188,  263.9794921875,  266.0205078125,  265.6205139160156,  267.2349853515625,  267.635009765625,  -86.24923706054688,  -86.7442626953125,  -52.3003044128418,  -51.91516876220703,  53.512393951416016,  53.512393951416016,  53.453330993652344,  85.501220703125,  85.23539733886719,  -73.98045349121094,  -73.76063537597656,  -74.27043151855469,  73.12118530273438,  72.77932739257812,  -106.85971069335938,  -106.81586456298828,  107.15310668945312,  106.81586456298828};

      std::cout << "Here is: model_state->fLST->run()" << std::endl;
      model_state->fLST->run(
          stream,
          verbose,
          see_px,
          see_py,
          see_pz,
          see_dxy,
          see_dz,
          see_ptErr,
          see_etaErr,
          see_stateTrajGlbX,
          see_stateTrajGlbY,
          see_stateTrajGlbZ,
          see_stateTrajGlbPx,
          see_stateTrajGlbPy,
          see_stateTrajGlbPz,
          see_q,
          see_hitIdx,
          ph2_detId,
          ph2_x,
          ph2_y,
          ph2_z
      );
    };

    // We only need to produce an output if it was requested.
    std::cout << "THIS IS requested_output_count " << requested_output_count << std::endl;
    if (requested_output_count > 0) {
      // This backend simply copies the input tensor to the output
      // tensor. The input tensor contents are available in one or
      // more contiguous buffers. To do the copy we:
      //
      //   1. Create an output tensor in the response.
      //
      //   2. Allocate appropriately sized buffer in the output
      //      tensor.
      //
      //   3. Iterate over the input tensor buffers and copy the
      //      contents into the output buffer.
      TRITONBACKEND_Response* response = responses[r];
      //const void* output_tmp = model_state->fBSTest->getOutput(); // COMMENT THIS
      //uint64_t output_buffer_byte_size = model_state->fBSTest->getSize();//7200000;//8146596;//reinterpret_cast<uint32_t*>(output_buffer)[0]*4*sizeof(uint32_t); 

      //const void* output_tmp = lst_output->data();
      std::cout << "Testing this........." << std::endl;
      const void* output_tmp = model_state->fLST->lst_output.data();

      //uint64_t output_buffer_byte_size = model_state->fLST->lst_outsize;
      uint64_t output_buffer_byte_size = model_state->fLST->lst_outsize;
      std::cout << "LST output size (bytes): " << model_state->fLST->lst_outsize << std::endl;
      std::cout << "LST output_buffer_byte_size: " << output_buffer_byte_size << std::endl;
      std::cout << "Just print OUTPUT TMP??" << output_tmp << std::endl;
      int first_item_lst_output;
      std::memcpy(&first_item_lst_output, output_tmp, sizeof(int));
      std::cout << "first_item lst output???????? " << first_item_lst_output << std::endl;
      int first_item;
      std::memcpy(&first_item, output_tmp, sizeof(int));
      std::cout << "first_item???????? " << first_item << std::endl;

      int64_t* output_shape = new int64_t[1];
      //output_shape[0] = 1;
      output_shape[0] = output_buffer_byte_size;
      uint32_t output_dims_count = 1;
      // Step 1. Input and output have same datatype and shape...
      TRITONBACKEND_Output* output;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
				       response, &output, requested_output_name, TRITONSERVER_TYPE_INT8,
				       output_shape, output_dims_count));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create response output, error response sent")
                .c_str());
        continue;
      }
      // Step 2. Get the output buffer. We request a buffer in CPU
      // memory but we have to handle any returned type. If we get
      // back a buffer in GPU memory we just fail the request.
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, output_buffer_byte_size, &output_memory_type,
              &output_memory_type_id));
      if ((responses[r] == nullptr) ||
          (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to create output buffer in CPU memory"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create output buffer in CPU memory, error response "
             "sent")
                .c_str());
        continue;
      }
      std::cout << "DID WE MAKE IT HERE? 1" << std::endl;
      memcpy(output_buffer,output_tmp,output_buffer_byte_size);
      std::cout << "output_buffer " << output_buffer << std::endl;
      std::cout << "&output_buffer " << &output_buffer << std::endl;
      std::cout << "DID WE MAKE IT HERE? 2" << std::endl;

      int test1 = -999.9;
      memcpy(&test1,output_buffer,sizeof(int));
      std::cout << "Test1: " << test1 << std::endl;

      int test2 = -999.9;
      memcpy(&test2,output_tmp,sizeof(int));
      std::cout << "Test2: " << test2 << std::endl;

      /*
      int8_t *output_buffer2 = new int8_t[output_buffer_byte_size];
      memcpy(output_buffer2,output_tmp,output_buffer_byte_size);

      
      uint32_t pdigi_[150000];
      uint32_t rawIdArr_[150000];
      uint16_t adc_ [150000];
      int32_t  clus_[150000];
      uint32_t hits_[2001];
      float    pos_ [4*35000];

      unsigned int pCount = 0; 
      uint32_t nHits = 0;
      uint32_t nDigis    = 0; //output[pCount]; pCount++;                                                                                                                  
      std::memcpy(&nHits,output_buffer2+pCount,sizeof(uint32_t)); pCount += 4;
      std::memcpy(hits_, output_buffer2+pCount,(2000+1)*sizeof(uint32_t));    pCount += 4*(2000+1);
      std::memcpy(pos_,  output_buffer2+pCount,4*nHits*sizeof(float));         pCount += 4*4*nHits;
      std::memcpy(&nDigis,output_buffer2+pCount,sizeof(uint32_t)); pCount += 4;
      std::memcpy(pdigi_,   output_buffer2+pCount,nDigis*sizeof(uint32_t)); pCount += 4*nDigis;
      std::memcpy(rawIdArr_,output_buffer2+pCount,nDigis*sizeof(uint32_t)); pCount += 4*nDigis;
      std::memcpy(adc_,     output_buffer2+pCount,nDigis*sizeof(uint16_t)); pCount += 2*nDigis;
      std::memcpy(clus_,    output_buffer2+pCount,nDigis*sizeof(int32_t));  pCount += 4*nDigis;
      std::cout << "---> digis " << nDigis  << " --" << pdigi_[0] << " -- " << rawIdArr_[0] << " -- " << adc_[0] << " -- " << adc_[nDigis-1] << " -- " << clus_[0] << " -- " << clus_[1] << " -- " << clus_[nDigis-1] << std::endl;
      */
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to get input buffer in CPU memory, error response "
             "sent")
                .c_str());
        continue;
      }
    }
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");


    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }
  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::identity
