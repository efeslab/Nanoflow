/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <nvbench/cupti_profiler.cuh>

#include <nvbench/detail/throw.cuh>
#include <nvbench/device_info.cuh>

#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>

#include <fmt/format.h>

#include <stdexcept>
#include <type_traits>

namespace nvbench::detail
{

namespace
{

void cupti_call(const CUptiResult status)
{
  if (status != CUPTI_SUCCESS)
  {
    const char *errstr{};
    cuptiGetResultString(status, &errstr);

    NVBENCH_THROW(std::runtime_error, "CUPTI call returned error: {}", errstr);
  }
}

void nvpw_call(const NVPA_Status status)
{
  if (status != NVPA_STATUS_SUCCESS)
  {
    NVBENCH_THROW(std::runtime_error, "NVPW call returned error: {}", static_cast<std::underlying_type_t<NVPA_Status>>(status));
  }
}

} // namespace

cupti_profiler::cupti_profiler(nvbench::device_info device, std::vector<std::string> &&metric_names)
    : m_metric_names(metric_names)
    , m_device(device)
{
  initialize_profiler();
  initialize_chip_name();
  initialize_availability_image();
  initialize_nvpw();
  initialize_config_image();
  initialize_counter_data_prefix_image();
  initialize_counter_data_image();

  m_available = true;
}

cupti_profiler::cupti_profiler(cupti_profiler &&rhs) noexcept
    : m_device(rhs.m_device.get_id(), rhs.m_device.get_cuda_device_prop())
{
  (*this) = std::move(rhs);
}

cupti_profiler &cupti_profiler::operator=(cupti_profiler &&rhs) noexcept
{
  m_device              = rhs.m_device;
  m_available           = rhs.m_available;
  m_chip_name           = std::move(rhs.m_chip_name);
  m_metric_names        = std::move(rhs.m_metric_names);
  m_data_image_prefix   = std::move(rhs.m_data_image_prefix);
  m_config_image        = std::move(rhs.m_config_image);
  m_data_image          = std::move(rhs.m_data_image);
  m_data_scratch_buffer = std::move(rhs.m_data_scratch_buffer);
  m_availability_image  = std::move(rhs.m_availability_image);

  rhs.m_available = false;

  return *this;
}

void cupti_profiler::initialize_profiler()
{
  if (!m_device.is_cupti_supported())
  {
    NVBENCH_THROW(std::runtime_error,
                  "Device: {} isn't supported (CC {})",
                  m_device.get_id(),
                  m_device.get_sm_version());
  }

  CUpti_Profiler_Initialize_Params params{};
  params.structSize = CUpti_Profiler_Initialize_Params_STRUCT_SIZE;
  cupti_call(cuptiProfilerInitialize(&params));
}

void cupti_profiler::initialize_chip_name()
{
  CUpti_Device_GetChipName_Params params{};
  params.structSize  = CUpti_Device_GetChipName_Params_STRUCT_SIZE;
  params.deviceIndex = static_cast<size_t>(m_device.get_id());
  cupti_call(cuptiDeviceGetChipName(&params));

  m_chip_name = std::string(params.pChipName);
}

void cupti_profiler::initialize_availability_image()
{
  CUpti_Profiler_GetCounterAvailability_Params params{};

  params.structSize = CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE;
  params.ctx        = m_device.get_context();

  cupti_call(cuptiProfilerGetCounterAvailability(&params));

  m_availability_image.clear();
  m_availability_image.resize(params.counterAvailabilityImageSize);
  params.pCounterAvailabilityImage = m_availability_image.data();

  cupti_call(cuptiProfilerGetCounterAvailability(&params));
}

void cupti_profiler::initialize_nvpw()
{
  NVPW_InitializeHost_Params params{};
  params.structSize = NVPW_InitializeHost_Params_STRUCT_SIZE;
  nvpw_call(NVPW_InitializeHost(&params));
}

namespace
{

class eval_request
{
  NVPW_MetricsEvaluator *evaluator_ptr;

public:
  eval_request(NVPW_MetricsEvaluator *evaluator_ptr, const std::string &metric_name)
      : evaluator_ptr(evaluator_ptr)
  {
    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params params = {};

    params.structSize =
      NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE;
    params.pMetricsEvaluator           = evaluator_ptr;
    params.pMetricName                 = metric_name.c_str();
    params.pMetricEvalRequest          = &request;
    params.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;

    nvpw_call(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&params));
  }

  [[nodiscard]] std::vector<const char *> get_raw_dependencies()
  {
    std::vector<const char *> raw_dependencies;

    NVPW_MetricsEvaluator_GetMetricRawDependencies_Params params{};

    params.structSize          = NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE;
    params.pMetricsEvaluator   = evaluator_ptr;
    params.pMetricEvalRequests = &request;
    params.numMetricEvalRequests       = 1;
    params.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
    params.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);

    nvpw_call(NVPW_MetricsEvaluator_GetMetricRawDependencies(&params));

    raw_dependencies.resize(params.numRawDependencies);
    params.ppRawDependencies = raw_dependencies.data();

    nvpw_call(NVPW_MetricsEvaluator_GetMetricRawDependencies(&params));

    return raw_dependencies;
  }

  NVPW_MetricEvalRequest request;
};

class metric_evaluator
{
  bool initialized{};
  NVPW_MetricsEvaluator *evaluator_ptr;
  std::vector<std::uint8_t> scratch_buffer;

public:
  metric_evaluator(const std::string &chip_name,
                   const std::uint8_t *counter_availability_image = nullptr,
                   const std::uint8_t *counter_data_image         = nullptr,
                   const std::size_t counter_data_image_size      = 0)
  {
    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params scratch_buffer_param{};

    scratch_buffer_param.structSize =
      NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE;
    scratch_buffer_param.pChipName                 = chip_name.c_str();
    scratch_buffer_param.pCounterAvailabilityImage = counter_availability_image;

    nvpw_call(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&scratch_buffer_param));

    scratch_buffer.resize(scratch_buffer_param.scratchBufferSize);

    NVPW_CUDA_MetricsEvaluator_Initialize_Params evaluator_params{};

    evaluator_params.structSize        = NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE;
    evaluator_params.scratchBufferSize = scratch_buffer.size();
    evaluator_params.pScratchBuffer    = scratch_buffer.data();
    evaluator_params.pChipName         = chip_name.c_str();
    evaluator_params.pCounterAvailabilityImage = counter_availability_image;
    evaluator_params.pCounterDataImage         = counter_data_image;
    evaluator_params.counterDataImageSize      = counter_data_image_size;

    nvpw_call(NVPW_CUDA_MetricsEvaluator_Initialize(&evaluator_params));

    evaluator_ptr = evaluator_params.pMetricsEvaluator;
    initialized   = true;
  }

  ~metric_evaluator()
  {
    if (initialized)
    {
      NVPW_MetricsEvaluator_Destroy_Params params{};

      params.structSize        = NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE;
      params.pMetricsEvaluator = evaluator_ptr;

      nvpw_call(NVPW_MetricsEvaluator_Destroy(&params));
    }
  }

  [[nodiscard]] eval_request create_request(const std::string &metric_name)
  {
    return {evaluator_ptr, metric_name};
  }

  [[nodiscard]] operator NVPW_MetricsEvaluator *() const { return evaluator_ptr; }
};

} // namespace

namespace
{

[[nodiscard]] std::vector<NVPA_RawMetricRequest>
get_raw_metric_requests(const std::string &chip_name,
                        const std::vector<std::string> &metric_names,
                        const std::uint8_t *counter_availability_image = nullptr)
{
  metric_evaluator evaluator(chip_name, counter_availability_image);

  std::vector<const char *> raw_metric_names;
  raw_metric_names.reserve(metric_names.size());

  for (auto &metric_name : metric_names)
  {
    for (auto &raw_dependency : evaluator.create_request(metric_name).get_raw_dependencies())
    {
      raw_metric_names.push_back(raw_dependency);
    }
  }

  std::vector<NVPA_RawMetricRequest> raw_requests;
  raw_requests.reserve(raw_metric_names.size());

  for (auto &raw_name : raw_metric_names)
  {
    NVPA_RawMetricRequest metricRequest{};
    metricRequest.structSize    = NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE;
    metricRequest.pMetricName   = raw_name;
    metricRequest.isolated      = true;
    metricRequest.keepInstances = true;
    raw_requests.push_back(metricRequest);
  }

  return raw_requests;
}

class metrics_config
{
  bool initialized{};

  void create(const std::string &chip_name, const std::uint8_t *availability_image)
  {
    NVPW_CUDA_RawMetricsConfig_Create_V2_Params params{};

    params.structSize                = NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE;
    params.activityKind              = NVPA_ACTIVITY_KIND_PROFILER;
    params.pChipName                 = chip_name.c_str();
    params.pCounterAvailabilityImage = availability_image;

    nvpw_call(NVPW_CUDA_RawMetricsConfig_Create_V2(&params));

    raw_metrics_config = params.pRawMetricsConfig;
    initialized        = true;
  }

  void set_availability_image(const std::uint8_t *availability_image)
  {
    NVPW_RawMetricsConfig_SetCounterAvailability_Params params{};

    params.structSize        = NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE;
    params.pRawMetricsConfig = raw_metrics_config;
    params.pCounterAvailabilityImage = availability_image;

    nvpw_call(NVPW_RawMetricsConfig_SetCounterAvailability(&params));
  }

  void begin_config_group()
  {
    NVPW_RawMetricsConfig_BeginPassGroup_Params params{};

    params.structSize        = NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE;
    params.pRawMetricsConfig = raw_metrics_config;

    nvpw_call(NVPW_RawMetricsConfig_BeginPassGroup(&params));
  }

  void add_metrics(const std::vector<NVPA_RawMetricRequest> &raw_metric_requests)
  {
    NVPW_RawMetricsConfig_AddMetrics_Params params{};

    params.structSize         = NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE;
    params.pRawMetricsConfig  = raw_metrics_config;
    params.pRawMetricRequests = raw_metric_requests.data();
    params.numMetricRequests  = raw_metric_requests.size();

    nvpw_call(NVPW_RawMetricsConfig_AddMetrics(&params));
  }

  void end_config_group()
  {
    NVPW_RawMetricsConfig_EndPassGroup_Params params{};

    params.structSize        = NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE;
    params.pRawMetricsConfig = raw_metrics_config;

    nvpw_call(NVPW_RawMetricsConfig_EndPassGroup(&params));
  }

  void generate()
  {
    NVPW_RawMetricsConfig_GenerateConfigImage_Params params{};

    params.structSize        = NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE;
    params.pRawMetricsConfig = raw_metrics_config;

    nvpw_call(NVPW_RawMetricsConfig_GenerateConfigImage(&params));
  }

public:
  metrics_config(const std::string &chip_name,
                 const std::vector<NVPA_RawMetricRequest> &raw_metric_requests,
                 const std::uint8_t *availability_image)
  {
    create(chip_name, availability_image);
    set_availability_image(availability_image);

    begin_config_group();
    add_metrics(raw_metric_requests);
    end_config_group();
    generate();
  }

  [[nodiscard]] std::vector<std::uint8_t> get_config_image()
  {
    NVPW_RawMetricsConfig_GetConfigImage_Params params{};

    params.structSize        = NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE;
    params.pRawMetricsConfig = raw_metrics_config;
    params.bytesAllocated    = 0;
    params.pBuffer           = nullptr;

    nvpw_call(NVPW_RawMetricsConfig_GetConfigImage(&params));

    std::vector<std::uint8_t> config_image(params.bytesCopied);
    params.bytesAllocated = config_image.size();
    params.pBuffer        = config_image.data();

    nvpw_call(NVPW_RawMetricsConfig_GetConfigImage(&params));
    return config_image;
  }

  ~metrics_config()
  {
    if (initialized)
    {
      NVPW_RawMetricsConfig_Destroy_Params params{};

      params.structSize        = NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE;
      params.pRawMetricsConfig = raw_metrics_config;

      NVPW_RawMetricsConfig_Destroy(&params);
    }
  }

  NVPA_RawMetricsConfig *raw_metrics_config;
};

} // namespace

void cupti_profiler::initialize_config_image()
{
  m_config_image = metrics_config(m_chip_name,
                                  get_raw_metric_requests(m_chip_name,
                                                          m_metric_names,
                                                          m_availability_image.data()),
                                  m_availability_image.data())
                     .get_config_image();
}

namespace
{

class counter_data_builder
{
  bool initialized{};

public:
  counter_data_builder(const std::string &chip_name, const std::uint8_t *pCounterAvailabilityImage)
  {
    NVPW_CUDA_CounterDataBuilder_Create_Params params{};

    params.structSize                = NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE;
    params.pChipName                 = chip_name.c_str();
    params.pCounterAvailabilityImage = pCounterAvailabilityImage;

    nvpw_call(NVPW_CUDA_CounterDataBuilder_Create(&params));

    builder     = params.pCounterDataBuilder;
    initialized = true;
  }

  ~counter_data_builder()
  {
    if (initialized)
    {
      NVPW_CounterDataBuilder_Destroy_Params params{};

      params.structSize          = NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE;
      params.pCounterDataBuilder = builder;

      NVPW_CounterDataBuilder_Destroy(&params);
    }
  }

  NVPA_CounterDataBuilder *builder;
};

} // namespace

void cupti_profiler::initialize_counter_data_prefix_image()
{
  const std::uint8_t *counter_availability_image = nullptr;

  std::vector<NVPA_RawMetricRequest> raw_metric_requests =
    get_raw_metric_requests(m_chip_name, m_metric_names, counter_availability_image);

  counter_data_builder data_builder(m_chip_name, counter_availability_image);

  {
    NVPW_CounterDataBuilder_AddMetrics_Params params{};

    params.structSize          = NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE;
    params.pCounterDataBuilder = data_builder.builder;
    params.pRawMetricRequests  = raw_metric_requests.data();
    params.numMetricRequests   = raw_metric_requests.size();

    nvpw_call(NVPW_CounterDataBuilder_AddMetrics(&params));
  }

  {
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params params{};

    params.structSize          = NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE;
    params.pCounterDataBuilder = data_builder.builder;
    params.bytesAllocated      = 0;
    params.pBuffer             = nullptr;

    nvpw_call(NVPW_CounterDataBuilder_GetCounterDataPrefix(&params));

    m_data_image_prefix.resize(params.bytesCopied);
    params.bytesAllocated = m_data_image_prefix.size();
    params.pBuffer        = m_data_image_prefix.data();

    nvpw_call(NVPW_CounterDataBuilder_GetCounterDataPrefix(&params));
  }
}

namespace
{

[[nodiscard]] std::size_t
get_counter_data_image_size(CUpti_Profiler_CounterDataImageOptions *options)
{
  CUpti_Profiler_CounterDataImage_CalculateSize_Params params{};

  params.structSize = CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE;
  params.pOptions   = options;
  params.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  cupti_call(cuptiProfilerCounterDataImageCalculateSize(&params));
  return params.counterDataImageSize;
}

} // namespace

void cupti_profiler::initialize_counter_data_image()
{
  CUpti_Profiler_CounterDataImageOptions counter_data_image_options;

  counter_data_image_options.pCounterDataPrefix    = &m_data_image_prefix[0];
  counter_data_image_options.counterDataPrefixSize = m_data_image_prefix.size();
  counter_data_image_options.maxNumRanges          = m_num_ranges;
  counter_data_image_options.maxNumRangeTreeNodes  = m_num_ranges;
  counter_data_image_options.maxRangeNameLength    = 64;

  m_data_image.resize(get_counter_data_image_size(&counter_data_image_options));

  {
    CUpti_Profiler_CounterDataImage_Initialize_Params params{};

    params.structSize = CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE;
    params.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    params.pOptions                      = &counter_data_image_options;
    params.counterDataImageSize          = m_data_image.size();

    params.pCounterDataImage = &m_data_image[0];
    cupti_call(cuptiProfilerCounterDataImageInitialize(&params));
  }

  {
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params params{};

    params.structSize =
      CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE;
    params.counterDataImageSize = m_data_image.size();
    params.pCounterDataImage    = &m_data_image[0];

    cupti_call(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&params));

    m_data_scratch_buffer.resize(params.counterDataScratchBufferSize);
  }

  {
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params params{};

    params.structSize = CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE;
    params.counterDataImageSize         = m_data_image.size();
    params.pCounterDataImage            = &m_data_image[0];
    params.counterDataScratchBufferSize = m_data_scratch_buffer.size();
    params.pCounterDataScratchBuffer    = &m_data_scratch_buffer[0];

    cupti_call(cuptiProfilerCounterDataImageInitializeScratchBuffer(&params));
  }
}

cupti_profiler::~cupti_profiler()
{
  if (is_initialized())
  {
    CUpti_Profiler_DeInitialize_Params params{};
    params.structSize = CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE;
    cuptiProfilerDeInitialize(&params);
  }
}

bool cupti_profiler::is_initialized() const { return m_available; }

void cupti_profiler::prepare_user_loop()
{
  {
    CUpti_Profiler_BeginSession_Params params{};

    params.structSize                   = CUpti_Profiler_BeginSession_Params_STRUCT_SIZE;
    params.ctx                          = nullptr;
    params.counterDataImageSize         = m_data_image.size();
    params.pCounterDataImage            = &m_data_image[0];
    params.counterDataScratchBufferSize = m_data_scratch_buffer.size();
    params.pCounterDataScratchBuffer    = &m_data_scratch_buffer[0];

    // Each kernel is going to produce its own set of metrics
    params.range              = CUPTI_UserRange;
    params.replayMode         = CUPTI_UserReplay;
    params.maxRangesPerPass   = m_num_ranges;
    params.maxLaunchesPerPass = m_num_ranges;

    cupti_call(cuptiProfilerBeginSession(&params));
  }

  {
    CUpti_Profiler_SetConfig_Params params{};

    params.structSize       = CUpti_Profiler_SetConfig_Params_STRUCT_SIZE;
    params.pConfig          = &m_config_image[0];
    params.configSize       = m_config_image.size();
    params.minNestingLevel  = 1;
    params.numNestingLevels = 1;
    params.passIndex        = 0;

    cupti_call(cuptiProfilerSetConfig(&params));
  }
}

void cupti_profiler::start_user_loop()
{
  {
    CUpti_Profiler_BeginPass_Params params{};
    params.structSize = CUpti_Profiler_BeginPass_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerBeginPass(&params));
  }

  {
    CUpti_Profiler_EnableProfiling_Params params{};
    params.structSize = CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerEnableProfiling(&params));
  }

  {
    CUpti_Profiler_PushRange_Params params{};

    std::string rangeName = "nvbench";

    params.structSize = CUpti_Profiler_PushRange_Params_STRUCT_SIZE;
    params.pRangeName = rangeName.c_str();

    cupti_call(cuptiProfilerPushRange(&params));
  }
}

void cupti_profiler::stop_user_loop()
{
  {
    CUpti_Profiler_PopRange_Params params{};
    params.structSize = CUpti_Profiler_PopRange_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerPopRange(&params));
  }

  {
    CUpti_Profiler_DisableProfiling_Params params{};
    params.structSize = CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerDisableProfiling(&params));
  }
}

bool cupti_profiler::is_replay_required()
{
  CUpti_Profiler_EndPass_Params params{};
  params.structSize = CUpti_Profiler_EndPass_Params_STRUCT_SIZE;
  cupti_call(cuptiProfilerEndPass(&params));

  return !params.allPassesSubmitted;
}

void cupti_profiler::process_user_loop()
{
  {
    CUpti_Profiler_FlushCounterData_Params params{};
    params.structSize = CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerFlushCounterData(&params));
  }

  {
    CUpti_Profiler_UnsetConfig_Params params{};
    params.structSize = CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerUnsetConfig(&params));
  }

  {
    CUpti_Profiler_EndSession_Params params{};
    params.structSize = CUpti_Profiler_EndSession_Params_STRUCT_SIZE;
    cupti_call(cuptiProfilerEndSession(&params));
  }
}

std::vector<double> cupti_profiler::get_counter_values()
{
  metric_evaluator evaluator(m_chip_name,
                             m_availability_image.data(),
                             m_data_image.data(),
                             m_data_image.size());

  {
    NVPW_CounterData_GetNumRanges_Params params{};

    params.structSize        = NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE;
    params.pCounterDataImage = m_data_image.data();
    nvpw_call(NVPW_CounterData_GetNumRanges(&params));

    if (params.numRanges != 1)
    {
      NVBENCH_THROW(std::runtime_error, "{}", "Something's gone wrong, one range is expected");
    }
  }

  std::size_t range_id{}; // there's only one range
  std::size_t result_id{};
  std::vector<double> result(m_metric_names.size());

  for (const std::string &metric_name : m_metric_names)
  {
    eval_request request = evaluator.create_request(metric_name);

    {
      NVPW_MetricsEvaluator_SetDeviceAttributes_Params params{};

      params.structSize           = NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE;
      params.pMetricsEvaluator    = evaluator;
      params.pCounterDataImage    = m_data_image.data();
      params.counterDataImageSize = m_data_image.size();

      nvpw_call(NVPW_MetricsEvaluator_SetDeviceAttributes(&params));
    }

    {
      NVPW_MetricsEvaluator_EvaluateToGpuValues_Params params{};

      params.structSize            = NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE;
      params.pMetricsEvaluator     = evaluator;
      params.pMetricEvalRequests   = &request.request;
      params.numMetricEvalRequests = 1;
      params.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
      params.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
      params.pCounterDataImage           = m_data_image.data();
      params.counterDataImageSize        = m_data_image.size();
      params.rangeIndex                  = range_id;
      params.isolated                    = true;
      params.pMetricValues               = &result[result_id++];

      nvpw_call(NVPW_MetricsEvaluator_EvaluateToGpuValues(&params));
    }
  }

  return result;
}

} // namespace nvbench::detail
