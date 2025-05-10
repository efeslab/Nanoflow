set(DEVICE_MAXPOOL_BWD_INSTANCES)
list(APPEND DEVICE_MAXPOOL_BWD_INSTANCES device_max_pool_bwd_f16_instance.cpp
                                         device_max_pool_bwd_bf16_instance.cpp
                                         device_max_pool_bwd_f32_instance.cpp)
add_instance_library(device_max_pool_bwd_instance ${DEVICE_MAXPOOL_BWD_INSTANCES})
