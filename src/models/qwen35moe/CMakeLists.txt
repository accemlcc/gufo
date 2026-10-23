# Qwen3.5-MoE text model package (qwen35moe).

# Config and weight binding only for now: the trunk graph (Gated DeltaNet and
# full-attention layers with a softmax top-k mixture in every layer) is bound
# and validated, the HIP engine that executes it is the next step. The kernels
# it will reuse already exist: the SSM and GQA path of `qwen/hip`, and the
# grouped expert projection of `kolibri1` / `qwen38_flash_next`.
add_library(gufo_qwen35moe STATIC config.cpp weights.cpp)
target_include_directories(gufo_qwen35moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen35moe PUBLIC gufo_core Threads::Threads)

# The device path: the upload of the weights, the per-row norms, the router
# whose group table a grouped projection consumes, and the end of a mixture
# block. Host code that calls the kernels above. The grouped expert
# projection, the attention paths and the executor follow in later steps.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen35moe PRIVATE kernels/rocm/device_model.cpp
                      kernels/rocm/route.hip.cpp)
  set_source_files_properties(kernels/rocm/device_model.cpp
                              kernels/rocm/route.hip.cpp
    PROPERTIES LANGUAGE HIP)
endif()

# The block arithmetic every layer runs several times: the norms, the residual
# add, the expert activation and the end of a mixture. Strict floating point,
# because a row has to round the same whatever else shares its launch and a
# batch boundary must not move its value.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen35moe PRIVATE kernels/rocm/blocks.hip.cpp)
  set_source_files_properties(kernels/rocm/blocks.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/blocks.hip.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()
# The scalar oracle decodes every weight on the fly; unoptimized it is unusable
# even as a reference, so it keeps -O2 in every configuration.
add_library(gufo_qwen35moe_reference STATIC EXCLUDE_FROM_ALL reference.cpp)
target_link_libraries(gufo_qwen35moe_reference PUBLIC gufo_qwen35moe)
set_source_files_properties(reference.cpp PROPERTIES COMPILE_OPTIONS "-O2")
