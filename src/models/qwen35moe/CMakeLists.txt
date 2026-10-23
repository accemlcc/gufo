# Qwen3.5-MoE text model package (qwen35moe).
#
# Config, weight binding and the scalar oracle plus the HIP engine: the trunk
# graph (Gated DeltaNet and full-attention layers with a softmax top-k mixture
# in every layer) runs on the device through grouped projections, a fused
# QK-norm/RoPE/split/cache write, online GQA attention, the ported gated-delta
# recurrence and the mixture kernels, all validated against the oracle.
add_library(gufo_qwen35moe STATIC config.cpp weights.cpp)
target_include_directories(gufo_qwen35moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen35moe PUBLIC gufo_core Threads::Threads)

# The device path: the upload of the weights, the per-row norms, the router
# whose group table a grouped projection consumes, the fused QK-norm/RoPE
# gate-split and cache write, the online GQA attention, the gated-delta
# recurrence, the end of a mixture block, and the executor plus engine that
# run the trunk graph on them.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen35moe PRIVATE kernels/rocm/device_model.cpp
                      kernels/rocm/route.hip.cpp
                      kernels/rocm/qkv_post.hip.cpp
                      kernels/rocm/attention.hip.cpp
                      kernels/rocm/ssm.hip.cpp
                      kernels/rocm/executor.cpp
                      engine.cpp)
  set_source_files_properties(kernels/rocm/device_model.cpp
                              kernels/rocm/route.hip.cpp
                              kernels/rocm/qkv_post.hip.cpp
                              kernels/rocm/attention.hip.cpp
                              kernels/rocm/ssm.hip.cpp
                              kernels/rocm/executor.cpp
                              engine.cpp
    PROPERTIES LANGUAGE HIP)
endif()

# The grouped projection over the experts. Strict floating point, because a
# slot has to round the same whoever else shares the launch and a batch
# boundary must not move its value.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen35moe PRIVATE kernels/rocm/gemv.hip.cpp)
  set_source_files_properties(kernels/rocm/gemv.hip.cpp PROPERTIES
    LANGUAGE HIP COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
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
