# Kolibri 1 text model package (sigmoid_logit_add MoE with sliding-window and
# global attention layers). The scalar oracle is built only by explicit
# validation targets.
add_library(gufo_kolibri1 STATIC chat_template.cpp config.cpp tokenizer.cpp
                      weights.cpp)
target_include_directories(gufo_kolibri1 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_kolibri1 PUBLIC gufo_core Threads::Threads)

# The device path: attention over the binary16 cache on the matrix cores, and
# the per-head norm, rotation and cache write in front of it.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_kolibri1 PRIVATE kernels/rocm/attention.hip.cpp
                     kernels/rocm/qkv_post.hip.cpp)
  set_source_files_properties(kernels/rocm/attention.hip.cpp
                              kernels/rocm/qkv_post.hip.cpp
    PROPERTIES LANGUAGE HIP)

  # The device-side model: the upload of the weights, the scratch arena and the
  # order of a layer. Host code that calls the kernels above.
  target_sources(gufo_kolibri1 PRIVATE kernels/rocm/device_model.cpp
                     kernels/rocm/executor.cpp engine.cpp)
endif()

# The block arithmetic every layer runs several times: the norms, the residual
# add, the expert activation and the end of a mixture. Strict floating point,
# because a row has to round the same whatever else shares its launch and a
# batch boundary must not move its value.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_kolibri1 PRIVATE kernels/rocm/blocks.hip.cpp)
  set_source_files_properties(kernels/rocm/blocks.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/blocks.hip.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()

# The oracle decodes every weight on the fly; unoptimized it is unusable even
# as a reference, so it keeps -O2 in every configuration.
add_library(gufo_kolibri1_reference STATIC EXCLUDE_FROM_ALL reference.cpp)
target_link_libraries(gufo_kolibri1_reference PUBLIC gufo_kolibri1)
set_source_files_properties(reference.cpp PROPERTIES COMPILE_OPTIONS "-O2")
