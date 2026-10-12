# Kolibri 1 text model package (sigmoid_logit_add MoE with sliding-window and
# global attention layers). The scalar oracle is built only by explicit
# validation targets.
add_library(gufo_kolibri1 STATIC chat_template.cpp config.cpp tokenizer.cpp
                      weights.cpp)
target_include_directories(gufo_kolibri1 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_kolibri1 PUBLIC gufo_core Threads::Threads)

# The device path: attention over the binary16 cache on the matrix cores.
if (ENGINE_ENABLE_HIP)
  target_sources(gufo_kolibri1 PRIVATE kernels/rocm/attention.hip.cpp)
  set_source_files_properties(kernels/rocm/attention.hip.cpp
    PROPERTIES LANGUAGE HIP)
endif()

# The oracle decodes every weight on the fly; unoptimized it is unusable even
# as a reference, so it keeps -O2 in every configuration.
add_library(gufo_kolibri1_reference STATIC EXCLUDE_FROM_ALL reference.cpp)
target_link_libraries(gufo_kolibri1_reference PUBLIC gufo_kolibri1)
set_source_files_properties(reference.cpp PROPERTIES COMPILE_OPTIONS "-O2")
