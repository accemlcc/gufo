# Qwen3.5-MoE text model package (qwen35moe).

# Config and weight binding only for now: the trunk graph (Gated DeltaNet and
# full-attention layers with a softmax top-k mixture in every layer) is bound
# and validated, the HIP engine that executes it is the next step. The kernels
# it will reuse already exist: the SSM and GQA path of `qwen/hip`, and the
# grouped expert projection of `kolibri1` / `qwen38_flash_next`.
add_library(gufo_qwen35moe STATIC config.cpp weights.cpp)
target_include_directories(gufo_qwen35moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen35moe PUBLIC gufo_core Threads::Threads)
