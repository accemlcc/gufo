# Kolibri 1 text model package (sigmoid_logit_add MoE with sliding-window and
# global attention layers). The scalar oracle is built only by explicit
# validation targets.
add_library(gufo_kolibri1 STATIC config.cpp)
target_include_directories(gufo_kolibri1 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_kolibri1 PUBLIC gufo_core Threads::Threads)
