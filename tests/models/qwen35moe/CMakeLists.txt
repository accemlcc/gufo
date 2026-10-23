# Host contracts run in the hosted PR suite and need no GPU. Model-backed
# checks read the GGUF named by GUFO_QWEN35MOE_MODEL and skip (77) without it.
add_custom_target(qwen35moe_tests)

function(add_qwen35moe_test name library labels)
  set(target qwen35moe_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen35moe.${name} COMMAND ${target})
  set_tests_properties(qwen35moe.${name} PROPERTIES
    LABELS "qwen35moe;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(qwen35moe_tests ${target})
endfunction()

add_qwen35moe_test(config gufo_qwen35moe "cpu")
add_qwen35moe_test(weights gufo_qwen35moe "cpu")
add_qwen35moe_test(artifact gufo_qwen35moe "cpu;external-model")

# The device checks need the matrix cores of this generation.
if (ENGINE_ENABLE_HIP)
  add_qwen35moe_test(route gufo_qwen35moe "hip;gpu;gfx1151")
  add_qwen35moe_test(blocks gufo_qwen35moe "hip;gpu;gfx1151")
endif()

# The oracle is checked against a NumPy fixture of the same tiny artifact;
# both must implement the graph independently, so the check keeps -O2.
add_qwen35moe_test(reference gufo_qwen35moe_reference "cpu;slow")
target_compile_definitions(qwen35moe_reference_test PRIVATE
  GUFO_QWEN35MOE_FIXTURES="${PROJECT_SOURCE_DIR}/tests/models/qwen35moe/fixtures")
