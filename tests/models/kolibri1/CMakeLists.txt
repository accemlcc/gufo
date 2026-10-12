# Host contracts run in the hosted PR suite and need no GPU. Model-backed
# checks read the GGUF named by GUFO_KOLIBRI1_MODEL and skip (77) without it.
add_custom_target(kolibri1_tests)

function(add_kolibri1_test name library labels)
  set(target kolibri1_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME kolibri1.${name} COMMAND ${target})
  set_tests_properties(kolibri1.${name} PROPERTIES
    LABELS "kolibri1;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(kolibri1_tests ${target})
endfunction()

add_kolibri1_test(artifact gufo_kolibri1 "cpu;external-model")
add_kolibri1_test(config gufo_kolibri1 "cpu")

# The oracle is checked against a NumPy fixture of the same tiny artifact;
# both must implement the graph independently, so the check is slow in
# debug builds and keeps -O2.
add_kolibri1_test(reference gufo_kolibri1_reference "cpu;slow")
target_compile_definitions(kolibri1_reference_test PRIVATE
  GUFO_KOLIBRI1_FIXTURES="${PROJECT_SOURCE_DIR}/tests/models/kolibri1/fixtures")
