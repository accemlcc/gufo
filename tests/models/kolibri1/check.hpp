#ifndef GUFO_TESTS_MODELS_KOLIBRI1_CHECK_HPP_
#define GUFO_TESTS_MODELS_KOLIBRI1_CHECK_HPP_

#include <iostream>
#include <stdexcept>
#include <string>

namespace kolibri1_test {

/// Assertion that stays active in optimized builds.
inline void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template<class Fn>
int Run(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}

}  // namespace kolibri1_test

#endif  // GUFO_TESTS_MODELS_KOLIBRI1_CHECK_HPP_
