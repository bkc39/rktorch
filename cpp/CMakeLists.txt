cmake_minimum_required(VERSION 3.20)
project(torchrkt VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/TorchrktWarnings.cmake)
include(cmake/TorchrktTools.cmake)

# Provided by libtorch-bin (or python3Packages.torch's dev output) on
# CMAKE_PREFIX_PATH; see flake.nix's `torchSource` knob.
find_package(Torch CONFIG REQUIRED)

# Torch's bundled Caffe2 config leaks MKL stub link directories
# ("/lib;/lib/intel64;...") into the RPATH CMake expects to rewrite at
# install time, which makes `cmake --install` fail on x86_64-linux with
# "file RPATH_CHANGE could not write new RPATH". The toolchain already
# records the real libtorch RUNPATH at link time, so drop the stubs.
if(TARGET caffe2::mkl)
  set_property(TARGET caffe2::mkl PROPERTY INTERFACE_LINK_DIRECTORIES "")
endif()

add_library(torchrkt SHARED)
add_library(torchrkt::torchrkt ALIAS torchrkt)

target_sources(torchrkt
  PRIVATE
    src/torchrkt/autograd.cpp
    src/torchrkt/creation.cpp
    src/torchrkt/detail/error.cpp
    src/torchrkt/device.cpp
    src/torchrkt/elementwise.cpp
    src/torchrkt/global.cpp
    src/torchrkt/linalg.cpp
    src/torchrkt/random.cpp
    src/torchrkt/reduce.cpp
    src/torchrkt/shape_ops.cpp
    src/torchrkt/tensor.cpp
)

# Generated shim sources; the file is owned by the codegen generator
# (python3 -m codegen), which rewrites it on every run.
include(src/torchrkt/generated/sources.cmake)

target_include_directories(torchrkt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Treat libtorch headers as system headers so -Wall/-Wextra/-Wpedantic report
# only our own code, not ATen's.
target_include_directories(torchrkt SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(torchrkt PRIVATE ${TORCH_LIBRARIES})

# libtorch ships ABI flags (notably -D_GLIBCXX_USE_CXX11_ABI=...) that our TUs
# including ATen headers must match, or std::string symbols won't link.
if(TORCH_CXX_FLAGS)
  separate_arguments(TORCHRKT_TORCH_CXX_FLAGS UNIX_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(torchrkt PRIVATE ${TORCHRKT_TORCH_CXX_FLAGS})
endif()

torchrkt_enable_warnings(torchrkt)
torchrkt_apply_nix_cflags(torchrkt)
torchrkt_add_format_target(format-check)
torchrkt_add_tidy_target(tidy torchrkt)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_library(torchrkt_c_api_compile_test OBJECT
    tests/torchrkt/c_api_compile_test.c
  )
  target_include_directories(torchrkt_c_api_compile_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  torchrkt_apply_nix_cflags(torchrkt_c_api_compile_test)

  add_executable(torchrkt_tests
    tests/torchrkt/autograd_test.cpp
    tests/torchrkt/device_test.cpp
    tests/torchrkt/finalizer_death_test.cpp
    tests/torchrkt/generated_golden_test.cpp
    tests/torchrkt/generated_tranche2_test.cpp
    tests/torchrkt/generated_tranche3_test.cpp
    tests/torchrkt/ops_test.cpp
    tests/torchrkt/random_test.cpp
    $<TARGET_OBJECTS:torchrkt_c_api_compile_test>
  )
  target_include_directories(torchrkt_tests
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      # finalizer_death_test is deliberately white-box: it builds a real
      # tr_tensor around a throwing from_blob deleter, so it needs the
      # private handle header and the torch headers other tests don't.
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_include_directories(torchrkt_tests SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(torchrkt_tests
    PRIVATE
      torchrkt
      # for the white-box finalizer death test's direct from_blob use.
      ${TORCH_LIBRARIES}
      GTest::gtest_main
  )
  torchrkt_enable_warnings(torchrkt_tests)
  torchrkt_apply_nix_cflags(torchrkt_tests)
  # PRE_TEST defers test discovery from build time to ctest time: on CI
  # runners the freshly linked binary aborts under the 5s build-time
  # discovery probe (cold-loading libtorch's dylibs), killing the build
  # before ctest ever runs. At ctest time discovery shares the test
  # timeout and failures come with real output.
  gtest_discover_tests(torchrkt_tests DISCOVERY_MODE PRE_TEST)
endif()

install(TARGETS torchrkt
  EXPORT torchrktTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT torchrktTargets
  FILE torchrktTargets.cmake
  NAMESPACE torchrkt::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/torchrkt
)
