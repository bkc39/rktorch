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

add_library(torchrkt SHARED)
add_library(torchrkt::torchrkt ALIAS torchrkt)

target_sources(torchrkt
  PRIVATE
    src/torchrkt/detail/error.cpp
    src/torchrkt/global.cpp
    src/torchrkt/random.cpp
    src/torchrkt/tensor.cpp
)

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
    tests/torchrkt/random_test.cpp
    $<TARGET_OBJECTS:torchrkt_c_api_compile_test>
  )
  target_include_directories(torchrkt_tests
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
  )
  target_link_libraries(torchrkt_tests
    PRIVATE
      torchrkt
      GTest::gtest_main
  )
  torchrkt_enable_warnings(torchrkt_tests)
  torchrkt_apply_nix_cflags(torchrkt_tests)
  gtest_discover_tests(torchrkt_tests)
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
