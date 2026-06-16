function(torchrkt_add_format_target target)
  find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format)
  file(GLOB_RECURSE TORCHRKT_FORMAT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
  )

  if(CLANG_FORMAT_EXECUTABLE)
    add_custom_target(${target}
      COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror
              ${TORCHRKT_FORMAT_SOURCES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Checking C/C++ formatting with clang-format"
      VERBATIM
    )
  else()
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E false
      COMMENT "clang-format was not found"
      VERBATIM
    )
  endif()
endfunction()

function(torchrkt_add_tidy_target target build_target)
  find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy)
  file(GLOB_RECURSE TORCHRKT_TIDY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  )

  if(CLANG_TIDY_EXECUTABLE)
    # clang-tidy re-parses the whole ATen header tree per TU (~50s/file), so the
    # only real lever for the full sweep is fanning the files out across cores.
    # tidy-parallel.sh runs one clang-tidy per file under xargs -P, sized from
    # $NIX_BUILD_CORES (honors `nix build --cores`, capped at 6 on the lab host;
    # CI sets it to its core count). For the inner loop, prefer
    # scripts/tidy-changed, which lints only the files you touched.
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
              --target ${build_target}
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tidy-parallel.sh
              ${CLANG_TIDY_EXECUTABLE} ${CMAKE_BINARY_DIR}
              ${TORCHRKT_TIDY_SOURCES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running clang-tidy (parallel) over torchrkt sources"
      VERBATIM
    )
  else()
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E false
      COMMENT "clang-tidy was not found"
      VERBATIM
    )
  endif()
endfunction()
