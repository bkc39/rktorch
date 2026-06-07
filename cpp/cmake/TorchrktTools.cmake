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
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
              --target ${build_target}
      COMMAND ${CLANG_TIDY_EXECUTABLE}
              --quiet
              -p ${CMAKE_BINARY_DIR}
              --extra-arg=-w
              ${TORCHRKT_TIDY_SOURCES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running clang-tidy over torchrkt sources"
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
