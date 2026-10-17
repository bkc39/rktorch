# The ABI stamp: a SHA-256 over every header under include/torchrkt/c_api/,
# each named by its path relative to that directory, in sorted order.
#
#   include(cmake/abi_stamp.cmake)    configure time, from CMakeLists.txt
#   cmake -P cpp/cmake/abi_stamp.cmake                  prints the stamp
#   cmake -DOUT=<file> -P cpp/cmake/abi_stamp.cmake     writes it as .rktd
#   cmake -DC_API_DIR=<dir> ...                         hashes another copy
#
# The bindings' copy, torch/foreign/raw/abi-stamp.rktd, is written by the
# second form; the Racket loader refuses a shim whose stamp differs.

function(torchrkt_abi_stamp out_var c_api_dir)
  file(GLOB_RECURSE headers RELATIVE "${c_api_dir}" "${c_api_dir}/*.h")
  list(SORT headers)
  set(manifest "")
  foreach(header IN LISTS headers)
    file(SHA256 "${c_api_dir}/${header}" digest)
    string(APPEND manifest "${header}\t${digest}\n")
  endforeach()
  string(SHA256 stamp "${manifest}")
  set(${out_var} "${stamp}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED C_API_DIR)
  set(C_API_DIR "${CMAKE_CURRENT_LIST_DIR}/../include/torchrkt/c_api")
endif()
get_filename_component(C_API_DIR "${C_API_DIR}" ABSOLUTE)

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  torchrkt_abi_stamp(stamp "${C_API_DIR}")
  if(DEFINED OUT)
    file(WRITE "${OUT}" "\"${stamp}\"\n")
  else()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "${stamp}")
  endif()
else()
  torchrkt_abi_stamp(TORCHRKT_ABI_STAMP "${C_API_DIR}")
  file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${C_API_DIR}/*.h")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${headers})
endif()
