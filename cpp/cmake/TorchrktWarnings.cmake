function(torchrkt_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(${target}
      PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
  endif()
endfunction()

# Nix's cc-wrapper passes header search paths through NIX_CFLAGS_COMPILE rather
# than the compiler's default search; mirror them as -isystem so warnings from
# third-party (libtorch) headers stay out of our build output. Ported verbatim
# from xgboost-rkt's xgbcompat_apply_nix_cflags.
function(torchrkt_apply_nix_cflags target)
  if(DEFINED ENV{NIX_CFLAGS_COMPILE})
    separate_arguments(TORCHRKT_NIX_CFLAGS UNIX_COMMAND "$ENV{NIX_CFLAGS_COMPILE}")
    set(TORCHRKT_EXPECT_SYSTEM_INCLUDE OFF)
    foreach(flag IN LISTS TORCHRKT_NIX_CFLAGS)
      if(TORCHRKT_EXPECT_SYSTEM_INCLUDE)
        target_compile_options(${target} PRIVATE "-isystem${flag}")
        set(TORCHRKT_EXPECT_SYSTEM_INCLUDE OFF)
      elseif(flag STREQUAL "-isystem")
        set(TORCHRKT_EXPECT_SYSTEM_INCLUDE ON)
      endif()
    endforeach()
  endif()
  foreach(dir IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
    if(EXISTS "${dir}")
      target_compile_options(${target} PRIVATE "-isystem${dir}")
    endif()
  endforeach()
endfunction()
