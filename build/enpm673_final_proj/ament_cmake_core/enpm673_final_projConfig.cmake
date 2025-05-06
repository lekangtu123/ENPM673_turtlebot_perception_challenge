# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_enpm673_final_proj_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED enpm673_final_proj_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(enpm673_final_proj_FOUND FALSE)
  elseif(NOT enpm673_final_proj_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(enpm673_final_proj_FOUND FALSE)
  endif()
  return()
endif()
set(_enpm673_final_proj_CONFIG_INCLUDED TRUE)

# output package information
if(NOT enpm673_final_proj_FIND_QUIETLY)
  message(STATUS "Found enpm673_final_proj: 0.0.0 (${enpm673_final_proj_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'enpm673_final_proj' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${enpm673_final_proj_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(enpm673_final_proj_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${enpm673_final_proj_DIR}/${_extra}")
endforeach()
