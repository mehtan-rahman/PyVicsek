cmake_minimum_required(VERSION 3.18...3.31)

# LANGUAGES NONE: C++ is enabled below only if a compiler is found, so the
# sdist still installs (as the pure-Python package, numpy path) without one.
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES NONE)

option(VICSEK_KERNEL "Build the C++ kernel vicsek._kernel" ON)

if(VICSEK_KERNEL)
  include(CheckLanguage)
  check_language(CXX)
  if(NOT CMAKE_CXX_COMPILER)
    message(WARNING "No C++ compiler found: installing vicsek without the C++ kernel "
                    "(the numpy path is used). Install a C++17 compiler to build it.")
    set(VICSEK_KERNEL OFF)
  endif()
endif()

if(VICSEK_KERNEL)
  enable_language(CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
               OPTIONAL_COMPONENTS Development.SABIModule)
  find_package(Threads REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_kernel STABLE_ABI NB_STATIC src/vicsek/_kernel.cpp)
  target_link_libraries(_kernel PRIVATE Threads::Threads)
  # No fused multiply-add contraction: keeps the arithmetic identical to numpy's.
  if(NOT MSVC)
    target_compile_options(_kernel PRIVATE -ffp-contract=off)
  endif()

  install(TARGETS _kernel LIBRARY DESTINATION vicsek)
endif()
