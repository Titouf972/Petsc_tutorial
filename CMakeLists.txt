cmake_minimum_required(VERSION 3.28)

set (CMAKE_CXX_COMPILER  "/usr/local/Cellar/gcc/14.2.0_1/bin/g++-14")
set(CMAKE_CXX_COMPILER_ID "GNU")
    set(CMAKE_CXX_STANDARD 17)

    if(CMAKE_CXX_COMPILER_LOADED)
     message(STATUS "Compiler path: ${CMAKE_CXX_COMPILER}")
     message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
     message(STATUS "Compiler version:    
             ${CMAKE_CXX_COMPILER_VERSION}")
     message(STATUS "Compiler is part of GCC: 
             ${CMAKE_COMPILER_IS_GNUCXX}")
   endif()
project(Petsc_Tutorial)
include(FindPkgConfig REQUIRED)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# set root of location to find PETSc's pkg-config
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

set(CMAKE_CXX_COMPILER mpicxx)

add_executable(solve1 test1.cpp)
add_executable(Laplacian2D Laplacian2D.cpp)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
target_link_libraries(solve1 PkgConfig::PETSC)
target_link_libraries(Laplacian2D PkgConfig::PETSC)