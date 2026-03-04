cmake_minimum_required(VERSION 3.16)
project(mini_detector_new LANGUAGES CXX)

find_package(DD4hep REQUIRED COMPONENTS DDCore DDRec)
include(DD4hep)

dd4hep_add_plugin(detector_plugin
  SOURCES geometries/src/hcal.cpp
)

target_link_libraries(detector_plugin PRIVATE DD4hep::DDCore DD4hep::DDRec)
target_compile_features(detector_plugin PRIVATE cxx_std_20)

set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR};${CMAKE_BINARY_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

install(TARGETS detector_plugin LIBRARY DESTINATION lib)

option(BUILD_PROCESSOR "Build the standalone EDM4hep processor tool" ON)
if(BUILD_PROCESSOR)
  find_package(ROOT REQUIRED COMPONENTS Core Hist Tree)
  find_package(EDM4HEP REQUIRED)
  find_package(podio REQUIRED COMPONENTS podioRootIO)

  add_executable(process analysis/processor.cc)
  target_compile_features(process PRIVATE cxx_std_20)
  target_link_libraries(process
    PRIVATE
      ROOT::Core ROOT::Hist ROOT::Tree
      EDM4HEP::edm4hep
      podio::podio
      podio::podioRootIO
  )

  set_target_properties(process PROPERTIES
    BUILD_RPATH "${CMAKE_BUILD_RPATH}"
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  install(TARGETS process RUNTIME DESTINATION bin)
endif()
