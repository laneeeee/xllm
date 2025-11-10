include(cc_binary)

include_directories(.)
if(USE_ILU)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
endif()
add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
add_subdirectory(proto)
add_subdirectory(processors)
add_subdirectory(pybind)
add_subdirectory(server)

cc_binary(
  NAME
    xllm
  SRCS
    xllm.cpp
  DEPS
    torch
    :flags
    :xllm_server
    :master
    absl::strings
    Boost::serialization
    gflags::gflags
    glog::glog
    Folly::folly
    nlohmann_json::nlohmann_json
)

# link brpc
target_link_libraries(xllm PRIVATE glog::glog brpc leveldb::leveldb ZLIB::ZLIB protobuf::libprotobuf ${OpenCV_LIBS})
add_dependencies(xllm brpc-static)
target_link_libraries(xllm
    PUBLIC
    ${LIBNL-3_LIBRARIES}
    ${LIBNL-ROUTE-3_LIBRARIES}
)
target_include_directories(xllm
    PUBLIC
    ${LIBNL-3_INCLUDE_DIRS}
    ${LIBNL-ROUTE-3_INCLUDE_DIRS}
)

if(USE_NPU)
  set(COMMON_LIBS Python::Python ascendcl hccl c_sec nnopbase ms_tools_ext)
elseif(USE_MLU)
  set(COMMON_LIBS Python::Python)
elseif(USE_ILU)
  set(COMMON_LIBS Python::Python)
endif()

if(USE_MSPTI)
  list(APPEND COMMON_LIBS mspti)
endif()
target_link_libraries(xllm PUBLIC ${COMMON_LIBS})

# install xllm binary
install(TARGETS xllm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# Install all dependencies for xllm
install(CODE [[
    file(GET_RUNTIME_DEPENDENCIES
          RESOLVED_DEPENDENCIES_VAR DEPENDENCIES
          UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPENDENCIES
          EXECUTABLES $<TARGET_FILE:xllm>)

    file(INSTALL
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
          FILES ${DEPENDENCIES}
          FOLLOW_SYMLINK_CHAIN)

    # This should not be possible, but error out when a dependency cannot
    # be resolved.
    list(LENGTH UNRESOLVED_DEPENDENCIES UNRESOLVED_LENGTH)
    if(${UNRESOLVED_LENGTH} GREATER 0)
        message(FATAL_ERROR "Unresolved dependencies: ${UNRESOLVED_DEPENDENCIES}")
    endif()
]])

