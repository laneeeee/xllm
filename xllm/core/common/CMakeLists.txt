include(cc_library)
include(cc_test)

if(TARGET cpprest)
    # 方法A: 设置 C++ 标准为 20
    set_target_properties(cpprest PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    # 方法B: 或者直接添加编译选项
    target_compile_options(cpprest PRIVATE 
        -std=c++20
        -Wno-c++20-extensions
    )
    
    message(STATUS "Modified cpprest target to use C++20")
endif()

cc_library(
  NAME 
    common
  HDRS
    device_memory.h
    etcd_client.h
    global_flags.h
    instance_name.h
    layer_synchronizer.h
    macros.h
    metrics.h
    $<$<BOOL:${USE_NPU}>:mspti_helper.h>
    options.h
    rate_limiter.h
    types.h
    device_monitor.h
  SRCS
    device_memory.cpp
    etcd_client.cpp
    global_flags.cpp
    layer_synchronizer.cpp
    metrics.cpp
    $<$<BOOL:${USE_NPU}>:mspti_helper.cpp>
    options.cpp
    rate_limiter.cpp
    device_monitor.cpp
  DEPS
    util
    absl::random_random
    absl::strings
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_MSPTI}>:mspti>
    $<$<BOOL:${USE_NPU}>:ms_tools_ext>
    Boost::serialization
    cpprest
    etcd-cpp-api
)

cc_library(
  NAME
    flags
  HDRS
    global_flags.h
  SRCS
    global_flags.cpp
  DEPS
    gflags::gflags
)

cc_test(
  NAME 
    common_test
  SRCS
    rate_limiter_test.cpp
  DEPS
    common
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
    glog::glog
)
target_link_libraries(common PRIVATE OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
add_dependencies(common brpc-static)


