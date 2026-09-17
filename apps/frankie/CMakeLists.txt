set(NINFER_LLAMA_SOURCE_DIR "" CACHE PATH "Frankie-enabled llama.cpp source directory")
if(NOT EXISTS "${NINFER_LLAMA_SOURCE_DIR}/tools/frankie/realtime-server.h")
  message(FATAL_ERROR "NINFER_LLAMA_SOURCE_DIR must contain the reusable Frankie runtime")
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
# Keep httplib class layouts identical to the HTTP-only NInfer transport.
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_BORINGSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_LIBRESSL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${NINFER_LLAMA_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama")
add_subdirectory("${NINFER_LLAMA_SOURCE_DIR}/tools/frankie" "${CMAKE_CURRENT_BINARY_DIR}/runtime")

add_executable(ninfer-frankie main.cpp options.cpp ninfer-brain.cpp)
target_compile_features(ninfer-frankie PRIVATE cxx_std_20)
target_include_directories(ninfer-frankie PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-frankie PRIVATE ninfer_serve ninfer_product_logging frankie-runtime)
set_target_properties(ninfer-frankie PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/apps")
