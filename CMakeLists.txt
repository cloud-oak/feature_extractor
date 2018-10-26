cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 14)
project(feature_extractor LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/FindTBB")

find_package(TBB COMPONENTS tbbmalloc tbbmalloc_proxy)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -march=native -Wno-unknown-pragmas -pthread -O3")

find_package(ZLIB REQUIRED)

include_directories(
  cnpy
  ${ZLIB_INCLUDE_DIRS}
)

add_library(cnpy SHARED "cnpy/cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

add_executable(feature_extractor feature_extractor.cpp)

target_link_libraries(
  feature_extractor
  cnpy
  pthread
  tbb
  Eigen3::Eigen
)
