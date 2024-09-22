set (source_files
    Transform.h
    Transform.cpp
)
pybind11_add_module(Transform_HEVC SHARED ${source_files} exportDCTHEVC.cpp)
target_include_directories(Transform_HEVC PRIVATE ../TLibCommon)
target_link_libraries(Transform_HEVC PRIVATE TLibCommon)
