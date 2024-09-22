cmake -S . -B build -DPYTHON_INCLUDE_DIRS=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARIES=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=FALSE -DUSE_ADDRESS_SANITIZER=FALSE
cmake --build build -j
