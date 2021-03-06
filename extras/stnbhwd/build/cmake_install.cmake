# Install script for directory: /home/szf/spynet3/extras/stnbhwd

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/szf/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so"
         RPATH "$ORIGIN/../lib:/home/szf/torch/install/lib:/opt/OpenBLAS/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib" TYPE MODULE FILES "/home/szf/spynet3/extras/stnbhwd/build/libstn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so"
         OLD_RPATH "/home/szf/torch/install/lib:/opt/OpenBLAS/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/szf/torch/install/lib:/opt/OpenBLAS/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lua/stn" TYPE FILE FILES
    "/home/szf/spynet3/extras/stnbhwd/BilinearSamplerBHWD.lua"
    "/home/szf/spynet3/extras/stnbhwd/ScaleBHWD.lua"
    "/home/szf/spynet3/extras/stnbhwd/test.lua"
    "/home/szf/spynet3/extras/stnbhwd/AffineGridGeneratorBHWD.lua"
    "/home/szf/spynet3/extras/stnbhwd/AffineTransformMatrixGenerator.lua"
    "/home/szf/spynet3/extras/stnbhwd/init.lua"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libcustn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libcustn.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libcustn.so"
         RPATH "$ORIGIN/../lib:/home/szf/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib" TYPE MODULE FILES "/home/szf/spynet3/extras/stnbhwd/build/libcustn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libcustn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libcustn.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libcustn.so"
         OLD_RPATH "/home/szf/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/szf/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libcustn.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/szf/spynet3/extras/stnbhwd/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
