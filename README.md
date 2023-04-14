# Parallel Matrix Multiplication (Pthreads/MPI)

## Used libraries

### Pthreads

Was originally installed in MinGW along with CLion.

Add in *CMakeLists.txt* (already added):

```cmake
set(CMAKE_CXX_FLAGS -pthread)
set(THREADS_PREFER_PTHREAD_FLAG ON)
```

### Microsoft MPI

[Download](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi-release-notes)

Install `msmpisetup.exe` and `msmpisdk.msi`.

Add in *CMakeLists.txt* (already added):

```cmake
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(parallels ${MPI_C_LIBRARIES})
```

### Eigen3

[Download](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip)

Unpack the archive and add the path to the directory to *CMakeLists.txt* (change path to yours):

```cmake
include_directories(Path\\to\\Eigen3) # For Windows
```

## CLion Configuration (Windows)

**Executable**: `mpiexec.exe`

**Program arguments**: `-np 20 Path\to\project\build\parallels.exe 20 1000 100 Path\to\save\statistics`

- `-np 20` - Specify the number of processes to use
- `Path\to\project\build\parallels.exe` - Path to the project's executable file
- `20` - Number of threads for Pthreads
- `1000` - Matrix size (rows and columns)
- `100` - Number of runs to collect statistics
- `Path\to\save\statistics` - Path to the folder where to save statistics files