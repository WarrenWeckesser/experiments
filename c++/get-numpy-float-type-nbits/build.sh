PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")
g++ -pedantic -std=c++17 -I$PYTHON_INCLUDE -I$NUMPY_INCLUDE main.cpp -o main
