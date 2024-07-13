# Quick hack to build the extension module on my Linux machine.
g++ -c experiment.cpp -I $(python3 -c "import numpy; print(numpy.get_include())") $(python3.12-config --cflags) $(python3.12-config --ldflags) -fpic
g++ -shared -o experiment.so experiment.o
