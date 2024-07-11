# Quick hack to build the extension module on my Linux machine.
gcc -Wall -c experiment.c -I $(python3 -c "import numpy; print(numpy.get_include())") $(python3-config --cflags) $(python3-config --ldflags) -fpic
gcc -shared -o experiment.so experiment.o
