# Quick hack to build the extension module on my Linux machine.
gcc -Wall -c iter_examples.c $CFLAGS -I $(python3 -c "import numpy; print(numpy.get_include())") $(python3-config --cflags) $(python3-config --ldflags) -fpic
gcc -shared -o iter_examples.so iter_examples.o
