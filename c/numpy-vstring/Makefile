SRC = vstring_demo.c vstring_print_utils.c vstring.c
HEADERS = vstring_print_utils.h vstring.h

all: vstring_demo vstring_demo_mock_be

vstring_demo: $(SRC) $(HEADERS)
	$(CC) $(SRC) -o vstring_demo

vstring_demo_mock_be: $(SRC) $(HEADERS)
	$(CC) -DMOCK_BE=1 $(SRC) -o vstring_demo_mock_be

clean:
	rm -rf vstring_demo vstring_demo_mock_be
