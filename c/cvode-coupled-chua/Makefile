# Set SUNDIALS_DIR externally or by editing this file,
# e.g. SUNDIALS_DIR=/usr/local
SUNDIALS_LIB_DIR=$(SUNDIALS_DIR)/lib
SUNDIALS_INC_DIR=$(SUNDIALS_DIR)/include
SUNDIALS_LIBS=-lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense -lsundials_core
SUNDIALS_INCS=-I$(SUNDIALS_INC_DIR)
LIBS=-lm

all: chua2_root_switching

chua2_root_switching: chua2_root_switching.o chua2_cv7.o
	$(CC) $(LDFLAGS) -o chua2_root_switching chua2_root_switching.o chua2_cv7.o -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)

chua2_root_switching.o: chua2_root_switching.c chua2_cv7.h
	$(CC) $(CPPFLAGS) $(SUNDIALS_INCS) -c chua2_root_switching.c

chua2_cv7.o: chua2_cv7.c chua2_cv7.h
	$(CC) $(CPPFLAGS) $(SUNDIALS_INCS) -c chua2_cv7.c

clean:
	rm -f chua2_root_switching chua2_root_switching.o chua2_cv7.o

