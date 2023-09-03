#ifndef VSTRING_H

#include <stdbool.h>

#ifndef MOCK_BE

//
// Little-endian memory layout.
//

typedef struct _npy_static_string_t {
    char *buf;
    size_t size;
} npy_static_string_t;

typedef struct _short_string_buffer {
    unsigned char buffer[sizeof(npy_static_string_t) - 1];
    unsigned char high_byte;
} short_string_buffer;

#else  // MOCK_BE

//
// Big-endian memory layout.
//

typedef struct _npy_static_string_t {
    size_t size;
    char *buf;
} npy_static_string_t;

typedef struct _short_string_buffer {
    unsigned char high_byte;
    unsigned char buffer[sizeof(npy_static_string_t) - 1];
} short_string_buffer;

#endif

typedef union _npy_static_string {
    npy_static_string_t vstring;
    short_string_buffer direct_buffer;
} npy_static_string;


// These masks apply to the high-order byte of the size field
// of npy_static_string_t.
#define HIGH_BYTE_NOTSTANDARD  0x80
#define HIGH_BYTE_NOTASTRING   0x40
#define SHORT_STRING_SIZE_MASK 0x0F
#define SHORT_STRING_MAX_SIZE  (sizeof(npy_static_string) - 1)

void raw_dump(npy_static_string *string);

bool is_notastring(npy_static_string *string);
bool is_short_string(npy_static_string *string);

void assign_string(npy_static_string *string, size_t size, char *str);
void assign_notastring(npy_static_string *string);

void print_bytes(size_t size, char *buf);
void print_vstring(npy_static_string *string);
void print_vstring_array(size_t n, npy_static_string *arr);

#ifdef MOCK_BE
void set_mock_be_size_t(size_t *p, size_t size);
size_t get_mock_be_size_t(size_t *p);
#endif

#endif
