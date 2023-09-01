#include <stdio.h>
#include "vstring.h"

void
raw_dump(npy_static_string *string)
{
    unsigned char *p = (unsigned char *) string;
    for (size_t i = 0; i < sizeof(npy_static_string); ++i) {
        printf("%02x ", *(p + i));
    }
    printf("\n");
}

void
print_bytes(size_t size, char *buf)
{
    putchar('"');
    for (size_t i = 0; i < size; ++i) {
        putchar(*(buf+i));        
    }
    putchar('"');
}

void
print_vstring(npy_static_string *string)
{
    unsigned char high_byte = string->direct_buffer.high_byte;
    if (is_notastring(string)) {
        printf("notastring");
    }
    else if (is_short_string(string)) {
        // Short string.
        printf("short string: ");
        size_t size = high_byte & SHORT_STRING_SIZE_MASK;
        print_bytes(size, (char *)&(string->direct_buffer.buffer));
    }
    else {
#ifdef MOCK_BE
        size_t size = get_mock_be_size_t(&(string->vstring.size));
#else
        size_t size = string->vstring.size;
#endif
        printf("standard:     "); 
        print_bytes(size, string->vstring.buf);
    }
}

void
print_vstring_array(size_t n, npy_static_string *arr)
{
    for (size_t i = 0; i < n; ++i) {
        print_vstring(arr+i);
        printf("\n");
    }
}
