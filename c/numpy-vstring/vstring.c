#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "vstring.h"

#ifdef MOCK_BE

// This proof-of-concept code has been developed on a little-endian system.
// If MOCK_BE is defined when the code is built, the `size` field of the
// npy_static_string struct will be stored in big-endian format.  It is
// my crude attempt to test the concept for big-endian systems, without
// actually having a true big-endian system.

void
set_mock_be_size_t(size_t *p, size_t size)
{
    for (size_t i = 0; i < sizeof(size_t); ++i) {
        *(((char *) p) + i) = ((char *) &size)[sizeof(size_t) - i - 1];
    }
}

size_t
get_mock_be_size_t(size_t *p)
{
    size_t size;
    for (size_t i = 0; i < sizeof(size_t); ++i) {
        *(((char *) &size) + i) = ((char *) p)[sizeof(size_t) - i - 1];
    }
    return size;
}

#endif // MOCK_BE

bool
is_notastring(npy_static_string *string)
{
    unsigned char hb = string->direct_buffer.high_byte;
    return (hb & HIGH_BYTE_NOTSTANDARD) && (hb & HIGH_BYTE_NOTASTRING);
}

bool
is_short_string(npy_static_string *string)
{
    unsigned char hb = string->direct_buffer.high_byte;
    return (hb & HIGH_BYTE_NOTSTANDARD) && ~(hb & HIGH_BYTE_NOTASTRING);
}

//
// There is no memory management going on here--this is just
// a proof of concept for the npy_static_string representation.
//
void
assign_string(npy_static_string *string, size_t size, char *str)
{
    if (size == 0) {
        // Empty string.
        string->vstring.size = 0;
    }
    else if (size <= SHORT_STRING_MAX_SIZE) {
        string->direct_buffer.high_byte = HIGH_BYTE_NOTSTANDARD | size;
        memcpy(&(string->direct_buffer.buffer), str, size);
    }
    else {
#ifdef MOCK_BE
        set_mock_be_size_t(&(string->vstring.size), size);
#else
        string->vstring.size = size;
#endif
        string->vstring.buf = str;
    }
}

void
assign_notastring(npy_static_string *string)
{
    string->direct_buffer.high_byte = HIGH_BYTE_NOTSTANDARD | HIGH_BYTE_NOTASTRING;
}
