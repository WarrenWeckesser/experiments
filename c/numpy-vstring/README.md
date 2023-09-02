This directory contains some C code that is proof-of-concept for a
variation of the variable-length string type proposed in NEP 55.
This variation/refinement of the proposal in NEP 55 includes the implementation
of small string optimization and a simpler definition for the `not-a-string`
(a.k.a. missing value) element.

The simplest form of the data stored in the array is this `npy_static_string_t`
struct (note: I have changed the name of the `size_t` field from `len`--as used
in NEP 55--to `size`):

    typedef struct _npy_static_string {
        size_t size;
        char *buf;
    } npy_static_string_t

NEP 55 proposes a potential future enhancement to use *small string optimization*
(SSO), in which small strings are stored directly in the memory used by the
`npy_static_string_t` struct, instead of in a separate block of memory pointed to
by `buf`.  Implementing SSO will impose design constraints on `npy_static_string_t`,
so to ensure we don't implement something to which it will be difficult to add SSO
later, it is worthwhile considering a design now.

To implement SSO, a convention must be adopted that allows the code to
distinguish the SSO case (where the string data *and* the string length must
be stored in the memory used by the `npy_static_string_t` struct) from the standard
case (where `buf` points to the buffer).  We can reserve the most significant bit
of the `size` field to indicate that the data is not a standard indirect variable
string.  Care is needed because the offset of the most significant byte within
`npy_static_string_t` depends on the endianess of the platform.  We want that byte
to be at the beginning or end of the struct, so all the other bytes remain
contiguous and can be used to store the small string.

This can be accomplished by having the order of `size` and `buf` depend on the
platform endianess.  For little-endian, the most significant byte is the last
byte of the memory used by `size`, so we make `size` the last field:

    // Little-endian layout...
    typedef struct _npy_static_string_t {
        char *buf;
        size_t size;
    } npy_static_string_t;

For a big-endian platform, the most significant byte of `size` is the first
byte, so we keep `size` as the first field:

    // Big-endian layout...
    typedef struct _npy_static_string_t {
        size_t size;
        char *buf;
    } npy_static_string_t;

In the C code, this choice of layout can be made based on the `BYTE_ORDER`
macro.

For easy access to the bytes in the struct when SSO is active, we can form
a `union` of `npy_static_string_t` with a second struct that also has a layout
that depends on the platform endianess.  For little-endian:

    // Little-endian SSO buffer layout...
    typedef struct _short_string_buffer {
        unsigned char buffer[sizeof(npy_static_string_t) - 1];
        unsigned char high_byte;
    } short_string_buffer;

For big-endian:

    // Big-endian SSO buffer layout...
    typedef struct _short_string_buffer {
        unsigned char high_byte;
        unsigned char buffer[sizeof(npy_static_string_t) - 1];
    } short_string_buffer;

Then the final definition of the array structure is a C union:

    typedef union _npy_static_string {
        npy_static_string_t vstring;
        short_string_buffer direct_buffer;
    } npy_static_string;

For example, if `string` is an instance of `npy_static_string`, we can
access the high byte of the `size` field as `string.direct_buffer.high_byte`,
and if we know that `string` is using the `npy_static_string_t` part of
the union, we can access `buf` as `string.vstring.buf`.

When the most significant bit of `size` is set, it means the struct is not
a standard `size` and `buf` pair.  For SSO, we need to store the actual size
of the string stored in the struct.  We can use the lowest 4 bits of the
most significant byte of `size` for this, which can represent sizes up to 15.
That is sufficient for the case where `sizeof(size_t)` and `sizeof(char *)`
are 8 (i.e. typical modern 64 bit platforms).  Up to 6 bits can be reserved
for the SSO length (leaving one more bit for the `not-a-string` flag
described below), if we anticipate platforms where the sizes of those types
are larger.

(Note: for the platform-dependent layout described above to work, it is
essential that `sizeof(size_t) == sizeof(char *)`.  If this is not the case
on some platforms to be supported, some more design work is needed.)

When the most significant bit of `size` is set, the second most significant
bit is reserved as the `not-a-string` flag.  In other words, if the two
most significant bits of `size` are `11`, the element represents `not-a-string`
(analogous to `nan` for floats and `nat` for datetimes and timedeltas, and
also interpretable as a missing value).

The use of a simple bit pattern for `not-a-string` is similar to how
`np.nan` and `np.nat` work.  To expose this in the Python API, a Python
object (maybe called `nastring` or `notastring` or something similar)
would allow users to created elements containing the `not-a-string`
value, e.g.

    arr = np.array(["abc", np.nastring, "defghi"], dtype=StringDType())

(With this flag stored in `npy_static_string`, there is no need for
`StringDType` to have the `na_object` parameter.)


Here's a summary of how the highest order byte of `size` is handled:

    Highest order byte
    of `size`           Meaning
    ------------------  ----------------------------------------------------------
    0xxxxxxx            Standard variable string (i.e. `buf` points to a memory
                        buffer of `size` bytes holding the UTF-8 encoding of the
                        string).
    10xxbbbb            Short string direct storage; the bits marked `bbbb` hold
                        the length of the short string.  The remaining bytes of
                        `len` and `buf` are used as the storage buffer for the
                        string. (Design choice: the short string size could use 4,
                        5 or 6 bits of the highest byte.  Here I have suggested 4.
                        Ideally, we need enough bits to represent
                        `sizeof(npy_static_string) - 1`.)
    11xxxxxx            "Not-a-string" (i.e. like np.nan or np.nat, but for
                        strings).


Benefits
--------

* `size == 0` *always* means a string of length 0; `buf` is ignored in that
  case (and of course C code must never dereference `buf` that case, because
  there is no value to get from memory).  If the memory for an array of
  `StringDType` is created with `calloc()` (or the equivalent from the Python
  C API), it will automatically result in an array of empty strings.
* `not-a-string` is indicated by setting two bits in the `size` field.  Like
  the `size == 0` case, the `buf` pointer is unused in this case.  No special
  object is required to be stored internally to represent the `not-a-string`.
