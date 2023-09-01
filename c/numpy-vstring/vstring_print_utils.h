#ifndef VSTRING_PRINT_UTILS_H

void raw_dump(npy_static_string *string);
void print_bytes(size_t size, char *buf);
void print_vstring(npy_static_string *string);
void print_vstring_array(size_t n, npy_static_string *arr);

#endif
