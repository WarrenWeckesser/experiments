// compile with:
// $ gcc demo.c $(icu-config --cflags --ldflags) -o demo
//
// Reminder: http://icu-project.org/apiref/icu4c/utf8_8h.html

#include <stdio.h>
#include <string.h>
#include <unicode/uclean.h>
#include <unicode/ustring.h>

#define BUFSIZE 64

int main(int argc, char *argv[])
{
    UErrorCode status = U_ZERO_ERROR;

    u_init(&status);
    if (U_FAILURE(status)) {
        printf("Uh oh: %s\n", u_errorName(status));
        return 1;
    }

    const char *s = "नमस्तेABC";  // I think this will be UTF-8.
    int32_t i = 0;
    int32_t len = strlen(s);
    printf("s:");
    for (i = 0; i < len; ++i) {
        printf(" %02X", (uint8_t) s[i]);
    }
    printf("\n");
    UChar32 c;
    printf("strlen(s) = %d\n", len);
    i = 0;
    while (i < len) {
        int32_t pos = i;
        U8_NEXT(s, i, len, c);
        printf("pos = %2d,  c = U+%04x\n", pos, c);
    }

    UChar dest[BUFSIZE];
    int destlen;
    status = U_ZERO_ERROR;
    u_strFromUTF8(dest, BUFSIZE, &destlen, s, -1, &status);
    printf("destlen = %d\n", destlen);
    for (i = 0; i < destlen; ++i) {
        printf("U+%04x\n", dest[i]);
    }
    return 0;
}
