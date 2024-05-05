#include <stdio.h>
#include <zlib.h>

#define BUFSIZE 1000

int main(int argc, char *argv[])
{
    gzFile f;
    int status;
    int nbytes;
    char buffer[BUFSIZE];

    // Open the .gz file.
    f = gzopen("foo.txt.gz", "rb");
    if (f == NULL) {
        fprintf(stderr, "Failed to open the file.\n");
        return -1;
    }
    fprintf(stderr, "Opened!\n");

    // Set the buffer size of the gzFile.
    status = gzbuffer(f, 65536);
    if (status != 0) {
        fprintf(stderr, "gzbuffer returned %d\n", status);
        return -1;
    }
    fprintf(stderr, "gzbuffer() succeeded!\n");

    // Read a block of data from the file.
    nbytes = gzread(f, buffer, BUFSIZE-1);
    if (nbytes == -1) {
        fprintf(stderr, "gzread() failed.\n");
        return -1;
    }
    fprintf(stderr, "gzread() returned %d\n", nbytes);

    status = gzclose(f);
    if (status != Z_OK) {
        fprintf(stderr, "gzclose() returned %d!\n", status);
        return -1;
    }
    fprintf(stderr, "file closed.\n");

    buffer[nbytes] = '\0';
    printf("%s\n", "-----");
    printf("%s", buffer);
    printf("%s\n", "-----");   
}
