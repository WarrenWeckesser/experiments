#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    char first[100];
    char second[100];
    char third[100];
    char *line = "  AAA BBB  CCC   ";
    int result = sscanf(line, "%s %s %s", first, second, third);
    printf("'%s'\n", first);
    printf("'%s'\n", second);
    printf("'%s'\n", third);

    return 0;
}
