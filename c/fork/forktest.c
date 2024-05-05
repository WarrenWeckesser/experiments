
#include <stdio.h>
#include <unistd.h>

int main(void)
    {
    int pid = fork();
    if (pid == 0)
        {
        /* Child */
        execl("/bin/ls","ls","-l",0);
        }
    else
        {
        printf("I am the parent.\n");
        }
    }
