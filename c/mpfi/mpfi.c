//
// mpfi -- minor page-fault investigator.
// Copyright (c) 2022 Warren Weckesser.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>


#define INPUT_BUFFER_SIZE 256
#define NUM_SLOTS 20
#define MAX_LOOP 10


long get_nonnegative_integer(char *nptr)
{
    long size;
    char *endptr;
    while (*nptr == ' ') {
        ++nptr;
    }
    if (*nptr == '\0' || *nptr == '\n') {
        printf("missing nonnegative integer\n");
        size = -1;
    }
    else {
        size = strtol(nptr, &endptr, 10);
        if (*endptr != '\0' && *endptr != '\n') {
            printf("invalid nonnegative integer\n");
            size = -1;
        }
    }
    return size;
}

long update_minflt(long *ru_minflt)
{
    struct rusage usage;
    int status = getrusage(RUSAGE_SELF, &usage);
    long new_minflt = usage.ru_minflt - *ru_minflt;
    *ru_minflt = usage.ru_minflt;
    return new_minflt;
}

int main(int argc, char *argv[])
{
    char input_buffer[INPUT_BUFFER_SIZE];
    char *slots[NUM_SLOTS];
    size_t slot_size[NUM_SLOTS];
    struct rusage usage;
    long ru_minflt = 0;
    bool istty = isatty(fileno(stdin));

    if (argc > 1) {
        printf("%s\n", argv[1]);
    }

    update_minflt(&ru_minflt);

    int num_slots_used = 0;
    while (true) {
        if (istty) {
            fputs(": ", stdout);
            fflush(stdout);
        }
        char *s = fgets(input_buffer, INPUT_BUFFER_SIZE, stdin);
        if (s == NULL) {
            break;
        }
        char c = input_buffer[0];
        if (c == '\n' || c == '#') {
            continue;
        }
        if (c == 'x' || c == 'q') {
            break;
        }
        if (c == 'p') {
            printf("%4s  %10s\n", "slot", "size");
            for (int i = 0; i < num_slots_used; ++i) {
                printf("%4d  %10ld\n", i, slot_size[i]);
            }
        }
        else if (c == 'a') {
            if (num_slots_used == NUM_SLOTS) {
                printf("All slots in use.\n");
            }
            else {
                long size = get_nonnegative_integer(&input_buffer[1]);
                if (size >= 0) {
                    slots[num_slots_used] = malloc((size_t) size);
                    if (slots[num_slots_used] == NULL) {
                        printf("malloc failed\n");
                    }
                    else {
                        slot_size[num_slots_used] = (size_t) size;
                        long new_minflt = update_minflt(&ru_minflt);
                        printf("allocated %10ld bytes in slot %2d   [%6ld new minor page faults; brk=%p]\n",
                               size, num_slots_used, new_minflt, sbrk(0));
                        ++num_slots_used;
                    }
                }
            }
        }
        else if (c == 'f') {
            if (num_slots_used == 0) {
                printf("No slots in use.\n");
            }
            else {
                --num_slots_used;
                free(slots[num_slots_used]);
                long new_minflt = update_minflt(&ru_minflt);
                printf("freed slot %2d                           [%6ld new minor page faults; brk=%p]\n",
                       num_slots_used, new_minflt, sbrk(0));
            }
        }
        else if (c == 'w') {
            long slot;
            if (strlen(input_buffer) == 2) {
                if (num_slots_used == 0) {
                    printf("No slots in use.\n");
                    continue;
                }
                slot = num_slots_used - 1;
            }
            else {
                slot = get_nonnegative_integer(&input_buffer[1]);
            }
            if (slot >= 0) {
                if (slot >= num_slots_used) {
                    printf("slot not allocated\n");
                }
                else {
                    char *p = slots[slot];
                    for (int i = 0; i < slot_size[slot]; ++i) {
                        *(p + i) = '!';
                    }
                    long new_minflt = update_minflt(&ru_minflt);
                    printf("wrote to slot %2ld (%10ld bytes)     [%6ld new minor page faults; brk=%p]\n",
                           slot, slot_size[slot], new_minflt, sbrk(0));
                }
            }
        }
        else if (c == 'r') {
            long new_minflt = update_minflt(&ru_minflt);
            printf("%ld new minor page faults\n", new_minflt);
        }
        else if (c == 'o') {
            printf("%s", &input_buffer[2]);
        }
        else if (c == '?' || c == 'h') {
            printf("Command    Action\n");
            printf("---------  --------------------------------------------------------------\n");
            printf("a <size>   Allocate <size> bytes, store in the top slot, and report\n");
            printf("           new minor page faults.\n");
            printf("w <slot>   Write to each byte in the memory of slot <slot>.  The <slot>\n");
            printf("           parameter is optional.  If not given, the default is the top slot.\n");
            printf("           and report new minor page faults.\n");
            printf("f          Free the memory allocated in top slot, and report new minor\n");
            printf("           page faults.\n");
            printf("r          Report number of new minor page faults since last report.\n");
            printf("p          Print the sizes of the currently allocated slots.\n");
            printf("o <msg>    Output <msg> to the terminal.\n");
            printf("#          Ignored (used for comments).\n");
            printf("?          Show this information.\n");
            printf("h          Show this information.\n");
            printf("q          Exit the program.\n");
            printf("x          Exit the program.\n");
        }
        else {
            printf("unknown command '%c'\n", c);
        }
    }
}
