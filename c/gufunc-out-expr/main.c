
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#define OWL_PARSER_IMPLEMENTATION
#include "dim-expr-parser.h"

#include "int64_arithmetic.h"


static char *opcode_names[] = {
    [0]                = "END",
    [PARSED_INT]       = "PUSH CONSTANT",
    [PARSED_ABS]       = "ABS",
    [PARSED_MAX]       = "MAXIMUM",
    [PARSED_MIN]       = "MINIMUM",
    [PARSED_ADD]       = "ADD" ,
    [PARSED_SUBTRACT]  = "SUBTRACT",
    [PARSED_POWER]     = "POWER",
    [PARSED_DIVIDE]    = "DIVIDE",
    [PARSED_NEGATE]    = "NEGATE",
    [PARSED_PARENS]    = "PARENS",  // A parse node, but not an opcode.
    [PARSED_MULTIPLY]  = "MULTIPLY",
    [PARSED_VARIABLE]  = "PUSH VARIABLE",
    [PARSED_REMAINDER] = "REMAINDER",
};


#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define ABS(X) (((X) < 0) ? -(X) : (X))

//
// The return value of get_names must be freed.
// The memory pointed to by names is modified
// by this function.
//
char **
get_names(char *names, int *num_names)
{
    char *p = names;
    int count = 0;
    char *name_start = NULL;

    *num_names = 0;

    // First pass counts the number of names.
    p = names;
    while(*p) {
        if (*p == ' ') {
            if (name_start != NULL) {
                ++count;
                name_start = NULL;
            }
        }
        else {
            if (name_start == NULL) {
                name_start = p;
            }
        }
        ++p;
    }
    if (name_start != NULL) {
        ++count;
    }
    if (count == 0) {
        *num_names = 0;
        return NULL;
    }

    char **name_ptrs = malloc(count*sizeof(char *));
    if (name_ptrs == NULL) {
        *num_names = -1;
        return NULL;
    }

    // Second pass; record the start of each name in the
    // name_ptrs array.
    p = names;
    count = 0;
    name_start = NULL;
    while(*p) {
        if (*p == ' ') {
            if (name_start != NULL) {
                name_ptrs[count] = name_start;
                ++count;
                name_start = NULL;
                *p = 0;
            }
        }
        else {
            if (name_start == NULL) {
                name_start = p;
            }
        }
        ++p;
    }
    if (name_start != NULL) {
        name_ptrs[count] = name_start;
        ++count;
    }

    *num_names = count;
    return name_ptrs;
}

//
// The first argument of lookup_name is not null terminated.  The
// length is given as the second argument.
//
int lookup_name(const char *name, int len_name, char **names, int num_names)
{
    for (int k = 0; k < num_names; ++k) {
        if (len_name == strlen(names[k])) {
            if (strncmp(name, names[k], len_name) == 0) {
                return k;
            }
        }
    }
    return -1;
}

void print_range(const char *p, int from, int to)
{
    for (int i = from; i < to; ++i) {
        putchar(p[i]);
    }
}


typedef struct instruction {
    int64_t opcode;
    int64_t argument;
} instruction;

#define PARSE_OK                      0
#define PARSE_ERROR_INVALID_VARIABLE -1

static int
create_instructions_recursive(instruction *instructions, int *num_instructions,
                              struct parsed_expr expr,
                              char **names, int num_names,
                              char *input)
{
    int status;
    struct parsed_integer parsed_int;
    struct parsed_identifier parsed_id;


    switch (expr.type) {

        case PARSED_INT:
            parsed_int = parsed_integer_get(expr.integer);
            instructions[*num_instructions].opcode = PARSED_INT;
            instructions[*num_instructions].argument = parsed_int.integer;
            ++*num_instructions;
            break;

        case PARSED_ABS:
            status = create_instructions_recursive(instructions, num_instructions,
                                                   parsed_expr_get(expr.arg1),
                                                   names, num_names, input);
            if (status < 0) {
                return status;
            }
            instructions[*num_instructions].opcode = expr.type;
            ++*num_instructions;
            break;

        case PARSED_MAX:
        case PARSED_MIN:
            status = create_instructions_recursive(instructions, num_instructions,
                                                   parsed_expr_get(expr.arg1),
                                                   names, num_names, input);
            if (status < 0) {
                return status;
            }
            status = create_instructions_recursive(instructions, num_instructions,
                                                   parsed_expr_get(expr.arg2),
                                                   names, num_names, input);
            if (status < 0) {
                return status;
            }
            instructions[*num_instructions].opcode = expr.type;
            ++*num_instructions;
            break;

        case PARSED_ADD:
        case PARSED_SUBTRACT:
        case PARSED_POWER:
        case PARSED_MULTIPLY:
        case PARSED_DIVIDE:
        case PARSED_REMAINDER:
            status = create_instructions_recursive(instructions, num_instructions,
                                                   parsed_expr_get(expr.left),
                                                   names, num_names, input);
            if (status < 0) {
                return status;
            }
            status = create_instructions_recursive(instructions, num_instructions,
                                                   parsed_expr_get(expr.right),
                                                   names, num_names, input);
            if (status < 0) {
                return status;
            }
            instructions[*num_instructions].opcode = expr.type;
            ++*num_instructions;
            break;

        case PARSED_PARENS:
            status = create_instructions_recursive(instructions, num_instructions,
                                                   parsed_expr_get(expr.expr),
                                                   names, num_names, input);
            if (status < 0) {
                return status;
            }
            break;

        case PARSED_NEGATE:
            status = create_instructions_recursive(instructions, num_instructions,
                                                   parsed_expr_get(expr.operand),
                                                   names, num_names, input);
            if (status < 0) {
                return status;
            }
            instructions[*num_instructions].opcode = expr.type;
            ++*num_instructions;        
            break;

        case PARSED_VARIABLE:
            parsed_id = parsed_identifier_get(expr.identifier);
            int k = lookup_name(parsed_id.identifier, parsed_id.length,
                                names, num_names);
            if (k < 0) {
                printf("unknown variable name '");
                print_range(parsed_id.identifier, 0, parsed_id.length);
                printf("'\n");
                return PARSE_ERROR_INVALID_VARIABLE;
            }
            instructions[*num_instructions].opcode = expr.type;
            instructions[*num_instructions].argument = k;
            ++*num_instructions;
            break;
    }
    return PARSE_OK;
}

#define CREATE_STATUS_OK           0
#define CREATE_STATUS_NO_MEMORY   -1
#define CREATE_STATUS_PARSE_ERROR -2


//
// Error codes
//
// *perror                     Meaniing
// -----------------------     -----------------------------------------------------
// CREATE_STATUS_NO_MEMORY     out of memory (calloc failed)
// CREATE_STATUS_PARSE_ERROR   parsed failed to parse the expression
//
//
instruction *
create_instructions(struct owl_tree *tree, char **names, int num_names, char *input,
                    int *perror)
{
    *perror = CREATE_STATUS_OK;
    int status;
    int num_instructions = 0;

    // Use strlen(input)+1 as an overestimate of the size of the array
    // of instructions.
    instruction *instructions = calloc(strlen(input)+1, sizeof(instruction));
    if (instructions == NULL) {
        *perror = CREATE_STATUS_NO_MEMORY;
        return NULL;
    }

    struct parsed_expr expr = owl_tree_get_parsed_expr(tree);
    status = create_instructions_recursive(instructions, &num_instructions,
                                           expr, names, num_names, input);
    if (status != PARSE_OK) {
        free(instructions);
        *perror = CREATE_STATUS_PARSE_ERROR;
        return NULL;
    }
    // Append the last instruction, with opcode 0.
    instructions[num_instructions].opcode = 0;
    instructions[num_instructions].argument = 0;
    ++num_instructions;
    instructions = realloc(instructions, num_instructions*sizeof(instruction));
    return instructions;
}

void
print_instructions(instruction *instructions, char **names, int num_names)
{
    int k = 0;

    printf("Instructions\n");
    printf("opcode  arg   symbolic opcodes and args\n");
    while(instructions[k].opcode != 0) {
        int64_t opcode = instructions[k].opcode;
        printf("%5" PRId64 " ", opcode);
        if (opcode == PARSED_INT || opcode == PARSED_VARIABLE) {
            printf("%5" PRId64 "   ", instructions[k].argument);
        }
        else {
            printf("        ");
        }
        switch (opcode) {
        case 0:
        case PARSED_ABS:
        case PARSED_MAX:
        case PARSED_MIN:
        case PARSED_ADD:
        case PARSED_SUBTRACT:
        case PARSED_POWER:
        case PARSED_MULTIPLY:
        case PARSED_DIVIDE:
        case PARSED_REMAINDER:
        case PARSED_NEGATE:
            printf("%s", opcode_names[opcode]);
            break;
        case PARSED_INT:
            printf("%s %" PRId64, opcode_names[opcode], instructions[k].argument);
            break;
        case PARSED_VARIABLE:
            printf("%s %s", opcode_names[opcode], names[instructions[k].argument]);
            break;
        }
        printf("\n");
        ++k;
    }
}

void
print_stack_recursive(struct parsed_expr expr,
                      char **names, int num_names,
                      char *input)
{
    bool goodname;

    switch (expr.type) {
        case PARSED_INT:
            printf("%s ", opcode_names[expr.type]);
            print_range(input, expr.range.start, expr.range.end);
            printf("\n");
            break;
        case PARSED_ABS:
            print_stack_recursive(parsed_expr_get(expr.arg1), names, num_names, input);
            printf("%s\n", opcode_names[expr.type]);
            break;
        case PARSED_MAX:
        case PARSED_MIN:
            print_stack_recursive(parsed_expr_get(expr.arg1), names, num_names, input);
            print_stack_recursive(parsed_expr_get(expr.arg2), names, num_names, input);
            printf("%s\n", opcode_names[expr.type]);
            break;
        case PARSED_ADD:
        case PARSED_SUBTRACT:
        case PARSED_POWER:
        case PARSED_MULTIPLY:
        case PARSED_DIVIDE:
        case PARSED_REMAINDER:
            print_stack_recursive(parsed_expr_get(expr.left), names, num_names, input);
            print_stack_recursive(parsed_expr_get(expr.right), names, num_names, input);
            printf("%s\n", opcode_names[expr.type]);
            break;
        case PARSED_PARENS:
            print_stack_recursive(parsed_expr_get(expr.expr), names, num_names, input);
            break;
        case PARSED_NEGATE:
            print_stack_recursive(parsed_expr_get(expr.operand), names, num_names, input);
            printf("negate\n");
            break;
        case PARSED_VARIABLE:
            goodname = false;
            int varlen = expr.range.end - expr.range.start;
            int k = lookup_name(input + expr.range.start, varlen,
                                names, num_names);
            if (k >= 0) {
                goodname = true;
                printf("%s %d\n", opcode_names[expr.type], k);
            }
            if (!goodname) {
                printf("bad var name '");
                print_range(input, expr.range.start, expr.range.end);
                printf("'\n");
            }
            break;
    }
}


void print_stack(struct owl_tree *tree, char **names, int num_names, char *input)
{
    struct parsed_expr expr = owl_tree_get_parsed_expr(tree);
    print_stack_recursive(expr, names, num_names, input);
}


//
// Error conditions
//
// *error   Meaning
// ------   ----------------------------------------------------
//   -1     out of memory (calloc failed)
//   -2     program error: at end, the number of items remaining
//          on the stack was not 1.
//   -3     overflow
//   -4     negative power
//   -5     division by 0 (in division or remainder)
//
int64_t
evaluate_instructions(instruction *instructions, int64_t *vars, int *error)
{
    // XXX This function assumes:
    // (1) The arguments in the instructions array that refer to variables
    //     are all valid indices into the vars array.
    // (2) There is at least one instruction in the instructions array, and
    //     the last element in the instructions array has opcode 0.

    int overflow;
    int64_t value;

    *error = 0;

    // XXX Could maintain the num_instructions count externally and
    // accept it as an argument to this function.
    int num_instructions = 0;
    while (instructions[num_instructions].opcode != 0) {
        ++num_instructions;
    }

    // num_instructions is a safe overestimate of the required stack size.
    int64_t *stack = calloc(num_instructions, sizeof(int64_t));
    if (stack == NULL) {
        *error = -1;
        return 0;
    }
    int stack_counter = 0;

    for (int k = 0; k < num_instructions; ++k) {
        switch (instructions[k].opcode) {

        case PARSED_INT:
            stack[stack_counter] = instructions[k].argument;
            ++stack_counter;
            break;
        case PARSED_MAX:
            --stack_counter;
            value = MAX(stack[stack_counter-1], stack[stack_counter]);
            stack[stack_counter-1] = value;
            break;
        case PARSED_MIN:
            --stack_counter;
            value = MIN(stack[stack_counter-1], stack[stack_counter]);
            stack[stack_counter-1] = value;
            break;
        case PARSED_ADD:
            --stack_counter;
            // value = stack[stack_counter-1] + stack[stack_counter];
            value = add_int64(stack[stack_counter-1], stack[stack_counter], &overflow);
            if (overflow != 0) {
                *error = -3;
                break;
            }
            stack[stack_counter-1] = value;
            break;
        case PARSED_SUBTRACT:
            --stack_counter;
            // value = stack[stack_counter-1] - stack[stack_counter];
            value = subtract_int64(stack[stack_counter-1], stack[stack_counter], &overflow);
            if (overflow != 0) {
                *error = -3;
                break;
            }
            stack[stack_counter-1] = value;
            break;
        case PARSED_POWER:
            --stack_counter;
            if (stack[stack_counter] < 0) {
                // Negative power not allowed.
                *error = -4;
                break;
            }
            value = pow_int64(stack[stack_counter-1], stack[stack_counter], &overflow);
            if (overflow != 0) {
                *error = -3;
                break;
            }
            stack[stack_counter-1] = value;
            break;
        case PARSED_MULTIPLY:
            --stack_counter;
            // value = stack[stack_counter-1] * stack[stack_counter];
            value = multiply_int64(stack[stack_counter-1], stack[stack_counter], &overflow);
            if (overflow != 0) {
                *error = -3;
                break;
            }
            stack[stack_counter-1] = value;
            break;
        case PARSED_DIVIDE:
            --stack_counter;
            if (stack[stack_counter] == 0) {
                // Division by 0.
                *error = -5;
                break;
            }
            value = stack[stack_counter-1] / stack[stack_counter];
            stack[stack_counter-1] = value;
            break;
        case PARSED_REMAINDER:
            --stack_counter;
            if (stack[stack_counter] == 0) {
                // Division by 0.
                *error = -5;
                break;
            }
            value = stack[stack_counter-1] % stack[stack_counter];
            stack[stack_counter-1] = value;
            break;
        case PARSED_PARENS:
            // XXX PARSED_PARENS should not appear in the instructions array.
            break;
        case PARSED_NEGATE:
            if (stack[stack_counter-1] == INT64_MIN) {
                // Can't negate INT64_MIN; flag it as an overflow error.
                *error = -3;
                break;
            }
            stack[stack_counter-1] = -stack[stack_counter-1];
            break;
        case PARSED_ABS:
            if (stack[stack_counter-1] == INT64_MIN) {
                // Can't compute absolute value of INT64_MIN;
                // flag it as an overflow error.
                *error = -3;
                break;
            }
            stack[stack_counter-1] = ABS(stack[stack_counter-1]);
            break;
        case PARSED_VARIABLE:
            stack[stack_counter] = vars[instructions[k].argument];
            ++stack_counter;
            break;
        }
        if (*error != 0) {
            break;
        }
    }
    if (*error == 0 && stack_counter != 1) {
        *error = -2;
    }
    if (*error != 0) {
        value = 0;
    }
    else {
        value = stack[0];
    }
    free(stack);
    return value;
}


void print_alloc_failed(void)
{
    printf("failed to allocate memory\n");
}


int demonstrate_evaluation(instruction *instructions,
                           char **var_names,
                           int64_t var_values[],
                           int num_names)
{
    int error;

    printf("\n");
    printf("Demonstrate evaluation...\n");

    // Fill vars with some numbers for the demonstration.
    for (int k = 0; k < num_names; ++k) {
        printf("%s = %" PRId64 "\n", var_names[k], var_values[k]);
    }


    int64_t result = evaluate_instructions(instructions, var_values, &error);
    if (error < 0) {
        printf("evaluate_instructions failed; error = %d: ", error);
        if (error == -1) {
            printf("out of memory");
        }
        if (error == -2) {
            printf("program error; extra items on stack");
        }
        if (error == -3) {
            printf("overflow");
        }
        if (error == -4) {
            printf("negative power");
        }
        if (error == -5) {
            printf("division by 0 (in division or remainder operation)");
        }
        if (error < -5) {
            printf("unknown error");
        }
        printf("\n");
    }
    else {
        printf("evaluate_instructions returned %" PRId64 "\n", result);
    }
    return 0;
}


// For this demo, limit the number of variables.
#define MAX_VARS 8

int main(int argc, char *argv[])
{
    int status;
    //char **name_ptrs;
    int num_names;
    int retval = 0;
    char *var_names[MAX_VARS];
    int64_t var_values[MAX_VARS];

    if (argc < 2) {
        printf("use: ./main expr name=value name=value ...\n");
        printf("expr is a dimension expression.\n");
        printf("and each parameter of the form name=value defines a value \n");
        printf("to be used to evaluate expr.\n");
        printf("Example: ./main \"1 + min(m, n)\" m=5 n=12\n");
        exit(-1);
    }

    num_names = argc - 2;
    if (num_names > MAX_VARS) {
        fprintf(stderr, "too many variable definitions\n");
        exit(-1);
    }
    for (size_t k = 2; k < argc; ++k) {
        char *v = argv[k];
        while (*v) {
            if (*v == '=') {
                break;
            }
            ++v;
        }
        if (!*v) {
            fprintf(stderr, "invalid variable definition: '%s'\n", argv[k]);
            exit(-1);
        }
        char *pe = v;
        *v = '\0';
        var_names[k - 2] = argv[k];
        ++v;
        char *endptr;
        var_values[k - 2] = strtol(v, &endptr, 10);
        if (endptr == v || *endptr != '\0') {
            *pe = '=';
            fprintf(stderr, "invalid value in variable definition: '%s'\n", argv[k]);
            exit(-1);
        }
    }

    struct owl_tree *tree = owl_tree_create_from_string(argv[1]);
    if (owl_tree_get_error(tree, NULL) != ERROR_NONE) {
        owl_tree_print(tree);
        exit(-1);
    }

    instruction *instructions = create_instructions(tree, var_names, num_names,
                                                    argv[1], &status);
    // Done with tree
    owl_tree_destroy(tree);
    if (instructions == NULL) {
        if (status == CREATE_STATUS_NO_MEMORY) {
            print_alloc_failed();
        }
        else {
            printf("failed to create instructions\n");
        }
        exit(-1);
    }

    print_instructions(instructions, var_names, num_names);

    status = demonstrate_evaluation(instructions, var_names, var_values, num_names);
    if (status < 0) {
        print_alloc_failed();
        retval = -1;
    }

    free(instructions);
    return retval;
}
