
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>

#define OWL_PARSER_IMPLEMENTATION
#include "dim-expr-parser.h"


static char *opcode_names[] = {
    [0]                = "END",
    [PARSED_INT]       = "PUSH CONSTANT",
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


//
// Note: check_names modifies the string names by putting
// null characters at the end of each space-separated name
// that it finds.
char **
check_names(char *names, int *num_names)
{
    char *p = names;
    int count = 0;
    char *name_start = NULL;
    // First pass counts the number of names.
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
    char **name_ptrs = malloc(count*sizeof(char *));
    if (name_ptrs == NULL) {
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
int lookup_name(char *name, int len_name, char **names, int num_names)
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

void print_range(char *p, int from, int to)
{
    for (int i = from; i < to; ++i) {
        putchar(p[i]);
    }
}


typedef struct instruction {
    int64_t opcode;
    int64_t argument;
} instruction;

void
create_instructions_recursive(instruction *instructions, int *num_instructions,
                              struct parsed_expr expr,
                              char **names, int num_names,
                              char *input)
{
    int64_t value;
    bool goodname;

    switch (expr.type) {

        case PARSED_INT:
            value = strtol(input + expr.range.start, NULL, 10);
            instructions[*num_instructions].opcode = PARSED_INT;
            instructions[*num_instructions].argument = value;
            ++*num_instructions;
            break;

        case PARSED_MAX:
        case PARSED_MIN:
            create_instructions_recursive(instructions, num_instructions, parsed_expr_get(expr.arg1), names, num_names, input);
            create_instructions_recursive(instructions, num_instructions, parsed_expr_get(expr.arg2), names, num_names, input);
            instructions[*num_instructions].opcode = expr.type;
            ++*num_instructions;
            break;

        case PARSED_ADD:
        case PARSED_SUBTRACT:
        case PARSED_POWER:
        case PARSED_MULTIPLY:
        case PARSED_DIVIDE:
        case PARSED_REMAINDER:
            create_instructions_recursive(instructions, num_instructions, parsed_expr_get(expr.left), names, num_names, input);
            create_instructions_recursive(instructions, num_instructions, parsed_expr_get(expr.right), names, num_names, input);
            instructions[*num_instructions].opcode = expr.type;
            ++*num_instructions;
            break;

        case PARSED_PARENS:
            create_instructions_recursive(instructions, num_instructions, parsed_expr_get(expr.expr), names, num_names, input);
            break;

        case PARSED_NEGATE:
            create_instructions_recursive(instructions, num_instructions, parsed_expr_get(expr.operand), names, num_names, input);
            instructions[*num_instructions].opcode = expr.type;
            ++*num_instructions;        
            break;

        case PARSED_VARIABLE:
            goodname = false;
            int varlen = expr.range.end - expr.range.start;
            int k = lookup_name(input + expr.range.start, varlen,
                                names, num_names);
            if (k >= 0) {
                goodname = true;
                instructions[*num_instructions].opcode = expr.type;
                instructions[*num_instructions].argument = k;
                ++*num_instructions;
            }
            // XXX Fix the error handling!
            if (!goodname) {
                printf("bad var name '");
                print_range(input, expr.range.start, expr.range.end);
                printf("'\n");
            }
            break;
    }
}

instruction *
create_instructions(struct owl_tree *tree, char **names, int num_names, char *input)
{
    int num_instructions = 0;

    // Use strlen(input)+1 as an overestimate of the size of the array
    // of instructions.
    instruction *instructions = calloc(strlen(input)+1, sizeof(instruction));
    if (instructions == NULL) {
        return NULL;
    }

    struct parsed_expr expr = owl_tree_get_parsed_expr(tree);
    create_instructions_recursive(instructions, &num_instructions,
                                  expr, names, num_names, input);
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

    while(instructions[k].opcode != 0) {
        int64_t opcode = instructions[k].opcode;
        printf("%5lld ", opcode);
        if (opcode == PARSED_INT || opcode == PARSED_VARIABLE) {
            printf("%5lld   ", instructions[k].argument);
        }
        else {
            printf("        ");
        }
        switch (opcode) {
        case 0:
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
            printf("%s %5lld", opcode_names[opcode], instructions[k].argument);
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


int64_t
intpow(int64_t b, int64_t p)
{
  return (p == 0) ? 1 : (p == 1) ? b : ((p % 2) ? b : 1) * intpow(b * b, p / 2);
}


int64_t
evaluate_instructions(instruction *instructions, int64_t *vars, int *error)
{
    // XXX This function assumes:
    // (1) The arguments in the instructions array that refer to variables
    //     are all valid indicees into the vars array.
    // (2) There is at least one instruction in the instructions array, and
    //     the instructions array is ended with an instruction that has
    //     opcode 0.

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
            value = stack[stack_counter-1] + stack[stack_counter];
            stack[stack_counter-1] = value;
            break;
        case PARSED_SUBTRACT:
            --stack_counter;
            value = stack[stack_counter-1] - stack[stack_counter];
            stack[stack_counter-1] = value;
            break;
        case PARSED_POWER:
            --stack_counter;
            value = intpow(stack[stack_counter-1], stack[stack_counter]);
            stack[stack_counter-1] = value;
            break;
        case PARSED_MULTIPLY:
            --stack_counter;
            value = stack[stack_counter-1] * stack[stack_counter];
            stack[stack_counter-1] = value;
            break;
        case PARSED_DIVIDE:
            --stack_counter;
            value = stack[stack_counter-1] / stack[stack_counter];
            stack[stack_counter-1] = value;
            break;
        case PARSED_REMAINDER:
            --stack_counter;
            value = stack[stack_counter-1] % stack[stack_counter];
            stack[stack_counter-1] = value;
            break;
        case PARSED_PARENS:
            // XXX PARSED_PARENS should not appear in the instructions array.
            break;
        case PARSED_NEGATE:
            stack[stack_counter-1] = -stack[stack_counter-1];
            break;
        case PARSED_VARIABLE:
            stack[stack_counter] = vars[instructions[k].argument];
            ++stack_counter;
            break;
        }
    }
    if (stack_counter != 1) {
        *error = -2;
        free(stack);
        return 0;
    }
    value = stack[0];
    free(stack);
    return value;
}

int main(int argc, char *argv[])
{
    struct owl_tree *tree;
    int error;
    char **name_ptrs;
    int num_names;

    if (argc < 3) {
        printf("use: ./main names expr\n");
        printf("where names is a space-dilimited list of dimension names\n");
        printf("and expr is a dimension expression.\n");
        printf("Example: ./main \"m n\" \"1 + min(m, n)\"\n");
        exit(-1);
    }

    // Make a copy of argv[1].  Some characters in the string
    // will be modified in the call to check_names().
    char *names = malloc(strlen(argv[1])+1);
    if (names == NULL) {
        printf("malloc failed\n");
        exit(-1);
    }
    memcpy(names, argv[1], strlen(argv[1])+1);

    name_ptrs = check_names(names, &num_names);
    if (name_ptrs == NULL || num_names < 1) {
        exit(-1);
    }

    tree = owl_tree_create_from_string(argv[2]);
    if (owl_tree_get_error(tree, NULL) != ERROR_NONE) {
        owl_tree_print(tree);
        exit(-1);
    }

    // print_stack(tree, name_ptrs, num_names, argv[2]);

    instruction *instructions = create_instructions(tree, name_ptrs, num_names, argv[2]);

    printf("Instructions\n");
    printf("opcode  arg   symbolic opcodes and args\n");
    print_instructions(instructions, name_ptrs, num_names);

    printf("\n");
    printf("Demonstrate evaluation...\n");
    int64_t *vars = calloc(num_names, sizeof(int64_t));
    for (int k = 0; k < num_names; ++k) {
        vars[k] = 5*k + 3;
        printf("%s = %lld\n", name_ptrs[k], vars[k]);
    }
    int64_t result = evaluate_instructions(instructions, vars, &error);
    if (error < 0) {
        printf("evaluate_instructions failed; error = %d\n", error);
    }
    else {
        printf("evaluate_instructions returned %lld\n", result);
    }

    free(instructions);

    owl_tree_destroy(tree);

    return 0;
}
