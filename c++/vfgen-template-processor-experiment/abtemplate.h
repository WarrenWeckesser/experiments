#ifndef AB_TEMPLATE_H
#define AB_TEMPLATE_H

#include <map>
#include <vector>
#include "simple_array.h"

typedef std::map<std::string, SimpleArray<std::string>> symbol_table_t;

std::string expand(
    const std::string text,
    const std::string loop_start, const std::string loop_end,
    const std::string name_start, const std::string name_end,
    const symbol_table_t table,
    size_t index_start
);

#endif
