#include <regex>
#include <set>
#include <string>
#include <vector>

#include "simple_array.h"
#include "abtemplate.h"

using namespace std;


static bool match_pos(string text, string target, size_t pos)
{
    // There is room for optimization in this function!

    size_t target_len = target.size();
    if (target_len == 0) {
        return true;
    }
    if (target_len > text.size()) {
        return false;
    }
    return text.substr(pos, target_len) == target;
}

static void
get_names(const string text,
          const string loop_start, const string loop_end,
          const string name_start, const string name_end,
          symbol_table_t table,
          set<string> &names,
          set<string> &enum_names)
{
    size_t start = 0;
    size_t current = 0;
    size_t block_depth = 0;
    while (current < text.size()) {
        if (block_depth == 0 && match_pos(text, name_start, current)) {
            size_t actual_name_start = current + name_start.size();
            size_t name_current = actual_name_start;
            while (!match_pos(text, name_end, name_current) && name_current < text.size()-name_end.size()) {
                ++name_current;
            }
            // XXX Error if name_current > text.size() - name_end.size()...
            if (text[actual_name_start] == '#') {
                ++actual_name_start;
                enum_names.insert(text.substr(actual_name_start,
                                              name_current - actual_name_start));
            }
            else {
                names.insert(text.substr(actual_name_start,
                                         name_current - actual_name_start));
            }
            current = name_current + name_end.size();
        }
        else if (match_pos(text, loop_start, current)) {
            ++block_depth;
            current += loop_start.size();
        }
        else if (match_pos(text, loop_end, current)) {
            --block_depth;
            current += loop_end.size();
        }
        else {
            ++current;
        }
    }
}

struct text_block {
    string text;
    bool isloop;
};
typedef struct text_block text_block_t;

static vector<text_block_t>
split(const string text, const string loop_start, const string loop_end)
{
    // XXX/TODO: Validation--check for unmatched loop start/end delimiters.
    vector<text_block_t> blocks;
    size_t start = 0;
    size_t current = 0;
    size_t block_depth = 0;
    while (current < text.size()) {
        if (match_pos(text, loop_start, current)) {
            // Found the start of a loop block.
            if (block_depth == 0 && start < current) {
                blocks.push_back(text_block_t{text.substr(start, current - start),
                                              false});
                start = current + loop_start.size();
            }
            ++block_depth;
            current += loop_start.size();
        }
        else if (match_pos(text, loop_end, current)) {
            if (block_depth == 1 && start < current) {
                blocks.push_back(text_block_t{text.substr(start, current - start),
                                              true});
                start = current + loop_end.size();
            }
            --block_depth;
            current += loop_end.size();
        }
        else {
            ++current;
        }
    }
    if (start < current) {
        blocks.push_back(text_block_t{text.substr(start, current - start),
                                      false});
    }
    return blocks;
}


string expand(
    const string text,
    const string loop_start, const string loop_end,
    const string name_start, const string name_end,
    const symbol_table_t table,
    size_t index_start
)
{
    vector<string> expanded;

    // Get top-level names into sets.
    set<string> names_set;
    set<string> enum_names_set;
    get_names(text, loop_start, loop_end, name_start, name_end, table,
              names_set, enum_names_set);

    // Convert sets to vectors.
    vector<string> names;
    for (auto &name: names_set) {
        names.push_back(name);
    }
    vector<string> enum_names;
    for (auto &name: enum_names_set) {
        enum_names.push_back(name);
    }

    // Create a vector of all the names.
    vector<string> all_names = names;
    all_names.insert(all_names.end(), enum_names.begin(), enum_names.end());

    // Get the corresponding arrays associated with each name into a single vector.
    vector<SimpleArray<string>> all_arrays;
    for (auto &name: all_names) {
        all_arrays.push_back(table.at(name));
    }

    // Find the common broadcast shape of all the referenced arrays.
    shape_t common_shape = broadcast_shape(all_arrays);

    // Create a vector of all the referenced arrays broadcast up to the common shape.
    vector<SimpleArray<string>> all_broadcast_arrays;
    for (auto &a: all_arrays) {
        all_broadcast_arrays.push_back(broadcast_to(a, common_shape));
    }

    // Create a symbol table for the broadcast referenced arrays.
    // (Keys are the names, values are the referenced arrays broadcast to the
    // common shape.)
    symbol_table_t bctable;
    for (size_t k = 0; k < all_names.size(); ++k) {
        bctable.emplace(all_names[k], all_broadcast_arrays[k]);
    }

    // Main loop to make the substitutions using regex_replace.
    MultiIndexIterator it{common_shape};
    for (auto &index: it) {
        string new_text = text;
        for (int k = 0; k < names.size(); ++k) { 
            string name = names[k];
            SimpleArray a = bctable.at(name);
            string value = a[index];
            regex name_re("\\{\\{" + name + "\\}\\}");
            new_text = regex_replace(new_text, name_re, value);
        }
        for (int k = 0; k < enum_names.size(); ++k) { 
            string name = enum_names[k];
            SimpleArray a = bctable.at(name);
            string value;
            for (size_t t = 0; t < index.size(); ++t) {
                if (a._index_strides[t] != 0) {
                    if (value.size() > 0) {
                        value = value + ", "s;
                    }
                    value = value + to_string(index[t] + index_start);
                }
            }
            regex name_re("\\{\\{\\#" + name + "\\}\\}");

            new_text = regex_replace(new_text, name_re, value);
        }
        expanded.push_back(new_text + "\n");
    }

    // Split each item in expanded into blocks of text, and recurse on those
    // blocks that are loops (i.e. that were delimited by loop_start/loop_end).
    vector<string> done;
    for (auto &filledin_text: expanded) {
        vector<text_block_t> blocks = split(filledin_text, loop_start, loop_end);
        for (auto &block: blocks) {
            if (block.isloop) {
                string expanded_block = expand(block.text,
                                               loop_start, loop_end,
                                               name_start, name_end,
                                               table, index_start);
                done.push_back(expanded_block);
            }
            else {
                done.push_back(block.text);
            }
        }
    }
    string result;
    for (auto &part: done) {
        result.append(part);
    }
    return result;
}
