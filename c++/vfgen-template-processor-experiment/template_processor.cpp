
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace std;

typedef pair<int, int> matrix_shape_t;

typedef variant<string,
                vector<string>,
                vector<vector<string>>
                > mixed_t;

typedef map<string, mixed_t> symbol_table_t;

string matlab_template = R"(
function vf_ = {{NAME}}_vf(t, x_, p_)
{@    {{CONSTANT_NAMES}} = {{CONSTANT_VALUES}};@}
{@    {{PARAMETER_NAMES}} = p_({{#PARAMETER_NAMES}});@}
{@    {{STATE_NAMES}} = x_({{#STATE_NAMES}});@}
{@    {{EXPRESSION_NAMES}} = {{EXPRESSION_VALUES}};@}
    vf_ = zeros({{NV}},1);
{@    vf_({{#FORMULAS}}) = {{FORMULAS}};@}
end
)";

string radau5_template = R"(
c
c {{NAME}}_rhs.f
c
c Vector field functions for the vector field '{{NAME}}'
c These functions are to be used with the Fortran ODE solver RADAU5.
c
c {{VFGEN_COMMENT1}}
c {{VFGEN_COMMENT2}}
c

      subroutine {{NAME}}_rhs(n_, {{INDVAR}}, y_, f_, rpar_, ipar_)
      implicit none
      integer n_, ipar_
      double precision {{INDVAR}}, y_, f_, rpar_
      dimension y_({{NV}}), f_({{NV}}), rpar_({{NP}})
{@      double precision {{CONSTANT_NAMES}}@}
{@      double precision {{PARAMETER_NAMES}}@}
{@      double precision {{EXPRESSION_NAMES}}@}
{@      double precision {{STATE_NAMES}}@}
{@      {{CONSTANT_NAMES}} = {{CONSTANT_VALUES}}@}
{@      {{PARAMETER_NAMES}} = rpar_({{#PARAMETER_NAMES}})@}
{@      {{STATE_NAMES}} = y_({{#STATE_NAMES}})@}
{@      {{EXPRESSION_NAMES}} = {{EXPRESSION_VALUES}}@}
{@      f_({{#FORMULAS}}) = {{FORMULAS}}@}
      return
      end

      subroutine {{NAME}}_jac(n_, {{INDVAR}}, y_, dfy_, ldfy_,
     &                        rpar_, ipar_)
      implicit none
      integer n_, ldfy_, ipar_
      double precision {{INDVAR}}, y_, dfy_, rpar_
      dimension y_({{NV}}), dfy_(ldfy_, {{NV}}), rpar_({{NP}})
{@      double precision {{CONSTANT_NAMES}}@}
{@      double precision {{PARAMETER_NAMES}}@}
{@      double precision {{EXPRESSION_NAMES}}@}
{@      double precision {{STATE_NAMES}}@}
{@      {{CONSTANT_NAMES}} = {{CONSTANT_VALUES}}@}
{@      {{PARAMETER_NAMES}} = rpar_({{#PARAMETER_NAMES}})@}
{@      {{STATE_NAMES}} = y_({{#STATE_NAMES}})@}
{@      dfy_({{#JAC}}) = {{JAC}}@}
      return
      end

      subroutine {{NAME}}_out(nr, told, t, y, cont, lrc, n,
     &                  rpar, ipar, irtrn)
      implicit none
      integer nr, lrc, n, ipar, irtrn
      double precision told, t, y, cont, rpar
      integer i
      dimension y({{NV}}), rpar({{NP}})

      write (6,99) t, (y(i), i = 1, {{NV}})
99    format(1x,f10.5,3E18.10)
      return
      end
)";

string lsoda_template = R"(
      subroutine {{NAME}}_rhs(n_, {{INDVAR}}, y_, f_)
      implicit none
      integer n_
      double precision {{INDVAR}}, y_, f_
      dimension y_({{NV}} + {{NP}}), f_({{NV}})
{@      double precision {{CONSTANT_NAMES}}@}
{@      double precision {{PARAMETER_NAMES}}@}
{@      double precision {{EXPRESSION_NAMES}}@}
{@      double precision {{STATE_NAMES}}@}
{@      {{CONSTANT_NAMES}} = {{CONSTANT_VALUES}}@}
{@      {{PARAMETER_NAMES}} = y_({{NV}}+{{#PARAMETER_NAMES}})@}
{@      {{STATE_NAMES}} = y_({{#STATE_NAMES}})@}
{@      {{EXPRESSION_NAMES}} = {{EXPRESSION_VALUES}}@}
{@      f_({{#FORMULAS}}) = {{FORMULAS}}@}
      return
      end
)";

string gsl_template = R"(
int {{NAME}}_vf(double {{INDVAR}}, const double y_[], double f_[], void *params)
{
    p_ = (double *) params;
{@    const double {{CONSTANT_NAMES}} = {{CONSTANT_VALUES}};@}
{@    const double {{PARAMETER_NAMES}} = p_[{{#PARAMETER_NAMES}}];@}
{@    const double {{STATE_NAMES}} = y_[{{#STATE_NAMES}}];@}
{@    const double {{EXPRESSION_NAMES}} = {{EXPRESSION_VALUES}};@}
{@    f_[{{#FORMULAS}}] = {{FORMULAS}};@}
    return GSL_SUCCESS;
}
)";


int check_vector_names(symbol_table_t table, vector<string> names)
{
    int result = -1;
    for (auto &name : names) {
        auto item = table.find(name);
        if (item == table.end()) {
            cerr << "ERROR: '" << name << "' not in symbol table." << endl;
            return -1;
        }
        if (!holds_alternative<vector<string>>(item->second)) {
            cerr << "ERROR: '" << name << "' does not have type vector<string>." << endl;
            return -1;
        }
        vector<string> value = get<vector<string>>(item->second);
        int len = value.size();
        if (result == -1) {
            result = len;
        }
        else if (len != result) {
            cerr << "ERROR: Lengths of vectors differ." << endl;
            return -1;
        }
    }
    return result;
}

int check_vector_names(symbol_table_t table, vector<string> names1, vector<string> names2)
{
    int len1 = check_vector_names(table, names1);
    int len2 = len1 >= 0 ? check_vector_names(table, names2) : -1;
    if (len1 >= 0 && len2 >= 0 && len1 != len2) {
        cerr << "ERROR: Lengths of vectors differ." << endl;
        return -1;
    }
    return len1;
}

matrix_shape_t check_matrix_names(symbol_table_t table, vector<string> names)
{
    matrix_shape_t result{-1, -1};
    for (auto &name : names) {
        auto item = table.find(name);
        if (item == table.end()) {
            cerr << "ERROR: '" << name << "' not in symbol table." << endl;
            return matrix_shape_t{-1, -1};
        }
        if (!holds_alternative<vector<vector<string>>>(item->second)) {
            cerr << "ERROR: '" << name << "' does not have type vector<vector<string>>." << endl;
            return matrix_shape_t{-1, -1};
        }
        vector<vector<string>> value = get<vector<vector<string>>>(item->second);
        int nrows = value.size();
        int ncols = nrows > 0 ? value[0].size() : 0;
        matrix_shape_t shape{nrows, ncols};
        if (result == matrix_shape_t{-1, -1}) {
            result = shape;
        }
        else if (shape != result) {
            cerr << "ERROR: Matrix sizes differ." << endl;
            return matrix_shape_t{-1, -1};
        }
    }
    return result;
}

matrix_shape_t check_matrix_names(symbol_table_t table, vector<string> names1, vector<string> names2)
{
    matrix_shape_t shape1 = check_matrix_names(table, names1);
    matrix_shape_t shape2 = shape1.first >= 0 ? check_matrix_names(table, names2) : matrix_shape_t{-1, -1};
    if (shape1.first >= 0 && shape2.first >= 0 && shape1 != shape2) {
        cerr << "ERROR: Matrix shapes differ." << endl;
        return matrix_shape_t{-1, -1};
    }
    return shape1;
}

bool in_table(string name, symbol_table_t table)
{
    auto item = table.find(name);
    return !(item == table.end());
}

bool is_vector(string name, symbol_table_t table)
{
    auto item = table.find(name);
    if (item == table.end()) {
        cerr << "ERROR: '" << name << "' not in symbol table." << endl;
        return false;
    }
    return holds_alternative<vector<string>>(item->second);
}

bool is_matrix(string name, symbol_table_t table)
{
    auto item = table.find(name);
    if (item == table.end()) {
        cerr << "ERROR: '" << name << "' not in symbol table." << endl;
        return false;
    }
    return holds_alternative<vector<vector<string>>>(item->second);
}

string process_loops(string text, symbol_table_t table, int index_start)
{
    regex loop_re("\\{\\@([\\w\\W]*?) *\\@\\}");
    regex name_re("\\{\\{([A-Za-z][A-Za-z0-9_]*)\\}\\}");
    regex enum_name_re("\\{\\{\\#([A-Za-z][A-Za-z0-9_]*)\\}\\}");

    size_t start = 0;
    vector<string> expanded;
    for (sregex_iterator i = sregex_iterator(text.begin(), text.end(), loop_re);
         i != sregex_iterator(); ++i)
    {
        smatch m = *i;
        //cout << "start: " << start << "  m.position(): " << m.position() << endl;
        expanded.push_back(text.substr(start, m.position() - start));
        start = m.position() + m.length();
        string match = m.str();

        vector<string> vector_names;
        vector<string> enum_vector_names;
        vector<string> matrix_names;
        vector<string> enum_matrix_names;
        string loop_content = m[1];

        for (sregex_iterator j = sregex_iterator(match.begin(), match.end(), name_re);
             j != sregex_iterator(); ++j)
        {
            smatch m = *j;
            if (is_vector(m[1], table)) {
                vector_names.push_back(m[1]);
            }
            else if (is_matrix(m[1], table)) {
                matrix_names.push_back(m[1]);
            }
        }
        for (sregex_iterator j = sregex_iterator(match.begin(), match.end(), enum_name_re);
             j != sregex_iterator(); ++j)
        {
            smatch m = *j;
            if (is_vector(m[1], table)) {
                enum_vector_names.push_back(m[1]);
            }
            else if (is_matrix(m[1], table)) {
                enum_matrix_names.push_back(m[1]);
            }
            else {
                cout << "ERROR: '" << m[1] << "' is not a vector or matrix name." << endl;
                return ""s;
            }
        }
        int len = check_vector_names(table, vector_names, enum_vector_names);
        // TODO: More validation of sizes of vectors and lengths.

        if (matrix_names.size() > 0 || enum_matrix_names.size() > 0) {
            // The text block contains matrix symbols.
            cerr << "Processing text block containing matrix symbols!" << endl;
            matrix_shape_t shape = check_matrix_names(table, matrix_names, enum_matrix_names);
            if (shape == matrix_shape_t{-1, -1}) {
                return ""s;
            }
            for (int row = 0; row < shape.first; ++row) {
                for (int col = 0; col < shape.second; ++col) {
                    cerr << row << " " << col << endl;
                    string new_loop_content = loop_content;
                    for (int k = 0; k < matrix_names.size(); ++k) { 
                        string name = matrix_names[k];
                        cerr << "name: " << name << endl;
                        string value = get<vector<vector<string>>>(table[name])[row][col];
                        regex name_re("\\{\\{" + name + "\\}\\}");

                        new_loop_content = regex_replace(new_loop_content, name_re, value);
                    }
                    for (int k = 0; k < enum_matrix_names.size(); ++k) { 
                        string name = enum_matrix_names[k];
                        string value = to_string(row + index_start) + ", " + to_string(col + index_start);
                        regex name_re("\\{\\{\\#" + name + "\\}\\}");

                        new_loop_content = regex_replace(new_loop_content, name_re, value);
                    }
                    expanded.push_back(new_loop_content + "\n");
                }
            }
        }
        else {
            // No matrix symbols in the text block.
            for (int j = 0; j < len; ++j) {
                string new_loop_content = loop_content;
                for (int k = 0; k < vector_names.size(); ++k) { 
                    string name = vector_names[k];
                    string value = get<vector<string>>(table[name])[j];
                    regex name_re("\\{\\{" + name + "\\}\\}");

                    new_loop_content = regex_replace(new_loop_content, name_re, value);
                }
                for (int k = 0; k < enum_vector_names.size(); ++k) { 
                    string name = enum_vector_names[k];
                    string value = to_string(j + index_start);
                    regex name_re("\\{\\{\\#" + name + "\\}\\}");

                    new_loop_content = regex_replace(new_loop_content, name_re, value);
                }
                expanded.push_back(new_loop_content + "\n");
            }
        }
    }
    expanded.push_back(text.substr(start, text.size() - start));
    string result;
    for (auto &part: expanded) {
        result += part;
    }
    return result;
}

string process_names(const string &text, const symbol_table_t &table)
{
    string result = text;
    for (const auto& [name, replacement] : table) {
        if (holds_alternative<string>(replacement)) {
            string value = get<string>(replacement);
            regex re("\\{\\{" + name + "\\}\\}");
            result = regex_replace(result, re, value);
        }
   }
   return result;
}

string process(const string &text, const symbol_table_t &table, int start_index)
{
    string intermediate = process_loops(text, table, start_index);
    return process_names(intermediate, table);
}

int main()
{
    symbol_table_t table{
        {"NAME", "testcase"},
        {"INDVAR", "t"},
        {"CONSTANT_NAMES", vector<string>{"g", "pi", "Zero"}},
        {"CONSTANT_VALUES", vector<string>{"9.81", "3.13159265", "0"}},
        {"PARAMETER_NAMES", vector<string>{"alpha", "beta", "r0"}},
        {"EXPRESSION_NAMES", vector<string>{"sum", "alphaSI", "betaI"}},
        {"EXPRESSION_VALUES", vector<string>{"S+I_+R+Zero", "alpha*S*I_", "beta*I_"}},
        {"NV", "3"},
        {"NP", "3"},
        {"STATE_NAMES", vector<string>{"S", "I_", "R"}},
        {"FORMULAS", vector<string>{"-alphaSI", "alphaSI - betaI", "betaI"}},
        {"JAC", vector<vector<string>>{vector<string>{"-alpha*I_", "-S*alpha", "0.0D0"},
                                       vector<string>{"alpha*I_", "alpha*S - beta", "0.0D0"},
                                       vector<string>{"0.0D0", "beta", "0.0D0"}}},
        {"VFGEN_COMMENT1", "This file was generated by the program VFGEN, version: 2.6.0.dev4"},
        {"VFGEN_COMMENT2", "Generated on 12-Sep-2024 at 10:27"}
    };

    // string gsl_result = process(gsl_template, table, 0);
    // cout << "GSL" << endl;
    // cout << "----------------------------------------------------------------" << endl;
    // cout << gsl_result << endl;
    // cout << "----------------------------------------------------------------" << endl;
    // string matlab_result = process(matlab_template, table, 1);
    // cout << "MATLAB" << endl;
    // cout << "----------------------------------------------------------------" << endl;
    // cout << matlab_result << endl;
    // cout << "----------------------------------------------------------------" << endl;
    // cout << endl;
    // string lsoda_result = process(lsoda_template, table, 1);
    // cout << "LSODA" << endl;
    // cout << "----------------------------------------------------------------" << endl;
    // cout << lsoda_result << endl;
    // cout << "----------------------------------------------------------------" << endl;
    // cout << endl;
    string radau5_result = process(radau5_template, table, 1);
    cout << "RADAU5" << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << radau5_result << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << endl;
}
