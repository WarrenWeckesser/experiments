//
// Experiment to convert a ginac expression to ocaml code.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <ginac/ginac.h>
#include <string>
#include <vector>

using std::string;

static string print_ocaml(const GiNaC::ex & e)
{
    using GiNaC::is_a, GiNaC::ex_to;
    string ocaml;

    if (is_a<GiNaC::numeric>(e)) {
        std::stringstream buf;
        buf << e.evalf();
        ocaml += buf.str();
        return ocaml;
    }
    else if (is_a<GiNaC::symbol>(e)) {
        ocaml += ex_to<GiNaC::symbol>(e).get_name();;
        return ocaml;
    }
    else if (is_a<GiNaC::constant>(e)) {
        // Note: GiNaC::constant doesn't have a get_name() method, so use
        // a stringstream.
        std::stringstream buf;
        buf << e;
        string name = buf.str();
        if (name == "Pi") {
            // Replace "Pi" with "pi".
            // TODO: Check for other constants that should be renamed.
            name = "pi";
        }
        ocaml += name;
        return ocaml;
    }
    ocaml += "(";
    if (is_a<GiNaC::power>(e)) {
        if (e.op(1) == -1) {
            ocaml += "1.0 /. ";
            ocaml += print_ocaml(e.op(0));
        }
        else if (e.op(1) == 0.5) {
            ocaml += "sqrt ";
            ocaml += print_ocaml(e.op(0));
        }
        else {
            ocaml += print_ocaml(e.op(0));
            ocaml += " ** ";
            ocaml += print_ocaml(e.op(1));
        }
    }
    else if (is_a<GiNaC::add>(e)) {
        ocaml += print_ocaml(e.op(0));
        for (int i = 1; i < e.nops();  ++i) {
            ocaml += " +. ";
            ocaml += print_ocaml(e.op(i));
        }
    }
    else if (is_a<GiNaC::mul>(e)) {
        string mul;

        if (e.nops() == 1) {
            // I'm not sure ginac will ever create a mul expression with just
            // one element, but handle it here just in case...
            mul = print_ocaml(e.op(0));
        }
        else {
            bool negate = false;
            bool oneout = false;
            for (int i = 0; i < e.nops();  ++i) {
                auto opi = e.op(i);
                if (opi == -1) {
                    negate = !negate;
                }
                else if (is_a<GiNaC::power>(opi) && opi.op(1) == -1 && oneout) {
                    // Ginac stores an expression such as "x/(y*z)" as
                    // "x * pow(y, -1) * pow(z, -1)" (or something similar).
                    // Here operands of the form "pow(expr, -1)" are added to
                    // the ocaml string as division by expr.
                    mul += " /. ";
                    mul += print_ocaml(opi.op(0));
                }
                else {
                    if (oneout) {
                        mul += " *. ";
                    }
                    mul += print_ocaml(opi);
                    oneout = true;
                }
            }
            if (negate) {
                mul = "~-. " + mul;
            }
        }
        ocaml += mul;
    }
    else if (is_a<GiNaC::function>(e)) {
        ocaml += ex_to<GiNaC::function>(e).get_name();
        ocaml += " ";
        for (size_t i = 0; i < e.nops(); i++) {
            ocaml += print_ocaml(e.op(i));
            if (i != e.nops() - 1) {
                ocaml += " ";
            }
        }
    }
    else {
        // TODO: Figure out which other ginac classes require special handling.
        // For now, output whatever ginac writes to a stringstream.
        std::stringstream buf;
        buf << e;
        ocaml += buf.str();
    }
    ocaml += ")";

    return ocaml;
}

int main()
{
    using namespace GiNaC;  // symbol, ex, pow, etc.

    symbol x("x"), y("y");
    std::vector<ex> exprs = {
        -Pi * x * y,
        pow(3, x) - 2 * sin(-y / Pi) * cos(x - y - 1),
        (x+2)/(-y*(y - 1)*pow(y - 2, 2)) * (-(x+1)),
        atan2(x, -y),
        pow(x, 2*y)/(-y)*(Pi),
        exp(y - x)*sqrt(Pi*y + 1)
    };
    std::ofstream out("out.ml");
    out << "(* This file was generated by a program.  Do not edit! *)\n\n";
    out << "open Float\n";
    out << std::endl;
    out << "let x = 1.0;;\nlet y = 3.0;;\n";
    for (ex expr : exprs) {
        string ocaml = print_ocaml(expr);
        std::cout << "ginac: " << expr << std::endl;
        double result = ex_to<numeric>(evalf(expr.subs(x == 1.0).subs(y == 3.0) )).to_double();
        std::cout << "       [" << std::setprecision(12) << result << "]" << std::endl;
        std::cout << "ocaml: " << ocaml << std::endl;
        std::cout << std::endl;
        out << "Printf.printf \"%F\\n\" " << ocaml << ";;" << std::endl;
    }
    out.close();
    std::cerr << "Run 'ocaml out.ml' to see the values computed by ocaml.\n";
    return 0;
}
