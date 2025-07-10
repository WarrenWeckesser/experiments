//
// Experiment to convert a ginac expression to ocaml code.
//

#include <iostream>
#include <sstream>
#include <ginac/ginac.h>
#include <string>

//using namespace std;
using std::string;

static string print_ocaml(const GiNaC::ex & e)
{
    using GiNaC::is_a, GiNaC::ex_to;
    string ocaml;

    // std::cerr << ":: " << e << std::endl;

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
        if (e.nops() == 2 && e.op(1) == -1) {
            // Replace mul(expr, -1) with "~-. expr".
            ocaml += "~-. ";
            ocaml += print_ocaml(e.op(0));
        }
        else if (e.nops() == 2 && e.op(0) == -1) {
            // Replace mul(-1, expr) with "~-. expr".
            ocaml += "~-. ";
            ocaml += print_ocaml(e.op(1));
        }
        else {
            // Ginac stores an expression such as "x/(y*z)" as
            // "x * pow(y, -1) * pow(z, -1)" (or something similar).
            // In this loop, operands of the form "pow(expr, -1)" are
            // added to the ocaml string as division by expr.
            ocaml += print_ocaml(e.op(0));
            for (int i = 1; i < e.nops();  ++i) {
                auto opi = e.op(i);
                if (is_a<GiNaC::power>(opi) && opi.op(1) == -1) {
                    ocaml += " /. ";
                    ocaml += print_ocaml(opi.op(0));
                }
                else {
                    ocaml += " *. ";
                    ocaml += print_ocaml(opi);
                }
            }
        }
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
    using namespace GiNaC;

    symbol x("x"), y("y");
    // ex expr = pow(3, x) - 2 * sin(-1 * y / Pi) * cos(x - y - 1);
    // ex expr = (x+2)/(-y*(y - 1)*(y - 2)) * (x+1);
    ex expr = atan2(x, -y);
    string ocaml = print_ocaml(expr);
    std::cout << "ginac: " << expr << std::endl;
    std::cout << "ocaml: " << ocaml << std::endl;
    return 0;
}