

package rosslerODE;

import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;


public class RosslerODE implements FirstOrderDifferentialEquations {
    // Rossler parameters
    private double a;
    private double b;
    private double c;

    public RosslerODE(double a, double b, double c) {
        this.a = a;
        this.b = b;
        this.c = c;
    }

    public RosslerODE() {
        a = 0.17;
        b = 0.4;
        c = 8.5;
    }

    public int getDimension() {
        return 3;
    }

    public void computeDerivatives(double t, double[] state, double[] deriv) {
        double x = state[0];
        double y = state[1];
        double z = state[2];
        deriv[0] = -y - z;
        deriv[1] = x + a*y;
        deriv[2] = b - z*(c - x);
    }
}
