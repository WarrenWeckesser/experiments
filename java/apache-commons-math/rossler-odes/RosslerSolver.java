
import org.apache.commons.math3.ode.FirstOrderDifferentialEquations;
import org.apache.commons.math3.ode.FirstOrderIntegrator;
import org.apache.commons.math3.ode.nonstiff.DormandPrince853Integrator;
import rosslerODE.RosslerODE;


public class RosslerSolver {

    static void printState(double t, double[] state) {
        double x = state[0];
        double y = state[1];
        double z = state[2];
        String s = String.format("%14.6f  %20.15f %20.15f %20.15f",
                                 t, x, y, z);
        System.out.println(s);
    }

    public static void main(String[] args) {
        FirstOrderIntegrator dp853 = new DormandPrince853Integrator(1.0e-9, 10.0, 1.0e-12, 1.0e-9);
        FirstOrderDifferentialEquations ode = new RosslerODE();
        double[] xyz = new double[] {1.0, 1.0, 1.0};
        double stepsize = 0.08;
        double t = 0.0;
        double t1 = 400.0;

        printState(t, xyz);
        while (t < t1) {
            double ts = t + stepsize;
            if (Math.abs((ts - t1)/ts) < 1e-12) {
                ts = t1;
            }
            dp853.integrate(ode, t, xyz, ts, xyz);
            printState(ts, xyz);
            t = ts;
        }
    }
}
