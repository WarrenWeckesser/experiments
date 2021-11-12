      PROGRAM SIRDELAY2
C
C     This file contains a driver program and skeletons for the
C     necessary user provided subroutines for DKLAG6.
C
C     The problem solved here is the "delay" version of the
C     SIR model, in which it takes exactly h time units for
C     an individual to recover:
C
C           S'(t) = -r*S(t)*I(t)
C           I'(t) = r*S(t)*I(t) - r*S(t-h)*I(t-h)
C           R'(t) = r*S(t-h)*I(t-h)
C
C     The initial condition is a jump at t=0 in I(t) from
C     zero to I_0.  This is handled by using the explicit
C     solution to the ODEs
C           S'(t) = -r*S(t)*I(t)
C           I'(t) = r*S(t)*I(t)
C     with I(-h)=I0, S(-h)=1-I (and R(-h)=0) as the initial
C     condition for the DKLAG6 solver on [-h,0].
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
C     WORK ARRAYS
C
      DIMENSION DW(5000), IW(5000), DQ(80000)
C
C     USER ARRAYS
C
C     RPAR(1) = r
C     RPAR(2) = h  (the recovery time)
C     IPAR is not used.
C
      DIMENSION RPAR(2), IPAR(1)
C
C     PROBLEM DEPENDENT SUBROUTINES
C
      EXTERNAL INITAL, DERIVS, GUSER, BETA, YINIT, DYINIT, 
     *         CHANGE, DPRINT
C
C     LWDIM = DIMENSION FOR DW ABOVE
C     LQDIM = DIMENSION FOR DQ
C     LIDIM = DIMENSION FOR IW
C
      DATA LWDIM /5000/, LIDIM /5000/, LQDIM /80000/
C
C     LIN = LOGICAL NUMBER FOR INPUT
C
      DATA LIN /5/
C
C     LOUT = LOGICAL NUMBER FOR OUTPUT
C
      DATA LOUT /6/
C
C     OPEN THE OUTPUT FILES
C
      OPEN (6, FILE='sir2.dat')
C
C     NUMBER OF DIFFERENTIAL EQUATIONS
C
      N = 3
C
C     DIMENSIONS OF WORK ARRAYS
C
      LW = LWDIM
      LI = LIDIM
      LQ = LQDIM
C
C     CALL THE DELAY SOLVER
C
      CALL DKLAG6(INITAL, DERIVS, GUSER, CHANGE, DPRINT, BETA, 
     *            YINIT, DYINIT, N, DW, LW, IW, LI, DQ, LQ, 
     *            IFLAG, LIN, LOUT, RPAR, IPAR)
C
C     SUCCESSFUL INTEGRATION
C
      IF (IFLAG .EQ.0 ) THEN
C         WRITE (LOUT,99999)
         WRITE (0,99999)
         WRITE (0,99997)
      ENDIF
C
C     UNSUCCESSFUL INTEGRATION
C
      IF (IFLAG .NE.0 ) THEN
         WRITE (LOUT,99998)
         WRITE (0,99998)
         WRITE (0,99997)
      ENDIF
C
99999 FORMAT (/, ' NORMAL RETURN OCCURRED FROM DKLAG6 ')
99998 FORMAT (/, ' ABNORMAL RETURN OCCURRED FROM DKLAG6 ')
99997 FORMAT (/, ' The output was written to file sir2.dat')
C
      STOP
      END
C
C
      SUBROUTINE BETA (N, T, Y, BVAL, RPAR, IPAR)
C
C
C     CALCULATE THE SYSTEM DELAY TIMES
C
C
C     INPUT -
C
C     N                = number of equations
C     T                = independent variable (time)
C     Y(I), I=1,...,N  = solution at time T
C     RPAR(*), IPAR(*) = user arrays passed via DKLAG6
C
C     This subroutine must define the system delay times
C     BVAL(I), I=1,...,N. For each T, BVAL(I) must be
C     less than or equal to T if the integration goes
C     from left to right and greater than or equal to
C     T if it goes from right to left. BVAL(I) is the
C     value of the independent variable used in the
C     approximation of the I-th delay solution value.
C     The actual delay for this component is equal to
C     BVAL(I) - T. Define BVAL(I) = T for any component
C     which does not involve a delay (in particular,
C     for ODEs).
C
C     You may terminate the integration by setting N to
C     a negative value when BETA is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      DIMENSION BVAL(*), Y(*), RPAR(*), IPAR(*)
C
      BVAL(1) = T - RPAR(2)
      BVAL(2) = T - RPAR(2)
      BVAL(3) = T - RPAR(2)
C
      RETURN
      END
C
C
      SUBROUTINE DERIVS (N, T, Y, YLAG, DYLAG, DY, RPAR, IPAR)
C
C
C     CALCULATE THE SYSTEM DERIVATIVES
C
C
C     INPUT -
C
C     N                        = number of odes
C     T                        = independent variable (time)
C     Y(I), I=1,...,N          = solution at time T
C     YLAG(I), I=1,...,N       = delayed solution
C     DYLAG(I), I=1,...,N      = delayed derivative
C     RPAR(*), IPAR(*)         = user arrays passed via DKLAG6
C
C     This subroutine must define the system derivatives
C     DY(I), I=1,...,N.
C
C     You may terminate the integration by setting N to
C     a negative value when DERIVS is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
C
      DIMENSION Y(*), DY(*), YLAG(*), DYLAG(*), RPAR(*), IPAR(*)
C
      DY(1) = -RPAR(1)*Y(1)*Y(2)
      DY(2) =  RPAR(1)*Y(1)*Y(2) - RPAR(1)*YLAG(1)*YLAG(2)
      DY(3) =  RPAR(1)*YLAG(1)*YLAG(2)
C
      RETURN
      END
C
C
      SUBROUTINE INITAL (LOUT, LIN, N, NGUSER, TOLD, TFINAL, TINC,
     +                   YOLD, IROOT, HINIT, HMAX, ITOL, ABSERR,
     +                   RELERR, EXTRAP, NUTRAL, RPAR, IPAR)
C
C
C     DEFINE THE PROBLEM DEPENDENT DATA
C
C
C     INPUT -
C
C     N                  = number of differential equations
C                          (supplied by calling program to
C                          DKLAG6)
C
C     LOUT               = output unit number
C
C     LIN                = input unit number
C
C     RPAR(*), IPAR(*)   = user arrays passed via DKLAG6
C
C     This subroutine must define the following
C     problem dependent data.
C
C     NGUSER             = number of user-defined root functions
C
C     TOLD               = initial value of independent
C                          variable
C
C     TFINAL             = final value of independent
C                          variable
C
C     TINC               = independent variable increment
C                          at which the solution and
C                          derivative are to be
C                          interpolated
C                          Consider replacing subroutine
C                          D6OUTP if a nonconstant TINC
C                          is desired.
C
C     YOLD(I), I=1,...,N = inital solution Y(TOLD)
C
C     IROOT              = rootfinding flag. If IROOT=0
C                          rootfinding will be used to
C                          locate points with derivative
C                          discontinuities. If IROOT=1
C                          such discontinuities will be
C                          ignored.
C
C     HINIT              = initial stepsize for DKLAG6. If
C                          HINIT = 0.0D0, DKLAG6 will determine
C                          the initial stepsize.
C
C     HMAX               = maximum allowable stepsize
C
C     ITOL               = type of error control flag. If you
C                          define ITOL = 1, you need only define
C                          ABSERR(1) and RELERR(1). These absolute
C                          and relative error tolerances will be
C                          used for each component in the system
C                          of differential equations. If you define
C                          ITOL = N, you then need to define
C                          ABSERR(I) and RELERR(I), I=1,...,N.
C                          In this case ABSERR(I) and RELERR(I)
C                          will be used as the tolerances for the
C                          Ith component of the system.
C
C     ABSERR(I)          = local absolute error tolerance
C        I=1,...,ITOL
C
C     RELERR(I)          = local relative error tolerance
C        I=1,...,ITOL
C
C     EXTRAP             = logical method flag. If this flag is
C                          not set .TRUE. by INITAL, the default
C                          value which will be used is .FALSE.,
C                          in which case the solver will assume
C                          this is not a vanishing lag problem.
C                          Refer to the discussion in the prologue
C                          regarding ``Vanishing Lags'' for a
C                          detailed description of this parameter.
C
C     NUTRAL             = logical problem type flag. If this
C                          flag is not set .TRUE. by INITAL, the
C                          default value which will be used is
C                          .FALSE., in which case, the solver
C                          will assume this is not a neutral
C                          problem in which the calculation of
C                          dy/dt involves a delayed derivative.
C     RPAR(*), IPAR(*)   = user arrays passed via DKLAG6
C
C     You may terminate the integration by setting N to a
C     negative value when INITAL is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
      LOGICAL EXTRAP, NUTRAL
C
      DIMENSION YOLD(*), ABSERR(*), RELERR(*), RPAR(*), IPAR(*)
C
C     NUMBER OF USER DEFINED ROOT FUNCTIONS
      NGUSER = 0
C
C     Parameters values:
      RPAR(1) = 0.5
      RPAR(2) = 4.0
C
C     INITIAL INTEGRATION TIME
      TOLD = 0.0D0
C
C     OUTPUT PRINT INCREMENT
      TINC = 0.05D0
C
C     FINAL INTEGRATION TIME
      TFINAL = 500.0D0 - RPAR(2)
C
C     INITIAL SOLUTION YOLD(*)
      DO 20 I = 1 , N
         CALL YINIT (N, TOLD, YOLD(I), I, RPAR, IPAR)
   20 CONTINUE
C
C     DISCONTINUITY ROOTFINDING FLAG
      IROOT = 1
C
C     INITIAL STEPSIZE
      HINIT = 0.0D0
C
C     MAXIMUM STEPSIZE
      HMAX = 1.0D0
C
C     ERROR TOLERANCE FLAG
      ITOL = 1
C
C     ERROR TOLERANCES
      ABSERR(1) = 1.0D-12
      RELERR(1) = 1.0D-10
C
C     VANISHING DELAY FLAG
      EXTRAP = .FALSE.
C
C     NEUTRAL PROBLEM FLAG
      NUTRAL = .FALSE.
C
      RETURN
      END
C
C
      SUBROUTINE YINIT (N, T, Y, I, RPAR, IPAR)
C
C
C     CALCULATE THE SOLUTION ON THE INITIAL INTERVAL
C
C
C     INPUT -
C
C     N                  = number of equations
C
C     T                  = independent variable (time)
C
C     I                  = component number
C
C     RPAR(*), IPAR(*)   = user arrays passed via DKLAG6
C
C     This subroutine must define Y = Y(T,I), the solution
C     at time T corresponding to the initial delay interval
C     for component I (not the entire Y vector, just the
C     I-th component). YINIT is called any time the code
C     needs a value of the solution corresponding to the
C     initial delay interval.
C
C     You may terminate the integration by setting N to
C     a negative value when YINIT is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      DIMENSION RPAR(*), IPAR(*)
C
      R = RPAR(1)
      H = RPAR(2)
      STARTI = 0.1D0
      STARTS = 1.0D0 - STARTI

      S = STARTS*EXP(-R*(T+H))/(1.0D0-STARTS+STARTS*EXP(-R*(T+H)))
      
      IF (I .EQ. 1) Y = S
      IF (I .EQ. 2) Y = 1.0D0 - S
      IF (I .EQ. 3) Y = 0.0D0
C
      RETURN
      END
C
C
      SUBROUTINE DYINIT (N, T, DY, I, RPAR, IPAR)
C
C
C     CALCULATE THE DERIVATIVE ON THE INITIAL INTERVAL
C
C
C     INPUT -
C
C     N                  = number of equations
C
C     T                  = independent variable (time)
C
C     I                  = component number
C
C     RPAR(*), IPAR(*)   = user arrays passed via DKLAG6
C
C     This subroutine must define DY = DY(T,I)/DT, the derivative 
C     at time T corresponding to the initial delay interval
C     for component I (not the entire DY vector, just the
C     I-th component). DYINIT is called any time the code
C     needs a value of the solution corresponding to the
C     initial delay interval.
C
C     If this is not a neutral problem, that is, if the
C     derivative evaluation not involve a delayed derivative
C     (i.e., DYLAG is not used in subroutine DERIVS), you 
C     can simply replace DYINIT in the call to DKLAG6 by
C     YINIT and avoid supplying this subroutine.
C
C     You may terminate the integration by setting N to a
C     negative value when DYINIT is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      DIMENSION RPAR(*), IPAR(*)
C
C     y'(t) = 0 ...
C
      DY = 0.0D0
C
      RETURN
      END
C
C
      SUBROUTINE DPRINT (N, T, Y, DY, NG, JROOT, INDEXG, ITYPE,
     +                   LOUT, INDEX, RPAR, IPAR)
C
C
C     PROCESS THE SOLUTION AT OUTPUT POINTS
C
C
C     INPUT -
C
C     N                    = number of odes
C
C     T                    = independent variable (time)
C
C     Y(I), I=1,...,N      = solution at time T
C
C     DY(I), I=1,...,N     = derivative at time T
C
C     NG                   = number of event functions
C
C     JROOT(I), I=1,...,NG = integrator event function flag
C                            array
C                            If JROOT(I) = 1, the I-th residual
C                            function has a zero at T.
C
C     INDEXG(I), I=1,...,N = component number indicator for
C                            the root functions
C                            INDEXG(I) indicates which of the
C                            system delays is responsible for
C                            the zero at T of the residual
C                            function.
C
C     LOUT                 = output unit number
C
C     INDEX                = last value of index from the
C                            integrator
C
C     RPAR(*), IPAR(*)     = user arrays passed via DKLAG6
C
C     This subroutine is called by DKLAG6 to allow the
C     user to print and otherwise process the solution
C     at the initial time, at multiples of TINC, and at
C     roots of the event functions. If ITYPE = 0, this
C     call to DPRINT corresponds to a root of an event
C     function. Otherwise, it corresponds to a multiple
C     of the print increment TINC or the initial time.
C     Do not change any of the input parameter values
C     when DPRINT is called.
C
C     You may terminate the integration by setting N to
C     a negative value when DPRINT is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      DIMENSION Y(*), DY(*), JROOT(*), INDEXG(*), RPAR(*), IPAR(*)
C
      R = RPAR(1)
      H = RPAR(2)
C
C     WRITE THE SOLUTION
C
C     The following test is a bit of a hack.  If T=0, we first
C     print the solution for the interval [-h,0].
      IF (T .EQ. 0) THEN
          M = 20
          TINC = H/M;
          TOUT = 0.0D0;
          DO 700 K = 1, M
              CALL YINIT(3,TOUT-H,S,1,RPAR,IPAR)
              WRITE(LOUT,99999) TOUT, S, 1.0D0-S, 0.0D0
              TOUT = TOUT + TINC
700       CONTINUE 
      ENDIF
C
C     Note that for the output, T is incremented by h.
C
      WRITE (LOUT,99999) T+H, Y(1), Y(2), Y(3)
C
C     DETERMINE IF A ROOT OF AN EVENT FUNCTION WAS LOCATED
C
C      IF (ITYPE .EQ. 0) THEN
C         DO 20 I = 1 , NG
C            IROOT = JROOT(I)
C            IF (IROOT.EQ.1)
C     *      WRITE (LOUT,99998) I, INDEXG(I)
C   20    CONTINUE
C      ENDIF
C
99999 FORMAT (' ', D17.7, D17.7, D17.7, D17.7)
99998 FORMAT ('  COMPONENT ', I3, ' OF DKGFUN HAS A ZERO.', /, 
     *        '  THIS CORRESPONDS TO ODE NO. ', I3)
C
      RETURN
      END
C
C
      SUBROUTINE GUSER (N, T, Y, DY, NGUSER, G, RPAR, IPAR)
C
C
C     INPUT -
C
C     N                   = number of odes
C
C     T                   = independent variable (time)
C
C     Y(I), I=1,...,N     = solution at time T
C
C     DY(I), I=1,...,N    = approximation to derivative
C                           at time T (not exact derivative
C                           from DERIVS)
C
C     NGUSER              = number of event functions
C
C     RPAR(*), IPAR(*)    = user arrays passed via DKLAG6
C
C     This subroutine must define the event function
C     residuals G(I), I=1,...,NGUSER.
C
C     Use a dummy subroutine which does nothing if no
C     rootfinding is performed.
C
C     You may terminate the integration by setting N to
C     a negative value when GUSER is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      DIMENSION Y(*), DY(*), G(*)
C
C
      RETURN
      END
C
C
      SUBROUTINE CHANGE (INDEX, JROOT, INDEXG, NG, N, YROOT, DYROOT, 
     +                    HINIT, TROOT, TINC, ITOL, ABSERR, 
     +                    RELERR, LOUT, RPAR, IPAR)
C
C
C     MAKE PROBLEM CHANGES AT ROOTS OF THE EVENT FUNCTIONS
C
C
C     INPUT -
C
C     INDEX                = last value of index from the
C                            integrator
C
C     JROOT(I), I=1,...,NG = integrator event function flag
C                            array. If JROOT(I) = 1, the
C                            I-th residual function has a
C                            zero at T.
C
C     INDEXG(I), I=1,...,N = component number indicator for
C                            the root functions INDEXG(I)
C                            indicates which of the system
C                            delays is responsible for the
C                            zero at T of the residual
C                            function.
C
C     NG                   = number of event functions
C
C     N                    = number of odes
C
C     YROOT(I), I=1,...,N  = solution at time TROOT
C
C     DYROOT(I), I=1,...,N = derivative at time TROOT
C
C     HINIT                = initial stepsize
C
C     TROOT                = value of the independent variable
C                            at which the root occurred
C
C     TINC                 = independent variable increment
C                            at which the solution and
C                            derivative are to be
C                            interpolated
C
C     ITOL                 = error control flag. If ITOL = 1
C                            ABSERR and RELERR are vectors of
C                            length 1. If ITOL = N they are
C                            vectors of length N.
C
C     ABSERR(I)            = local absolute error tolerance(s)
C        I = 1,...,ITOL
C
C     RELERR(I)            = local relative error tolerance(s)
C        I = 1,...,ITOL
C
C     LOUT                 = output unit number
C
C     RPAR(*), IPAR(*)     = user arrays passed via DKLAG6
C
C     When CHANGE is called, any of the parameters HINIT, TINC, 
C     ITOL, ABSERR, and RELERR may be changed.
C
C     Use a dummy subroutine which does nothing if no changes
C     are required at any root.
C
C     You may terminate the integration by setting N to a
C     negative value when CHANGE is called.
C
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      DIMENSION JROOT(*), INDEXG(*), YROOT(*), DYROOT(*)
      DIMENSION RPAR(*), IPAR(*), ABSERR(*), RELERR(*)
C
      RETURN
      END
