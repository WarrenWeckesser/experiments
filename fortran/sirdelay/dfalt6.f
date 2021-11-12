C
C THIS FILE CONTAINS THE DEFAULT VERSIONS OF THE DKLAG6
C USER REPLACEABLE SUBROUTINES.
C
C THIS IS A DOUBLE PRECISION VERSION. A SINGLE PRECISION VERSION
C MAY BE CREATED BY MAKING THE FOLLOWING GLOBAL STRING REPLACEMENTS:
C 
C   CHANGE:  DOUBLE PRECISION  TO:  REAL
C   CHANGE:  DBLE              TO:
C   CHANGE:  D1MACH            TO:  R1MACH
C   CHANGE:  D0                TO:  E0
C   CHANGE:  D1                TO:  E1
C   CHANGE:  D2                TO:  E2
C   CHANGE:  D3                TO:  E3
C   CHANGE:  D4                TO:  E4
C   CHANGE:  D5                TO:  E5
C   CHANGE:  D-                TO:  E-
C   CHANGE:  DAXPY             TO:  SAXPY
C   CHANGE:  DCOPY             TO:  SCOPY
C   CHANGE:  DKLAG6            TO:  SKLAG6
C   CHANGE:  D6                TO:  S6
C   CHANGE:  KPRED             TO:  KRPEE
C
C   DEPENDING ON YOUR f77 COMPILER, YOU MAY WISH TO DELETE
C   THE "IMPLICIT NONE" STATEMENTS FROM EACH SUBROUTINE
C   IN THIS FILE.
C
C*DECK D6LEVL
      SUBROUTINE D6LEVL(LEVMAX,NUTRAL)
C
C***BEGIN PROLOGUE  D6LEVL
C***SUBSIDIARY
C***PURPOSE  The purpose of D6LEVL is to define the maximum
C            level to which potential derivative discontinuities
C            are to be tracked. This is a user replaceable
C            subroutine; but considerable thought should be
C            given before doing so.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***DESCRIPTION
C
C           This subroutine must define the following
C           parameter:
C
C           LEVMAX = maximum level to which potential
C                    derivative discontinuities are
C                    to be tracked if discontinuity
C                    rootfinding is being used
C                    (IROOT = 0 in subroutine INITAL).
C
C           If LEVMAX = 0, full tracking will be done.
C           If LEVMAX > 0, tracking will be done only
C                          to derivative order LEVMAX.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH
C***REVISION HISTORY  (YYMMDD)
C   960501  DATE WRITTEN
C***END PROLOGUE  D6LEVL
C
      IMPLICIT NONE
C
      INTEGER LEVMAX
      LOGICAL NUTRAL
C
C***FIRST EXECUTABLE STATEMENT  D6LEVL
C
      LEVMAX = 7
      IF (NUTRAL) LEVMAX = 0
C
      RETURN
      END
C*DECK D6EXTP
      SUBROUTINE D6EXTP(IROOTL)
C
C***BEGIN PROLOGUE  D6EXTP
C***SUBSIDIARY
C***PURPOSE  The purpose of D6EXTP is to indicate whether
C            extrapolatory rootfinding is to be used as a
C            secondary stepsize control prior to each
C            integration step. If it is, and the extrapolated
C            solution polynomial predicts a zero of one of
C            the event functions over the next step, the
C            stepsize will be reduced to hit the predicted
C            root. This is a user replaceable subroutine.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***DESCRIPTION
C
C           This subroutine must define the following
C           parameter:
C
C           IROOTL = 0 IF EXTRAPOLATORY ROOTFINDING
C                      IS TO BE PERFORMED.
C
C           IROOTL = 1 IF EXTRAPOLATORY ROOTFINDING
C                      IS NOT TO BE PERFORMED.
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH
C***REVISION HISTORY  (YYMMDD)
C   960501  DATE WRITTEN
C***END PROLOGUE  D6EXTP
C
      IMPLICIT NONE
C
      INTEGER IROOTL
C
C***FIRST EXECUTABLE STATEMENT  D6EXTP
C
      RETURN
      END
C*DECK D6DISC
      SUBROUTINE D6DISC(IWANT,IDCOMP,IDISC,TDISC,JCOMP,
     +                  RPAR,IPAR)
C
C***BEGIN PROLOGUE  D6DISC
C***SUBSIDIARY
C***PURPOSE  The purpose of D6DISC is to provide the
C            capability of providing discontinuity
C            information for the initial solution
C            for the DKLAG6 delay solver. This is a
C            user replaceable subroutine.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  NONE
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6DISC
C
C     USAGE NOTES:
C
C     IF IROOT = 0 IN SUBROUTINE INITIAL, DKLAG6 USES ROOTFINDING
C     TO LOCATE POINTS AT WHICH DERIVATIVE DISCONTINUITIES EXIST
C     AND INCLUDE THEM AS INTEGRATION GRID POINTS. IT DOES THIS
C     BY STARTING WITH A SYSTEM OF ROOT FUNCTIONS WITH THE FORMS
C     G_I = DELAY_I - TINIT, WHERE TINIT IS THE INITIAL TIME. EACH
C     TIME A ZERO T_I OF G_I IS LOCATED, A ROOT FUNCTION
C     DELAY_I - T_I IS ADDED TO THE SYSTEM OF ROOT FUNCTIONS. THIS
C     IS DONE IN ORDER TO TRACK THE TREE OF DISCONTINUITY POINTS
C     WHICH PROPAGATES FROM DISCONTINUITIES AT THE INITIAL TIME.
C
C     THIS SUBROUTINE, WHICH IS USER REPLACEABLE, ALLOWS THE USER
C     TO DEFINE ADDITIONAL DISCONTINUITY INFORMATION. FOR EACH
C     COMPONENT IN THE SYSTEM, TIMES PRIOR TO THE INITIAL TIME
C     AT WHICH THE INITIAL FUNCTION FOR THAT COMPONENT MAY BE
C     SPECIFIED. FOR EACH SUCH TIME T_D, DKLAG6 STARTS WITH AN
C     ADDITIONAL ROOT FUNCTION DELAY - T_D AND PROCEEDS AS BEFORE.
C
C     THIS SUBROUTINE IS NOT CALLED UNLESS IROOT = 0 IN
C     IN SUBROUTINE INITAL. IF IROOT = 0, THEN FOR EACH
C     VALUE OF IDCOMP = 1 TO N, D6DISC WILL BE CALLED WITH
C     IWANT = 1. WHEN SUCH A CALL IS MADE, DEFINE IDISC TO
C     BE THE NUMBER OF POINTS STRICTLY PRIOR TO THE INITIAL
C     TIME AT WHICH THE INITIAL FUNCTION IS DISCONTINUOUS.
C
C     FOR ANY VALUE OF IDCOMP FOR WHICH THIS SUBROUTINE
C     RETURNS A VALUE LARGER THAN 0 FOR IDISC, THIS SUBROUTINE
C     WILL THEN BE CALLED WITH IWANT = 2 FOR EACH VALUE OF
C     JCOMP = 1 TO IDISC. WHEN THE CALL IS MADE, DEFINE
C     TDISC TO BE THE JCOMPth VALUE OF TIME AT WHICH THE
C     IDISCth COMPONENT OF THE INITIAL FUNCTION IS
C     DISCONTINUOUS.
C
      IMPLICIT NONE
C
      INTEGER IWANT,IDCOMP,IDISC,JCOMP,IPAR
      DOUBLE PRECISION RPAR,TDISC
      DIMENSION IPAR(*),RPAR(*)
C
C***FIRST EXECUTABLE STATEMENT  D6DISC
C
      IDISC = 0
C
      RETURN
      END
C*DECK D6ESTF
      SUBROUTINE D6ESTF(EXTRAP,ICVAL)
C
C
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1B1, I1B1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***DESCRIPTION
C***AUTHOR  THOMPSON, S., RADFORD UNIVERSITY, RADFORD, VIRGINIA.
C***BEGIN PROLOGUE  D6ESTF
C***SUBSIDIARY
C***PURPOSE  The purpose of D6ESTF is to set the flag which
C            which determines the type of per step error
C            estimate to be used by the DKLAG6 differential
C            equation solver. This is a user replaceable
C            subroutine.
C
C     INPUT:
C
C     EXTRAP - THE LOGICAL PARAMETER DEFINED WHEN SUBROUTINE
C     INITAL WAS CALLED BY DKLAG6
C
C     OUTPUT:
C
C     THIS SUBROUTINE MUST DEFINE THE VALUE OF THE ICVAL FLAG.
C     THE FOLLOWING VALUES ARE ALLOWED. IN THE FOLLOWING,
C     Y(6,8)(C) DENOTES THE SIXTH ORDER, EIGHT STAGE METHOD
C     USED AS THE BASIC INTEGRATION METHOD; AND Y(5,10)(C)
C     DENOTES THE FIFTH ORDER, TEN STAGE SUBSIDIARY METHOD.
C     C = (T-T_n)/H WHERE T_n IS THE LEFT ENDPOINT OF THE
C     INTERVAL AND H IS THE STEPSIZE.
C
C     ICVAL = 1 - THE DIFFERENCE OF THE SIXTH AND FIFTH ORDER
C                 SOLUTION POLYNOMIALS WILL BE USED TO ESTIMATE
C                 THE ERROR. THE ESTIMATE IS FORMED IN THE MANNER
C                 USED IN MOST RUNGE-KUTTA CODES. THE ESTIMATE IS
C                          (Y(6,8) - Y(5,10))(1).
C                 Y(6,8) and Y(5,10) ARE NOT FORMED EXPLICITLY.
C
C     ICVAL = 2 - SAME AS ICVAL = 1 EXCEPT THE ESTIMATE IS FORMED
C                 BY SUBTRACTING DIRECTLY THE FIFTH AND SIXTH
C                 ORDER APPROXIMATIONS. THE ESTIMATE IS
C                          (Y(6,8) - Y(5,10))(1).
C                 Y(6,8)(1) and Y(5,10)(1) ARE FORMED EXPLICITLY
C                 AND SUBTRACTED.
C
C     ICVAL = 3 - AT EACH STEP AN ERROR ESTIMATE WILL BE USED FOR
C                 WHICH THE STEPSIZE H IS ACTUALLY VALID FOR AN
C                 INTERVAL OF LENGTH 2H. THE ESTIMATE IS
C                          (Y(6,8) - Y(5,10))(2).
C                 Y(6,8)(2) and Y(5,10)(2) ARE FORMED EXPLICITLY
C                 AND SUBTRACTED.
C
C     ICVAL = 4 - THE ERROR ESTIMATE WILL CORRESPOND TO THE VALUE
C                 IN [T,T+H] FOR WHICH THE DIFFERENCE OF THE
C                 SOLUTION POLYNOMIALS HAS MAXIMUM MAGNITUDE.
C                 FOR THE METHODS USED, THIS GIVES THE SAME
C                 ESTIMATE AS ICVAL = 2.
C
C     ICVAL = 5 - THE ERROR ESTIMATE WILL BE THE L1 NORM OF THE
C                 DIFFERENCE OF THE SOLUTION POLYNOMIALS ON
C                 THE INTERVAL [T,T+H]. FOR THE METHODS USED,
C                 THIS GIVES THE SAME ESTIMATE AS ICVAL = 2.
C
C     ICVAL = 6 - SIMILAR TO ICVAL = 5 BUT THE NORM IS TAKEN OVER
C                 THE INTERVAL [T,T+2H]. FOR THE METHODS USED,
C                 THIS GIVES THE SAME ESTIMATE AS ICVAL = 3.
C
C     ICVAL = 7 - THE DIFFERENCE OF THE DERIVATIVES OF THE SOLUTION
C                 POLYNOMIAL AT T+H WILL BE USED AS THE ERROR
C                 ESTIMATE. THE ESTIMATE IS
C                          (Y(6,8) - Y(5,10))'(1).
C                 Y(6,8)'(1) and Y(5,10)'(1) ARE FORMED EXPLICITLY
C                 AND SUBTRACTED.
C
C     ICVAL = 8 - THE MAXIMUM DIFFERENCE OF THE DERIVATIVES OF THE
C                 SOLUTION POLYNOMIALS ON THE INTERVAL [T,T+H]
C                 WILL BE USED AS THE ERROR ESTIMATE. FOR THE
C                 METHODS USED, THIS GIVES THE SAME ESTIMATE AS
C                 ICVAL = 7.
C
C     ICVAL = 9 - THE DIFFERENCE OF THE DERIVATIVES OF THE SOLUTION
C                 POLYNOMIALS AT T+2H WILL BE USED AS THE ERROR
C                 ESTIMATE. THE ESTIMATE IS
C                          (Y(6,8) - Y(5,10))'(2).
C                 Y(6,8)'(2) and Y(5,10)'(2) ARE FORMED EXPLICITLY
C                 AND SUBTRACTED.
C
C     ICVAL =10 - THE INTEGRAL OF THE THE NORM OF THE DIFFERENCE
C                 OF THE DERIVATIVES OF THE SOLUTION POLYNOMIALS ON
C                 THE INTERVAL [T,T+H] WILL BE USED AS THE ERROR
C                 ESTIMATE. FOR THE METHODS USED, THIS GIVES THE
C                 SAME ESTIMATE AS ICVAL = 2.
C
C     NOTE:
C
C        IF ICVAL IS GIVEN A VALUE OTHER THAN ONE OF THE ABOVE
C        VALUES, DKLAG6 WILL RESET IT TO 1.
C
C
C***END PROLOGUE  D6ESTF
C
C
      IMPLICIT NONE
C
      INTEGER ICVAL
      LOGICAL EXTRAP
C
C***FIRST EXECUTABLE STATEMENT  D6ESTF
C
      ICVAL = 1
C
      RETURN
      END
C*DECK D6BETA
      SUBROUTINE D6BETA(N,T,H,BVAL,IOK)
C
C***BEGIN PROLOGUE  D6BETA
C***SUBSIDIARY
C***PURPOSE  The purpose of D6BETA is to check for illegal delays
C            for the DKLAG6 differential equation solver. This is
C            a user replaceable subroutine.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6BETA
C
C     CHECK FOR ILLEGAL DELAYS.
C
      IMPLICIT NONE
C
      INTEGER I,IOK,N
      DOUBLE PRECISION BVAL,T,H
C
      DIMENSION BVAL(*)
C
C***FIRST EXECUTABLE STATEMENT  D6BETA
C
      IOK = 0
C
      DO 10 I = 1,N
          IF ((BVAL(I).GT.T) .AND. (H.GT.0.0D0) .OR.
     +        (BVAL(I).LT.T) .AND. (H.LT.0.0D0)) IOK = 1
   10 CONTINUE
C
C
      IF (IOK.EQ.1) THEN
         CALL D6MESG(42,'D6BETA')
         IOK = 0
      ENDIF
C
      IF (IOK.EQ.1) CALL D6MESG(24,'D6BETA')
C
      RETURN
      END
C*DECK D6AFTR
      SUBROUTINE D6AFTR(N,TOLD,YOLD,DYOLD,TNEW,YNEW,DYNEW,
     +                 K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,
     +                 IORDER,RPAR,IPAR)
C
C***BEGIN PROLOGUE  D6AFTR
C***SUBSIDIARY
C***PURPOSE  The purpose of D6AFTR is to provide the
C            capability of allowing the user to
C            update anything in his program following
C            successful integration steps. When
C            D6AFTR is called following a successful
C            integration step, the arguments in the
C            parameter list have the same meaning as
C            in the documentation prologue for DKLAG6.
C            None of these arguments may be altered
C            when the call is made: DKLAG6 assumes
C            that no arguments will be changed;
C            unpredictable results may occur if the
C            value of any argument is changed.
C            This is a user replaceable subroutine.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  NONE
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6AFTR
C
      IMPLICIT NONE
C
      DOUBLE PRECISION TOLD, YOLD, DYOLD, TNEW, YNEW, DYNEW,
     +                 K0, K1, K2, K3, K4, K5, K6, K7, K8,
     +                 K9, RPAR
      INTEGER N, IORDER, IPAR
C
      DIMENSION YOLD(*), DYOLD(*), YNEW(*), DYNEW(*), RPAR(*),
     +          IPAR(*), K0(*), K1(*), K2(*), K3(*), K4(*),
     +          K5(*), K6(*), K7(*), K8(*), K9(*)
C
C***FIRST EXECUTABLE STATEMENT  D6AFTR
C
      RETURN
      END
C*DECK D6RERS
      SUBROUTINE D6RERS(RER)
C
C***BEGIN PROLOGUE  D6RERS
C***SUBSIDIARY
C***PURPOSE  The purpose of D6RERS is to define the floor
C            value RER for the relative error tolerances
C            for the DKLAG6 differential equation solver.
C            If the user specifies a relative error tolerance
C            which is less than this value (and nonzero),
C            it will be replaced by this floor value.
C            RER is defined in a manner consistent with
C            other ODE solvers in the SLATEC library.
C            This is a user replaceable subroutine; but
C            considerable thought should be given before
C            doing so.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***DESCRIPTION
C
C           This subroutine must define the following
C           parameter:
C
C           RER = the floor value below which an ode solver
C                 cannot reasonably be expected to deliver
C                 relative accuracy. The following value is
C                 returned by this subroutine:
C                         RER = 2.0D0*U + REMIN
C                 where U is the machine largest relative
C                 spacing and REMIN is 1.0D-12.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6RERS
C
      IMPLICIT NONE
C
      DOUBLE PRECISION D1MACH,REMIN,RER,U
C
C***FIRST EXECUTABLE STATEMENT  D6RERS
C
      U = D1MACH(4)
      REMIN = 1.0D-12
      RER = 2.0D0*U + REMIN
C
C
      RETURN
      END
C*DECK D6STAT
      SUBROUTINE D6STAT(N,ISTAT,H,RWORK,IWORK,IDUM,RDUM)
C
C***BEGIN PROLOGUE  D6STAT
C***SUBSIDIARY
C***PURPOSE  The purpose of D6STAT is to allow the user to
C            collect integration statistics. This version
C            of D6STAT is a dummy subroutine which does nothing.
C            This is a user replaceable subroutine which may be
C            collect statistics according to the description
C            given below. 
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***DESCRIPTION
C
C           When this subroutine is called, N is the number of
C           differential equations, H is the stepsize, and
C           RPAR and IPAR are the user provided arrays in
C           the call to DKLAG6.
C
C           ISTAT has one of the values 1,2,3,4,5,6 as follows:
C
C           ISTAT = 1 means that this call is made at the
C                     beginning of the integration to allow
C                     the user to perform any necessary
C                     initializations.
C
C           ISTAT = 2 means this call is made to allow the
C                     user to save the initial stepsize.
C
C           ISTAT = 3 means this call was made immediately
C                     following a successful integration step.
C
C           ISTAT = 4 means this call was made immediately
C                     following an unsuccessful integration step.
C
C           ISTAT = 5 means this call was made immediately
C                     after a successful correction.
C
C           ISTAT = 6 means this call was made immediately
C                     after a corrector iteration which did
C                     not converge for the given stepsize
C                     within the maximum number of allowable
C                     iterations.
C
C           Caution:
C           Do not change N or H when this subroutine is called.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6STAT
C
      IMPLICIT NONE
C
      INTEGER ISTAT, IWORK, N, IDUM
      DOUBLE PRECISION H, RWORK, RDUM
C
      DIMENSION RWORK(*), IWORK(*)
C
C***FIRST EXECUTABLE STATEMENT  D6STAT
C
      RETURN
      END
C*DECK D6OUTP
      DOUBLE PRECISION FUNCTION D6OUTP(T,TINC)
C
C***BEGIN PROLOGUE  D6OUTP
C***SUBSIDIARY
C***PURPOSE  The purpose of D6OUTP is to define the output
C            increment for the DKLAG6 differential equation
C            solver. This function subroutine simply returns
C            the value of TINC for the output point increment.
C            A time dependent output increment can be used by
C            replacing this function with a user provided one.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***DESCRIPTION
C
C     INPUT -
C
C     T       = INDEPENDENT VARIABLE (TIME)
C     TINC    = OUTPUT INCREMENT SUPPLIED BY
C               THE USER IN SUBROUTINE INITAL
C
C     THIS SUBROUTINE MUST DEFINE THE OUTPUT
C     PRINT INCREMENT, D6OUTP.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6OUTP
C
      IMPLICIT NONE
C
C     CALCULATE THE OUTPUT PRINT INCREMENT FOR THIS
C     VALUE OF T.
C
      DOUBLE PRECISION T, TINC
C
C***FIRST EXECUTABLE STATEMENT  D6OUTP
C
      D6OUTP = TINC
C
      RETURN
      END
C*DECK D6RTOL
      SUBROUTINE D6RTOL(NSIG,NGEMAX,U10)
C
C***BEGIN PROLOGUE  D6RTOL
C***SUBSIDIARY
C***PURPOSE  The purpose of D6RTOL is to define convergence
C            parameters for the DKLAG6 rootfinder. Loosely
C            speaking, the tolerance used by the rootfinder
C            is 10**(-NSIG) if NSIG is positive and 0 otherwise,
C            that is, NSIG is the number of significant digits
C            of accuracy in computed roots. NGEMAX is the maximum
C            number of iterations the rootfinder will perform
C            before terminating with an error condition if
C            convergence has not been achieved. The actual
C            tolerance used for time T and stepsize H is
C                 MAX( GRTOL*MAX(ABS(T),ABS(T+H)) , U10)
C            where
C                 GRTOL = MAX( 10**(-NSIG) , U10), and
C            U10 = 10.0D0*D1MACH(4). (D1MACH(4) is machine
C            unit roundoff.) With the following definitions
C            for these parameters, all roots are approximated to
C            within machine precision. Since this can be costly,
C            the user may replace this subroutine and relax the
C            tolerance if less accuracy is desired. For example,
C            it is sometimes reasonable for NSIG to correspond
C            to the same number of significant digits specified
C            by the integration error tolerance. However,
C            considerable thought should be given before doing so.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6RTOL
C
      IMPLICIT NONE
C
      INTEGER NSIG,NGEMAX
      DOUBLE PRECISION U10,DROUND,D1MACH
C
C     SET CONVERGENCE PARAMETERS FOR THE ROOTFINDER.
C
C***FIRST EXECUTABLE STATEMENT  D6RTOL
C
C     NSIG = 0
      NSIG = 14
      NGEMAX = 500
      DROUND = D1MACH(4)
      U10 = 10.0D0*DROUND
C
      RETURN
      END
C*DECK D6FINS
      SUBROUTINE D6FINS(NEXTRA,TFINS)
C
C***BEGIN PROLOGUE  D6FINS
C***SUBSIDIARY
C***PURPOSE  The purpose of D6FINS is to allow the
C            the inclusion of up five outpoint times
C            within the integration range which DKLG6C
C            must step to exactly.
C***LIBRARY   SLATEC
C***CATEGORY  I1, I3, I1A, I1A1, I1A1A
C***TYPE      DOUBLE PRECISION
C***KEYWORDS  ODE, ORDINARY DIFFERENTIAL EQUATIONS,
C             INITIAL VALUE PROBLEMS, REAL,
C             DELAY DIFFERENTIAL EQUATIONS
C***AUTHOR  THOMPSON, S.,
C             DEPT. OF MATHEMATICS AND STATISTICS
C             RADFORD UNIVERSITY
C             RADFORD, VIRGINIA 24142
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  NONE
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6AFTR
C
      IMPLICIT NONE
C
      DOUBLE PRECISION TFINS
      INTEGER NEXTRA
C
      DIMENSION TFINS(*)
C
C***FIRST EXECUTABLE STATEMENT  D6FINS
C
C      DEFINE NEXTRA TO BE THE NUMBER OF VALUES TO BE ADDED.
C      NEXTRA MUST BE BETWEEN 0 AND 5. (IF IT IS NECESSARY
C      TO CHANGE THIS LIMIT, DO SO IN SUBROUTINE D6FIND.)
C
       NEXTRA = 0
C
C      DEFINE THE TIMES TO BE ADDED TO THE DISCONTINUITY TREE.
C
      RETURN
      END
