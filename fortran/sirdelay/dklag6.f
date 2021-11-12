C
C----------------------------------------------------------------
C
C THIS IS THE BEGINNING OF THE DOCUMENTATION FOR DKLAG6.
C
C THIS IS THE FEBRUARY 2003 VERSION OF THE DDE SOLVER DKLAG6.
C IT ADDS THE ABILITY TO PROCESS THE DISCONTINUITY TREE FOR
C PROBLEMS WITH CONSTANT DELAYS DIRECTLY (VIA SUBROUTINE
C DKLG6C) AND AVOID DISCONTINUITY ROOTFINDING. IT ALSO DOES
C SECOND PASS INTERPOLATORY ROOTFINDING TO AVOID MISSING
C ROOTS OF EVENT FUNCTIONS WITHOUT USER INTERVENTION (DONE
C NORMALLY IN SUBROUTINE CHANGE).
C
C----------------------------------------------------------------
C
C                     GENERAL INFORMATION
C
C SECTION   I.  -  INSTALLATION INSTRUCTIONS 
C
C SECTION  II.  -  VERSION INFORMATION
C
C SECTION III.  -  USER REPLACEABLE SUBROUTINES
C
C SECTION  IV.  -  OVERVIEW OF SOLUTION PROCEDURE
C
C SECTION   V.  -  SUBROUTINE NAMES AND PURPOSES
C
C SECTION  VI.  -  DOCUMENTATION PROLOGUE
C
C----------------------------------------------------------------
C
C SECTION I.
C
C                           README6.2nd
C
C                     INSTALLATION INSTRUCTIONS 
C
C Main Features of the DKLAG6 Delay Differential Equation Solver:
C
C  1. A pair of (5,6) continuously imbedded Runge-Kutta-Sarafyan
C     methods is used. This facilitates the use of rootfinding
C     and interpolation for dense output.
C
C  2. State dependent delays are allowed.
C
C  3. Points with derivative discontinuities which are propagated
C     by discontinuities at the initial time may be located
C     automatically and included as integration mesh points.
C     Special care is taken near such points to avoid decreasing
C     the efficiency of DKLAG6. It is also possible via a user
C     replaceable subroutine to optionally track the points of
C     discontinuity propagated by points other than the initial
C     time at which the initial function or its derivative have
C     discontinuities.
C
C  4. For vanishing delay problems, a predictor-corrector like
C     iteration is used to avoid long sequences of small step
C     sizes near points at which a delay vanishes.
C
C  5. DKLAG6 may be used to solve neutral problems in which the
C     derivative evaluation includes delayed derivative terms.
C
C  6. It is possible to use any of several types of error control
C     via a user replaceable subroutine. It is possible to replace
C     the usual default error control with ones forcing the
C     underlying polynomial interpolants to be accurate for larger
C     intervals or ones which enforce more stringent error control
C     on the derivatives of the polynomial interpolants.
C
C  7. Subroutines are included for user interpolation of the solution
C     history queue for problems requiring interpolation at times
C     other than the system delay times.
C
C  8. It is possible via a user replaceable subroutine to collect
C     various integration statistics.
C
C  9. To invoke DKLAG6:
C
C      CALL DKLAG6 (INITAL, DERIVS, GUSER, CHANGE, DPRINT, BETA, 
C     +            YINIT, DYINIT, N, W, LW, IW, LIW, QUEUE,
C     +            LQUEUE, IFLAG, LIN, LOUT, RPAR, IPAR)
C
C     The parameters are described in the documentation prologue
C     for subroutine DKLAG6.
C
C 10. The DRAKE PC interface is available for creating interactively
C     the necessary user driver program and problem subroutines
C     required by DKLAG6. Note: If graphics is requested in the
C     DRAKE menu, DRAKE requires the Volksgrapher (VG) Fortran
C     callable graphics package. VG is available for several
C     computing platforms via anonymous ftp from the National
C     Institute of Standards. The contact at NIST is William
C     Anderson (email: anderson@eeel.nist.gov). As of 07/20/95,
C     the ftp location containing the necessary VG files is:
C     ftp.eeel.nist.gov. DKLAG6 does not itself require the
C     use of VG.
C
C 11. DKLAG6 is intended to work properly with any Fortran77
C     compiler. The only non-ansi standard construct used in
C     DKLAG6 is the presence of the statement: IMPLICIT NONE
C     in each subroutine. (These statements may be commented out
C     if desired.) The Forchek (version 2.4) FORTRAN checker
C     was used to eliminate non-portable code from DKLAG6.
C     (Note: This was not done for some of the test programs
C     mentioned below. If you experience difficulty with any
C     of the test programs, please contact the author of DKLAG6.
C
C Note:
C The following files are for the double precision version of D6LAG.
C Refer to the header comments at the beginning of the file dklag6.f
C for instructions how to create a single precision version.
C See also the SINGLE subdirectory.
C
C Note:
C The following instructions assume you are on a UNIX machine.
C If this is not the case, please contact the author of DKLAG6
C to obtain the necessary files in whatever other manner is
C appropriate.
C
C To obtain the items below, do the following:
C
C      1. Pick up the DKLAG6.tar.Z compressed tar file.
C         (Caution: If you are using ftp, note that
C         this is a binary file.)
C      
C      2. Type: uncompress DKLAG6.tar.Z
C
C      3. Type: tar xf DKLAG6.tar
C
C You will now have a directory DKLAG6 containing the files
C and subdirectories described below.
C
C The DKLAG6 directory contains the following subdirectories:
C
C  DOUBLE     - source code for the double precision solver
C
C  DEMOS      - double precision demo programs and makefiles
C
C Assuming you plan to use the double precision version, only
C the DOUBLE subdirectory is absolutely necessary; DEMOS may
C be deleted if you wish.
C
C Quick Start Instructions:
C
C  1. The source code for DKLAG6 consists of the three files
C     dklag6.f, dfalt6.f, and dkutil.f in subdirectory DOUBLE.
C
C  2. Make sure the machine constants in the function
C     subroutines I1MACH, R1MACH, and D1MACH of the file
C     dkutil.f are set for your machine. They are currently
C     set for a SUN/UNIX.  Others may be selected by commenting
C     out the SUN/UNIX statements and removing the appropriate
C     C's appearing in column 1 of these subroutines.
C
C  3. You may want to combine the files dklag6.f, dfalt6.f, and
C     dkutil.f into one file. It will contain all subroutines
C     used by DKLAG6.
C
C  4. For the quickest start or if reading long README files
C     bugs you, there is a program stub6.f (and a makefile
C     makestub6) in DOUBLE which contains a simple demonstration
C     program. Look at it and modify it to solve your problem.
C
C     Otherwise, continue reading ....
C
C  5. Compile and run subroutine demo69.f in subdirectory
C     DEMOS. demo69.f requires the data file demo69.dat.
C     To create the executable file demo69 and execute it,
C
C     type:
C
C     make -f make69
C     demo
C
C     The last message printed to the screen will indicate
C     whether or not all tests passed. If any test doesn't
C     pass, please contact the author of DKLAG6.
C
C  6. Compile and run demo60.f in subdirectory DEMOS. The
C     user subroutines required for your problem may be
C     modeled after those in demo60.f. demo60.f requires
C     the data file demo60.dat. The descriptions of the
C     parameters for each user subroutine are contained
C     in the documentation prologue for subroutine DKLAG6
C     in file dklag6.f. They are also contained in the file
C     prologues. To create the executable file demo and
C     execute it,
C
C     type:
C
C     make -f make60
C     demo
C
C     Messages are printed to the screen indicating whether
C     the demo was successful. If any demo is not successful
C     for any case, please contact the author of DKLAG6.
C
C More Detailed Instructions:
C
C dklag6.f, dfalt6.f, dkutil.f - the DKLAG6 source code
C
C    dklag6.f - core DKLAG6 subroutines
C
C    dfalt6.f - default versions of the optionally user
C               replaceable DKLAG6 subroutines
C
C    dkutil.f - utilities subroutines (e.g., machine
C               constants and error handlers) needed
C               by DKLAG6
C
C    The three files may be combined into one file if you
C    choose to do so.
C
C    Caution:
C    Be sure to activate the appropriate machine constants
C    in D1MACH, R1MACH, and I1MACH before using DKLAG6.
C    They are currently set for a SUN computer.
C
C    The subdirectory DEMOS contains several demonstration
C    programs and corresponding makefiles. Refer to the
C    make60 demonstration makefile to see how to use the
C    above three files. These demonstration programs are
C    described in more detail below and in the demo6*.f
C    file headers.
C
C    All error messages are printed using a version of the
C    SLATEC XERROR package (contained in file dkutil.f) via
C    subroutine D6MESG. If you desire, D6MESG may be replaced
C    by a subroutine which uses a less sophisticated error
C    handler or other local handler you prefer, thereby
C    avoiding the use of the XERROR package.
C
C In subdirectory DEMOS:
C
C demo6*.f, demo6*.dat (* = 0,...,19) - demonstration programs,
C make*, runthesedemos6, klean          data files, and makefiles
C
C    Note:
C    Look at demo60.f first for an example of what the user
C    provided subroutines look like for a simple problem.
C
C To run the demonstration program demo6*.f (* = 0,...,19), 
C
C    type:
C
C    make -f make6*
C    demo
C
C To run all the demos,
C
C    type:
C
C    runthesedemos6
C
C To clean up after running a demo,
C
C    type:
C
C    klean
C
C Note:
C Each of the demonstration programs writes messages to the screen
C (logical unit = 0) indicating the success or failure of the
C integrations performed. Detailed results are written to the
C files demo6*.ans (unit = 6) and summary results are written
C to the file demo6*.sum (unit = 7). Each requires a data file
C demo6*.dat (read with unit = 5).
C
C Other test programs of various types may be obtained by
C contacting the author of DKLAG6:
C
C        Skip Thompson
C        Department of Mathematics and Statistics
C        Radford University
C        Radford, Virginia 24142
C
C        Telephone: 540-831-5478
C        Email: thompson@runet.edu
C
C Please feel free to contact me if you have any questions
C concerning DKLAG6, if you experience any difficulty using
C it, or if you have any suggestions for improving it.
C Although it will bruise my ego, I particularly would like
C to hear about any bugs you might find (actually, anything
C that is nonportable or that even remotely resembling a bug).
C
C----------------------------------------------------------------
C
C SECTION II.
C
C                     VERSION INFORMATION
C
C THIS IS THE JUNE, 1996 VERSION OF DKLAG6. THIS VERSION USES
C THE FAMILY OF FIFTH AND SIXTH ORDER, TEN STAGE METHODS. IT
C ALSO DOES A PREDICTOR CORRECTOR LIKE ITERATION WHEN
C NECESSARY FOR VANISHING DELAY PROBLEMS.
C
C THIS VERSION ALLOWS THE USER TO INPUT DISCONTINUITY
C INFORMATION ABOUT THE INITIAL FUNCTION VIA SUBROUTINE
C D6DISC.
C
C   DEPENDING ON YOUR FORTRAN COMPILER , YOU MAY WISH TO DELETE
C   THE "IMPLICIT NONE" STATEMENTS FROM EACH SUBROUTINE IN THIS
C   FILE.
C
C----------------------------------------------------------------
C
C SECTION III.
C
C                USER REPLACEABLE SUBROUTINES
C
C     NOTE:
C     FOR AN EXAMPLE FILE WITH USER REPLACEMENTS FOR THE
C     FOLLOWING SUBROUTINES, REFER TO FILE duser6.f.
C
C     D6FINS  -  ALLOW THE ADDITION UP TO FIVE ADDITIONAL
C                POINTS WITHIN THE INTEGRATION RANGE TO
C                BE ADDED TO THE DERIVATIVE DISCONTINUITY
C                TREE.
C
C     D6LEVL  -  DEFINE THE MAXIMUM LEVEL (ORDER) TO
C                WHICH POTENTIAL DISCONTINUITIES WILL
C                BE TRACKED IF DISCONTINUITY ROOTFINDING
C                IS USED.
C
C     D6EXTP  -  INDICATE WHETHER EXTRAPOLATORY ROOTFINDING
C                IS TO BE USED TO PREDICT ROOTS OF THE
C                EVENT FUNCTIONS BEFORE EACH INTEGRATION
C                STEP.
C
C     D6OUTP  -  DEFINE THE OUTPUT PRINT INCREMENT.
C
C     D6STAT  -  DO NOTHING SUBROUTINE WHICH IS CALLED
C                FOLLOWING SUCCESSFUL STEPS. MAY BE
C                REPLACED BY A USER SUBROUTINE FOR
C                COLLECTING INTEGRATION STATISTICS.
C
C     D6RERS  -  DEFINE THE FLOOR VALUE FOR THE RELATIVE
C                ERROR TOLERANCE.
C
C     D6RTOL  -  DEFINE THE TOLERANCE TO BE USED IN
C                THEE ROOTFINDING CALCULATIONS.
C
C     D6ESTF  -  DEFINE THE TYPE OF ERROR ESTIMATE TO BE
C                USED BY DKLAG6.
C
C     D6DISC  -  SUPPLY DISCONTINUITY INFORMATION FOR
C                POINTS PRIOR TO THE INITIAL TIME.
C
C     D6AFTR  -  CALLED FOLLOWING EACH SUCCESSFUL
C                INTEGRATION STEP.
C
C     D6BETA  -  DETECT ADVANCED DELAYS.
C
C----------------------------------------------------------------
C
C SECTION IV.
C
C                OVERVIEW OF SOLUTION PROCEDURE
C
C     DKLAG6 allocates storage and calls the interval-oriented
C     driver D6DRV2. D6DRV2 in turn controls the integration
C     (in D6DRV3/D6STP*) in order to obtain the solution at the
C     desired output points. D6DRV3 controls the integration
C     step subroutine, the rootfinder, the error estimator,
C     and the stepsize calculation. D6STP* calculates the
C     Runge-Kutta derivative approximations.
C
C     If rootfinding is being performed then before each
C     integration step (beginning with the second step), the
C     most recent solution polynomial from the previous step
C     is extrapolated to search for a zero in the interval
C     (t_n , t_{n+1}) = (t_n , t_n + h_n). If a zero is found
C     at t_e, the stepsize is reset to h_n = t_e - t_n before
C     proceeding with the step.
C
C     To step from t_n to t_n + h_n:
C
C     If EXTRAP (cf, subroutine INITAL) is false:
C        For i = 1, ... , 9:
C           Attempt to calculate the Runge-Kutta derivative
C           approximation k_i:
C              If the necessary delay time for any k_i
C              calculation exceeds t_n:
C                 If h_n = hmin:
C                    Terminate with an error message.
C                 Else:
C                    Reset h_n = h_n / 2 and start over.
C        Once k_1, ... , k_9 have been calculated:
C           Proceed to the error test.
C
C     If EXTRAP is true:
C        For icor = 1, ... ,maxcor:
C           For i = 1, ... , 9:
C              Attempt to calculate the Runge-Kutta derivative
C              approximation k_i:
C                 If this is the first step and icor = 1:
C                    If the necessary delay time exceeds t_n:
C                       Extrapolate members of the lower order
C                       imbedded family (cf, D6INT9) to approximate
C                       the delayed solution and derivative.
C                          The higher order methods are used
C                          as they become available: different
C                          methods are used for different k_i.
C                 If this is not the first step and icor = 1:
C                    If the necessary delay time for any k_i
C                    calculation exceeds t_n:
C                       Extrapolate the solution polynomial from
C                       the previous step to approximate the
C                       delayed solution and derivative.
C                 If icor > 1, use the most recent polynomial
C                       iterate for this step to approximate
C                       the delayed solution and derivative.
C           Once k_1, ... , k_9 have been calculated:
C              If icor = 1:
C                 If no extrapolations were performed:
C                    Exit the corrector loop and proceed to
C                    the error test.
C                 Else:
C                    Perform a correction.
C              If icor > 1:
C                 Test for convergence of the derivative
C                 approximations k_1, ... , k_9:
C                    If convergence:
C                       Exit the corrector loop and
C                       and proceed to the error test.
C                    If not convergence:
C                       If icor = maxcor:
C                          If h_n = hmin:
C                             Terminate with an error message.
C                       Else:
C                         Set h_n = h_n / 2 and start over.
C
C     Following the step from t_n to t_n + h_{n+1}, perform the
C     error test:
C        If the estimated error (cf, D6ERR1) is too large:
C           Calculate a smaller stepsize h_n.
C           If the h_n is too small:
C              Terminate with an error message.
C           Else:
C              Repeat the step.
C        If the estimated error is small enuf:
C           Calculate the stepsize h_{n+1} to be attempted
C           for the next step.
C
C        If rootfinding is being performed, then interpolate
C        the solution polynomial for this step to search for
C        a zero in (t_n , t_n + h_n):
C           If a zero is found at t_r:
C              Reset h_n = t_r - t_n and repeat the step
C              to force t_r to be an integration grid point.
C              Do not perform the error test calculation
C              since a smaller stepsize is being used.
C
C        Update the solution history queue, return to D6DRV2,
C        and await further instructions.
C
C----------------------------------------------------------------
C
C SECTION V.
C
C                  SUBROUTINE NAMES AND PURPOSES
C
C   DKLAG6 ...
C
C   Purpose: The function of subroutine DKLAG6 is to solve N
C            ordinary differential equations with state dependent
C            delays. DKLAG6 is applicable to the integration of
C            systems of differential equations of the form
C
C            dY(T)/dT = F(T, Y(T), Y(BETA(T,Y)), Y'(BETA(T,Y))
C
C            given the initial function Y(T) = YINIT(T)
C            and the derivative of this function
C            DY(T) = DYINIT(T) for T between BETA(T0,Y(T0))
C            and TINIT where TINIT is the initial integration
C            time. DKLAG6 is applicable to several types of
C            problems, including constant delay problems, strictly
C            time dependent delay problems, state dependent
C            delay problems, and neutral problems (ones in which
C            the derivative at the solution is needed at the
C            delay times).
C
C   D6DRV2 ...
C
C   Purpose: The function of subroutine D6DRV2 is to solve N
C            ordinary differential equations with state dependent
C            delays. D6DRV2 is applicable to the integration of
C            systems of differential equations of the form
C
C            dY(I)/dT = F(T, Y(I), Y(BETA(T,Y)), Y'(BETA(T,Y)))
C
C            It is a second level driver which is intended
C            to be accessed via subroutine DKLAG6.
C
C   D6DRV3 ...
C
C   Purpose: The function of subroutine D6DRV3 is to solve N
C            ordinary differential equations with state dependent
C            delays. D6DRV3 is a third-level driver called by
C            subroutine D6DRV2.
C
C   D6STP1 ...
C
C   Purpose: The purpose of D6STP1 is to perform a single step
C            of the integration for the DKLAG6 differential
C            equation solver (for non-vanishing delay problems).
C
C   D6STP2 ...
C
C   Purpose: The purpose of D6STP2 is to perform a single step
C            of the integration for the DKLAG6 differential
C            equation solver (for vanishing delay problems).
C
C   D6INT1 ...
C
C   Purpose: The purpose of D6INT1 is to interpolate the past
C            solution history queue for the DKLAG6 differential
C            equation solver.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT1.
C
C   D6INT2 ...
C
C   Purpose: The purpose of D6INT2 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT2.
C
C   D6INT3 ...
C
C   Purpose: The purpose of D6INT3 is to interpolate the past
C            solution history queue for the DKLAG6 differential
C            equation solver.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT3.
C
C   D6INT4 ...
C
C   Purpose: The purpose of D6INT4 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver. This is a modification
C            of D6INT2 which allows extrapolations beyond the
C            interval spanned by the solution history queue.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT2.
C
C   D6INT5 ...
C
C   Purpose: The purpose of D6INT5 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver. This is a modification
C            of D6INT4 which allows extrapolations beyond the
C            interval spanned by the solution history queue. In
C            addition, only selected components are interpolated,
C            as dictated by array ICOMP, rather than the entire
C            solution array.
C
C   D6INT6 ...
C
C   Purpose: The purpose of D6INT6 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver.
C
C   D6INT7 ...
C
C   Purpose: The purpose of D6INT7 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver.
C
C   D6INT8 ...
C
C   Purpose: The purpose of D6INT8 is to evaluate the solution
C            polynomial and the derivative polynomial for the
C            DKLAG6 differential equation solver.
C
C   D6POLY ...
C
C   Purpose: The purpose of D6POLY is to evaluate the solution
C            polynomial and the derivative polynomial for the
C            DKLAG6 differential equation solver.
C
C   D6WSET ...
C
C   Purpose: The purpose of D6WSET is to evaluate the solution
C            and/or the derivative interpolation coefficients
C            for D6POLY.
C
C   D6INT9 ...
C
C   Purpose: The purpose of D6INT9 is to evaluate the solution
C            polynomial and the derivative polynomial for the
C            DKLAG6 differential equation solver using one of
C            the members of a family of imbedded methods of
C            orders 1,...,6.
C
C   D6ERR1 ...
C
C   Purpose: The purpose of D6ERR1 is to calculate an error
C            estimate for the DKLAG6 differential equation
C            solver.
C
C   D6HST1 ...
C
C   Purpose: The function of subroutine D6HST1 is to compute a
C            starting stepsize to be used in solving initial
C            value problems in delay differential equations.
C            It is based on a modification of DHSTRT from the
C            SLATEC library.
C
C   D6HST2 ...
C
C   Purpose: The purpose of D6HST2 is to interpolate the delayed
C            solution and derivative during the first step of
C            the integration for the DKLAG6 differential
C            equation solver.
C
C   D6VEC3 ...
C
C   Purpose: The function of D6VEC3 is to compute the maximum
C            norm for the DKLAG6 differential equation solver.
C            D6VEC3 is the same as DVNORM from the SLATEC
C            library.
C
C   D6GRT1 ...
C
C   Purpose: The purpose of D6GRT1 is to direct the rootfinding
C            for the DKLAG6 differential equation solver.
C
C   D6GRT2 ...
C
C   Purpose: The purpose of D6GRT2 is to perform rootfinding
C            for the DKLAG6 differential equation solver.
C
C   D6GRT3 ...
C
C   Purpose: The purpose of D6GRT3 is to evaluate the residuals
C            for the rootfinder for the DKLAG6 differential
C            equation solver.
C
C   D6GRT4 ...
C
C   Purpose: The purpose of D6GRT4 is to shuffle storage for
C            the rootfinder for the DKLAG6 differential
C            equation solver.
C
C   D6HST3 ...
C
C   Purpose: The purpose of D6HST3 is to calculate the stepsize
C            for the DKLAG6 differential equation solver.
C
C   D6HST4 ...
C
C   Purpose: The purpose of D6HST4 is to limit the stepsize to
C            be no larger than the maximum stepsize for the
C            DKLAG6 differential equation solver.
C
C   D6QUE1 ...
C
C   Purpose: The purpose of D6QUE1 is to update the past solution
C            history queue for the DKLAG6 differential equation
C            solver.
C
C   D6QUE2 ...
C
C   Purpose: The purpose of D6QUE2 is to adjust the last mesh
C            point entry in the solution history queue for
C            the DKLAG6 ordinary differential equation solver,
C            in the event D6DRV2 performs an extrapolation near
C            the final integration time.
C
C   D6VEC1 ...
C
C   Purpose: The purpose of D6VEC1 is to load a constant
C            into a vector for the DKLAG6 differential
C            equation solver.
C
C   D6VEC2 ...
C
C   Purpose: The purpose of D6VEC2 is to multiply a constant
C            times a vector for the DKLAG6 differential
C            equation solver.
C
C   D6PNTR ...
C
C   Purpose: The purpose of D6PNTR is to define and restore
C            pointers and integration flags for the DKLAG6
C            differential equation solver.
C
C   D6SRC1 ...
C
C   Purpose: The purpose of D6SRC1 is to perform binary searches
C            for the DKLAG6 differential equation solver.
C
C   D6SRC2 ...
C
C   Purpose: The purpose of D6SRC2 is to assist in the queue
C            search and interpolation for the DKLAG6 differential
C            equation solver (for non-vanishing delay problems).
C
C   D6SRC3 ...
C
C   Purpose: The purpose of D6SRC3 is to assist in the queue
C            search and interpolation for the DKLAG6 differential
C            equation solver (for vanishing delay problems).
C
C   D6SRC4 ...
C
C   Purpose: The purpose of D6SRC4 is to assist in the queue
C            search and interpolation for the DKLAG6 differential
C            equation solver (for non-vanishing delay problems).
C
C   D6MESG ...
C
C   Purpose: The purpose of D6MESG is to generate error messages
C            for the DKLAG6 differential equation solver.
C
C   D6DISC ...
C
C   Purpose: The purpose of D6DISC is to provide the
C            capability of providing discontinuity
C            information for the initial solution
C            for the DKLAG6 delay solver. This is a
C            user replaceable subroutine.
C
C   D6ESTF ...
C
C   Purpose: The purpose of D6ESTF is to set the flag which
C            which determines the type of per step error
C            estimate to be used by the DKLAG6 differential
C            equation solver. This is a user replaceable
C            subroutine.
C
C   D6BETA ...
C
C   Purpose: The purpose of D6BETA is to check for illegal delays
C            for the DKLAG6 differential equation solver. This is
C            a user replaceable subroutine.
C
C   D6AFTR ...
C
C   Purpose: The purpose of D6AFTR is to provide the
C            capability of allowing the user to
C            update anything in his program following
C            successful integration steps. When
C            D6AFTR is called following a successful
C            integration step, the arguments in the
C            have the same meaning as in the documentation
C            prologue for DKLAG6. None of these arguments
C            may be altered when the call is made:
C            DKLAG6 assumes that no arguments will be
C            changed; unpredictable results may occur
C            if the value of any argument is changed.
C            This is a user replaceable subroutine.
C
C   D6RERS ...
C
C   Purpose: The purpose of D6RERS is to define the floor
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
C
C   D6STAT ...
C
C   Purpose: The purpose of D6STAT is to allow the user to
C            collect integration statistics. This version
C            of D6STAT is a dummy subroutine which does nothing.
C            This is a user replaceable subroutine which may be
C            collect statistics according to the description
C            given below. 
C
C   D6OUTP ...
C
C   Purpose: The purpose of D6OUTP is to define the output
C            increment for the DKLAG6 differential equation
C            solver. This function subroutine simply returns
C            the value of TINC for the output point increment.
C            A time dependent output increment can be used by
C            replacing this function with a user provided one.
C
C   D6RTOL ...
C
C   Purpose: The purpose of D6RTOL is to define convergence
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
C
C   XGETUA ...
C
C   Purpose: Return unit number(s) to which error messages are being
C            sent.
C
C   J4SAVE ...
C
C   Purpose: J4SAVE saves and recalls several global variables needed
C            by the library error handling routines.
C
C   DAXPY ...
C
C   Purpose: The purpose of DAXPY is to copy a constant times
C            a vector plus another vector to a third vector
C            for the DKLAG6 differential equation solver.
C            DAXPY is the same as DAXPY from the SLATEC library.
C
C   DCOPY ...
C
C   Purpose: The purpose of DCOPY is to copy a vector to another
C            vector for the DKLAG6 differential equation solver.
C            DCOPY is the same as DCOPY from the SLATEC library.
C
C   SAXPY ...
C
C   Purpose: The purpose of SAXPY is to copy a constant times
C            a vector plus another vector to a third vector
C            for the SRKLAG differential equation solver.
C            SAXPY is the same as SAXPY from the SLATEC library.
C
C   SCOPY ...
C
C   Purpose: The purpose of SCOPY is to copy a vector to another
C            vector for the SRKLAG differential equation solver.
C            SCOPY is the same as SCOPY from the SLATEC library.
C
C   XERCNT ...
C
C   Purpose: Allow user control over handling of errors.
C
C   XERDMP ...
C
C   Purpose: Print the error tables and then clear them.
C
C   XERHLT ...
C
C   Purpose: Abort program execution and print error message.
C
C   XERMSG ...
C
C   Purpose: Processes error messages for SLATEC and other libraries
C
C   XERPRN ...
C
C   Purpose: This routine is called by XERMSG to print error messages
C
C   XERSVE ...
C
C   Purpose: Record that an error has occurred.
C
C   XSETF ...
C
C   Purpose: Set the error control flag.
C
C   XSETUA ...
C
C   Purpose: Set logical unit numbers (up to 5) to which error
C            messages are to be sent.
C
C   R1MACH ...
C
C   Purpose: Returns single precision machine dependent constants
C
C   D1MACH ...
C
C   Purpose: Returns double precision machine dependent constants
C
C   I1MACH ...
C
C   Purpose: Returns integer machine dependent constants
C
C----------------------------------------------------------------
C
C THIS IS THE BEGINNING OF THE FORTRAN SOURCE CODE FOR DKLAG6.
C
C----------------------------------------------------------------
C
C*DECK DKLAG6
      SUBROUTINE DKLAG6(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,
     +                  YINIT,DYINIT,N,W,LW,IW,LIW,QUEUE,LQUEUE,
     +                  IFLAG,LIN,LOUT,RPAR,IPAR)
C
C     SUBROUTINE DKLAG6(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,
C    +                  YINIT,DYINIT,N,W,LW,IW,LIW,QUEUE,LQUEUE,
C    +                  IFLAG,LIN,LOUT,RPAR,IPAR)
C
C SECTION VI.
C
C                    DOCUMENTATION PROLOGUE
C
C***BEGIN PROLOGUE  DKLAG6
C***PURPOSE  The function of subroutine DKLAG6 is to solve N
C            ordinary differential equations with state dependent
C            delays. DKLAG6 is applicable to the integration of
C            systems of differential equations of the form
C
C            dY(T)/dT = F(T, Y(T), Y(BETA(T,Y)), Y'(BETA(T,Y))
C
C            given the initial function Y(T) = YINIT(T)
C            and the derivative of this function
C            DY(T) = DYINIT(T) for T between BETA(T0,Y(T0))
C            and TINIT where TINIT is the initial integration
C            time. DKLAG6 is applicable to several types of
C            problems, including constant delay problems, strictly
C            time dependent delay problems, state dependent
C            delay problems, and neutral problems (ones in which
C            the derivative at the solution is needed at the
C            delay times).
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
C     INTRODUCTION:
C     ____________
C
C     This section describes some of the characteristics of
C     "delay" differential equations. Readers familiar with
C     these ideas should skip directly to the section
C     "USING DKLAG6".
C
C     The "delay", "lag", or "retarded" differential equations
C     considered here are in the form
C
C         dY(T)/dT = F(T, Y(T), Y(BETA(T,Y)), Y'(BETA(T,Y))
C
C     meaning that the right hand side of the equation may depend
C     on the solution Y and/or its derivative Y' at "time" values
C     other than the current time "T". Examples of such equations
C     are the following. Throughout this description, the delay
C     is BETA(T,Y) and the actual delay time is T - BETA(T,Y).
C
C      Y'(T) = (Y(T-1))**2                (delay function T-1)
C      Y'(T) = (Y'(T-1))**2               (delay function T-1)
C      Q'(T) = (1/T)*EXP(Q(Q(T)-LN(2)+1)) (delay function
C                                          Q(T)-LN(2)+1)
C      Z'(T) = Z(T-1)*Z(T-2)              (multiple delay functions
C                                          T-1 & T-2, for the same
C                                          solution component,
C                                          disallowed by DKLAG6)
C      X'(T) = X(T/2)                     (vanishing delay problem
C                                          with delay T/2)
C      W'(T) = W(LOG(W(T)))               (delay function LOG(W(T)),
C                                          assuming W is positive)
C      Y'(T) = Y(T+1)                     (LAG FUNCTION T+1,
C                                          disallowed by DKLAG6, if
C                                          the integration goes left
C                                          to right, since the delay
C                                          function exceeds T)
C
C      We denote the delay functions by BETA(T,Y) and always
C      assume that they represent "past" values
C      of the independent variable, i.e., if integration is
C      from left to right then BETA(T,Y) .LE. T.
C
C      Two characteristics of delay equations distinguish them
C      from the usual non-delay equations.
C
C      (1) Initial conditions. Instead of the initial conditions
C          being given at a single point, delay equations usually
C          have the initial conditions given on an interval. For
C          example, the first equation above would typically have
C          initial conditions given for all T<0.
C
C      (2) Discontinuities. The solution of a delay equation often
C          has jumps in the first or higher order derivatives. The
C          simplest delay equation has a "constant delay" such as the
C          first equation above. Depending on the initial function
C          its solution has a derivative jump at T=1,2,3, .... In
C          general, if BETA is the delay function and T0 is the
C          initial time, the solution can have a derivative jump
C          when BETA-T0=0. If this occurs at T1, the derivative may
C          then have a jump when BETA-T1=0, etc. Thus the number of
C          instances where the derivative jumps can grow as the
C          integration proceeds. DKLAG6 contains a provision to
C          automatically locate such jump discontinuities and prevent
C          them from degrading the accuracy of the solution. This type
C          of discontinuity is different from those associated with
C          "special events" which are often encountered in ordinary
C          differential equations. DKLAG6 is capable of handling the
C          latter type of discontinuity via the GUSER option.
C
C     Multiple Lags:
C
C     DKLAG6 is limited in that there can only be one delay function
C     for each component of the system. In other words, the system
C
C      Y'(T) = Y(T-1)**2
C      Z'(T) = Y(T-1)+Z(T-2)
C
C     is allowed, but the following is not.
C
C      Y'(T) = Y(T-1)**2
C      Z'(T) = Z(T-1)+Z(T-2)
C
C     Systems like the second one can usually be transformed to
C     ones without multiple delays for the same component, using a
C     technique referred to as "augmenting the system." (Readers
C     not familiar with this technique may want to consult the
C     following reference: K. W. Neves, Automatic Integration of
C     Functional Differential Equations - An Approach, ACM TOMS,
C     Vol. 1, No. 4, 1975, pp. 357-368.)
C
C     To "augment the system" for the first equation above, the
C     system could be transformed to the following one:
C
C     Y'(T) = Y(T-1)**2
C     Z'(T) = Z(T-1)+W(T-2)
C     W'(T) = Z(T-1)+W(T-2)
C
C     In the transformed system, Z and W are really the same; but
C     each component has only one delay associated with it. DKLAG6
C     may therefore be used to solve the transformed system.
C
C     DKLAG6 is based on methods designed for nonstiff or mildly
C     stiff systems (loosely speaking, ones which do not have
C     "time constants" with greatly different scales). A solver
C     based on stiff methods should be used in lieu of DKLAG6 for
C     stiff systems (e.g., STRIDE).
C
C     Vanishing Lags:
C
C     If the delay BETA - T vanishes at isolated points, the problem
C     is said to be a vanishing delay problem. DKLAG6 contains two
C     solution options. If EXTRAP = .FALSE., the solver will use
C     methods designed for problems which do not have vanishing
C     delays. This is the recommended option used by the solver.
C     However, if this option is used
C     for a problem with vanishing delays, the solver will usually
C     terminate with a ``stepsize is too small for machine precision''
C     error message. This is due to the fact that the solver restricts
C     the stepsize to avoid delay function values which are larger than
C     the last successful integration time. Consequently, the
C     stepsize will go to zero if the delay vanishes. For problems of
C     this type, EXTRAP = .TRUE. should be used. In this case, if
C     the solver encounters a delay function value which exceeds the
C     last successful integration time, it will extrapolate the
C     solution polynomials for the previous successful step, to
C     approximate the corresponding delayed solution value. The
C     stepsize therefore is not forced to zero at points for
C     which the delay vanishes. In order to ensure the extrapolated
C     values are accurate, the solver chooses performs a predictor-
C     corrector like iteration of the solution polynomials and
C     forces convergence of the solution and derivative approximations.
C     This iteration is performed only when a delay time falls
C     beyond the solution history queue.
C
C     Caution:
C
C     Any delay solver may experience difficulty in solving a
C     neutral problem (one in which y'(t) depends on a delayed
C     derivative). This is particularly true if a delay also
C     vanishes at some point. For example, consider the equation
C
C        y'(t) = y(t) * (1 - y(t)) + F(t) + y'(t - g(t)).
C
C     If F(t_v) = 0 and the delay g(t) vanishes at t_v so that
C     g(t_v) = 0, the two solutions of the resulting equation
C     are y(t_v) = 0 and y(t_v) = 1. By defining F(0) appropriately,
C     it is possible to have y(t) approaching a value other than
C     0 or 1 from the left at t_v. Therefore, the solution will
C     have a discontinuity at t_v. In general, for a neutral problem,
C     there results a system of implicit equations at points for
C     which the delay vanishes. The predictor-corrector iteration
C     used by DKLAG6 at such points enables it to solve many such
C     problems. However, care must be exercised in interpreting
C     the numerical results for any such problem.
C
C     Ordinary Differential Equations Without Delays:
C
C     If the delay vanishes identically, that is, BETA - T = 0 for
C     all T, this will have no effect on the solver. Consequently,
C     DKLAG6 can solve systems for which some (or all) of the
C     equations are non-delay equations. Of course, if a system
C     contains no delays, the user should consider using a standard
C     ode solver (since additional overhead is required by DKLAG6
C     to accommodate the possibility of delays).
C
C     Methods Used:
C
C     DKLAG6 is based on continuously imbedded Runge-Kutta
C     methods of Sarafyan (see references below). These methods
C     have underlying continuously differentiable Runge-Kutta
C     piecewise polynomial interpolants which allow the use
C     of interpolation. In this respect they are similar to the
C     linear multistep methods used in many ode solvers. The
C     polynomials are used to approximate the solution for values
C     of the delay functions less than the current time value.
C
C     USING DKLAG6:
C     ____________
C
C     DKLAG6 may be described as a "problem oriented" code. Using
C     DKLAG6 is more complicated than using a standard "interval
C     oriented" ode solver. DKLAG6 requires a subroutine to provide
C     the initialization information, one to print or otherwise
C     process the solution at output points, and a subroutine to
C     evaluate the derivatives F(T,Y,Y(BETA),Y'(BETA)). In addition,
C     it requires subroutines to evaluate the initial solution and
C     its derivative and one to evaluate the delay functions. Problem
C     information is communicated to DKLAG6 through these subroutines.
C     DKLAG6 returns control to the calling program only after it has
C     reached the final integration time. Users who find this approach
C     undesirable may wish to use the second level drivers D6DRV2
C     or D6DRV4 (constant delays) directly.
C
C     To use DKLAG6 (or DKLG6C or DRKS56) do the following:
C
C     First check that the machines constants which are defined
C     in D1MACH are appropriate for the computer you are using.
C
C     If you are calling the highest level driver DRKS56 rather than
C     DKLAG6, set the logical variable CONLAG to .FALSE. (state
C     dependent delays) or .TRUE. (constant delays). These are
C     equivalent to calling DKLAG6 or DKLG6C, respectively.
C
C     Note that IFLAG (output) is the integration termination
C     flag. IFLAG = 0 indicates no errors were encountered.
C     IFLAG .NE. 0 indicates an error was encountered.
C     In this case, refer to the printed error message(s).
C     You should monitor the value of IFLAG returned by DKLAG6.
C
C     Supply the the two parameters LIN AND LOUT. These
C     are respectively, the numbers for the input and
C     output units.
C
C     Define the number of differential equations N.
C     N must be positive.
C
C     RPAR and IPAR are dummy arrays that DKLAG6 passes to
C     the user supplied subroutines. It does not alter them
C     or use them in any way. They may be used to communicate
C     information between the user subroutines. RPAR is
C     DOUBLE PRECISION type and IPAR is INTEGER type.
C
C     Supply the W work array of length LW. LW must be at
C     least 34*N + 7*NG + 1, where NG is the
C     is the maximum number of root functions that will
C     be used at any time. NG will never be less than N
C     if IROOT=0 in SUBROUTINE INITAL. NG will be 0
C     if IROOT=1 in INITAL. If IROOT=0 in SUBROUTINE INITAL,
C     the code automatically looks for roots of functions
C               G(I) = BVAL(Y,T) - TG(I)
C     where BVAL is the delay defined in SUBROUTINE BETA and
C     TG is any previous such root. Each time such a root
C     is found, additional G equations are added to the
C     root search. For this reason, it is not possible to
C     tell in advance the amount of storage that will be
C     needed for the root information. The user should also
C     note that if IROOT=0 for a problem with constant delays,
C     a root will be located for each integral multiple of
C     the delay. Functions of the above form are used because
C     derivative discontinuities can occur at the roots of
C     these functions. Unless derivative discontinuities are
C     suspected, it is generally a good idea to use IROOT=1.
C     In addition to the above rootfinding which requires no
C     user intervention, the user may define other "event"
C     functions for which roots will be sought. See the
C     discussion below related to subroutine GUSER.
C
C     Note:
C     Following a suuceesful return, the solution vector at
C     TFINAL begins in location 1+6N of the W array.
C
C     Supply the IW integer work array of length LIW. LIW
C     must be at least 35 + 3*NG where NG is the
C     is the maximum number of root functions that will
C     be used at any time.
C
C     Supply the solution history QUEUE array of length
C     LQUEUE. LQUEUE should be approximately
C           (12*(N+1)) * (TFINAL-TINIT)/HAVG ,
C     where TINIT and TFINAL are the beginning and final
C     integration times and HAVG is the average stepsize
C     that will be used by the integrator. If LQUEUE is
C     too small, it may be necessary to terminate the
C     integration prematurely in order to avoid extrapolating
C     to the left of the history queue for a delay variable.
C     Some experimentation may be required in order to
C     determine an appropriate value for LQUEUE.
C     LQUEUE can never be less than 12*(N+1) and usually must
C     be much, much larger.
C
C     Supply the following subroutines:
C
C     (See below for specific examples.)
C
C     BETA    - evaluate the system delay functions
C
C     DERIVS  - calculate the system derivatives
C
C     DPRINT  - process the solution at output points
C
C     INITAL  - define the problem parameters and
C               initialize the integrator
C
C     YINIT   - evaluate the solution on the initial
C               delay interval
C
C     DYINIT  - evaluate the derivative on the initial
C               delay interval
C
C               Note:
C               If the evaluation of the system derivatives in
C               subroutine DERIVS does not require DYLAG, it is
C               okay to pass the name YINIT for DYINIT in the
C               call to DERIVS.
C
C     GUSER   - define the residuals for the user defined
C               root functions
C
C     CHANGE  - make changes in the problem parameters
C               at roots of the event functions
C
C     Description of user supplied subroutines:
C
C     (For a specific example of code usage, refer to the
C     sample program given below.)
C
C  1. SUBROUTINE BETA(N,T,Y,BVAL,RPAR,IPAR)
C
C     INPUT -
C
C     N                = number of equations
C     T                = independent variable (time)
C     Y(I), I=1,...,N  = solution at time T
C     RPAR(*),IPAR(*)  = user arrays passed via DKLAG6
C
C     This subroutine must define the system
C     delay function BVAL(I), I= 1,...,N. For each
C     T, BVAL(I) must be less than or equal to
C     T if the integration goes from left to right
C     and greater than or equal to T if it goes
C     from right to left. BVAL(I) is the value
C     of the independent variable used in the
C     approximation of the I-th delay solution value.
C     The actual delay for this component is
C     BVAL(I) - T.
C
C     You may terminate the integration by setting N to
C     a negative value when BETA is called.
C
C  2. SUBROUTINE DERIVS(N,T,Y,YLAG,DYLAG,DY,RPAR,IPAR)
C
C     INPUT -
C
C     N                   = number of odes
C     T                   = independent variable (time)
C     Y(I), I=1,...,N     = solution at time T
C     YLAG(I), I=1,...,N  = delayed solution
C     DYLAG(I), I=1,...,N = delayed derivative
C                           Caution:
C                           IF BETA(Y(I),T) = T, DYLAG(I)
C                           is not supplied when DERIVS is
C                           called (that is, no attempt is
C                           made to solve differential-
C                           algebraic systems).
C     RPAR(*),IPAR(*)     = user arrays passed via DKLAG6
C
C     This subroutine must define the system derivatives
C     DY(I), I=1,...,N.
C
C     You may terminate the integration by setting N to
C     a negative value when DERIVS is called.
C
C  3. SUBROUTINE DPRINT(N,T,Y,DY,NG,JROOT,INDEXG,ITYPE,
C    *                  LOUT,INDEX,RPAR,IPAR)
C
C     N                    = number of odes
C     T                    = independent variable (time)
C     Y(I), I=1,...,N      = solution at time T
C     DY(I), I=1,...,N     = derivative at time T
C     NG                   = number of event functions
C     JROOT(I), I=1,...,NG = integrator event function flag
C                            array
C                            If JROOT(I) = 1, the I-th residual
C                            function has a zero at T.
C     INDEXG(I), I=1,...,N = component number indicator for
C                            the root functions
C                            INDEXG(I) indicates which of the
C                            system delays is responsible for
C                            the zero at T of the residual
C                            function.
C     LOUT                 = output unit number
C     INDEX                = last value of index from the
C                            integrator
C     RPAR(*),IPAR(*)      = user arrays passed via DKLAG6
C
C     This subroutine is called by DKLAG6 to allow the
C     user to print and otherwise process the solution
C     at the initial time, at multiples of TINC, and at
C     roots of the event functions.
C     If ITYPE = 0, this call to DPRINT corresponds to
C     a root of an event function. Otherwise, it
C     corresponds to a multiple of the print increment
C     TINC or the initial time. Do not change any of
C     the input parameter values when DPRINT is called.
C
C     You may terminate the integration by setting N to
C     a negative value when DPRINT is called.
C
C  4. SUBROUTINE INITAL(LOUT,LIN,N,NGUSER,TOLD,TFINAL,TINC,
C    *                  YOLD,IROOT,HINIT,HMAX,ITOL,ABSERR,
C    *                  RELERR,EXTRAP,NUTRAL,RPAR,IPAR)
C
C     INPUT -
C
C     N                = number of differential equations
C                        (supplied by calling program to
C                        DKLAG6)
C     LOUT             = output unit number
C     LIN              = input unit number
C     RPAR(*),IPAR(*)  = user arrays passed via DKLAG6
C
C     This subroutine must define the following
C     problem dependent data.
C
C     NGUSER             = number of user defined root functions
C     TOLD               = initial value of independent
C                          variable
C     TFINAL             = final value of independent
C                          variable
C     TINC               = independent variable increment
C                          at which the solution and
C                          derivative are to be
C                          interpolated
C                          Consider replacing D6OUTP if a
C                          nonconstant TINC is desired.
C     YOLD(I), I=1,...,N = initial solution Y(TOLD)
C     IROOT              = rootfinding flag. If IROOT=0
C                          rootfinding will be used to
C                          locate points with derivative
C                          discontinuities. If IROOT=1
C                          such discontinuities will be
C                          ignored.
C     HINIT              = initial stepsize for DKLAG6. If
C                          HINIT = 0.0D0, DKLAG6 will determine
C                          the initial stepsize.
C     HMAX               = maximum allowable stepsize
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
C     ABSERR(I)          = local absolute error tolerance
C        I = 1,...,ITOL
C     RELERR(I)          = local relative error tolerance
C        I = 1,...,ITOL
C     EXTRAP             = logical method flag. If this
C                          flag is not set .TRUE. by
C                          INITAL, the default value which
C                          will be used is .FALSE., in
C                          which case the solver will assume
C                          this is not a vanishing delay problem.
C                          Refer to the discussion in the
C                          prologue regarding ``Vanishing Lags''
C                          for a detailed description of this
C                          parameter. If NUTRAL is set .TRUE.,
C                          EXTRAP will also be set .TRUE. after
C                          call to INITAL.
C
C     NUTRAL             = set the logical flag neutral = .TRUE. if
C                          the problem is a neutral one, that is,
C                          if a delayed derivative is used in the
C                          calculation of dy/dt; otherwise set
C                          nutral = .false. If this flag is not
C                          set by INITAL, the default value which
C                          will be used is .FALSE., in which case
C                          the solver will assume this is not a
C                          neutral problem. If NUTRAL is set .TRUE.,
C                          EXTRAP will also be set .TRUE. after the
C                          CALL TO INITAL.
C                          
C     RPAR(*),IPAR(*)    = user arrays passed via DKLAG6
C
C     You may terminate the integration by setting N to
C     a negative value when INITAL is called.
C
C  5. SUBROUTINE YINIT(N,T,Y,I,RPAR,IPAR)
C
C     INPUT -
C
C     N                = number of equations
C     T                = independent variable (time)
C     I                = component number
C     RPAR(*),IPAR(*)  = user arrays passed via DKLAG6
C
C     This subroutine must define Y = Y(T,I), the solution
C     at time T corresponding to the initial delay interval
C     for component I (not the entire Y vector, just the
C     I-th component). YINIT is called any time the code
C     needs a value of the solution corresponding to the
C     initial delay interval. If no delayed solution values
C     are involved in the evaluation of the system derivatives
C     (that is, the problem is a neutral problem in which
C     only delayed derivative values are involved in the
C     calculation of the system derivatives), the name DYINIT
C     may be used for YINIT.
C
C     You may terminate the integration by setting N to
C     a negative value when YINIT is called.
C
C  6. SUBROUTINE DYINIT(N,T,DY,I,RPAR,IPAR)
C
C     INPUT -
C
C     N                = number of equations
C     T                = independent variable (time)
C     I                = component number
C     RPAR(*),IPAR(*)  = user arrays passed via DKLAG6
C
C     This subroutine must define DY = DY(T,I), the derivative
C     at time T corresponding to the initial delay interval for
C     component I (not the entire DY vector, just the I-th
C     component). DYINIT is called any time the code needs
C     a value of the derivative corresponding to the initial
C     delay interval. If no delayed derivatives are involved in
C     the evaluation of the system derivatives, the name YINIT
C     may be used for DYINIT.
C
C     You may terminate the integration by setting N to
C     a negative value when DYINIT is called.
C
C  7. SUBROUTINE GUSER(N,T,Y,DY,NGUSER,G,RPAR,IPAR)
C
C     INPUT -
C
C     N                    = number of odes
C     T                    = independent variable (time)
C     Y(I), I=1,...,N      = solution at time T
C     DY(I), I=1,...,N     = approximation to derivative at time T
C                            (not exact derivative from DERIVS)
C     NGUSER               = number of event functions
C     RPAR(*),IPAR(*)      = user arrays passed via DKLAG6
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
C  8. SUBROUTINE CHANGE(INDEX,JROOT,INDEXG,NG,N,YROOT,
C    *                  DYROOT,HINIT,TROOT,TINC,ITOL,
C    *                  ABSERR,RELERR,LOUT,RPAR,IPAR)
C
C     INPUT -
C
C     INDEX                = last value of index from the
C                            integrator
C     JROOT(I), I=1,...,NG = integrator event function flag
C                            array
C                            If JROOT(I) = 1, the I-th residual
C                            function has a zero at T.
C     INDEXG(I), I=1,...,N = component number indicator for
C                            the root functions
C                            INDEXG(I) indicates which of the
C                            system delays is responsible for
C                            the zero at T of the residual
C                            function.
C     NG                   = number of event functions
C     N                    = number of odes
C     YROOT(I), I=1,...,N  = solution at time TROOT
C     DYROOT(I), I=1,...,N = derivative at time TROOT
C     HINIT                = initial stepsize
C     TROOT                = value of the independent variable
C                            at which the root occurred
C     TINC                 = independent variable increment
C                            at which the solution and
C                            derivative are to be
C                            interpolated
C     ITOL                 = error control flag. If ITOL = 1
C                            ABSERR and RELERR are vectors of
C                            length 1. If ITOL = N they are
C                            vectors of length N.
C     ABSERR(I)            = local absolute error tolerance(s)
C        I = 1,...,ITOL
C     RELERR(I)            = local relative error tolerance(s)
C        I = 1,...,ITOL
C     LOUT                 = output unit number
C     RPAR(*),IPAR(*)      = user arrays passed via DKLAG6
C
C     When CHANGE is called, any of the parameters HINIT, TINC,
C     ITOL, ABSERR, and RELERR may be changed.
C
C     Use a dummy subroutine which does nothing if no changes
C     are required at any root.
C
C     If automatic discontinuity is being used (IROOT = 0 in
C     subroutine INITAL), it may be turned off when CHANGE
C     is called by setting NG = -1.
C
C     You may terminate the integration by setting N to
C     a negative value when CHANGE is called.
C
C     SAMPLE PROGRAM
C     ______________
C
C     PROGRAM DEMO
C
C     This demonstration program solves the following problem:
C
C          DY(1) = Y(1) * Y(LOG(Y(1))) / T,
C          DY(2) = Y(Y(2)-SQRT(2.0D0)+1.0D0)/(2.0D0*SQRT(T)),
C                                               T > 1
C          Y(1) = 1 FOR T .LE. 1 , Y(2) = 1 FOR T .LE. 1
C          Y(2) = 1 FOR T .LE. 1 , Y(2) = 1 FOR T .LE. 1
C
C     SUBROUTINE BETA evaluates the delay functions:
C
C          BVAL(1) = LOG(Y(1))
C          BVAL(2) = Y(2) - SQRT(2.0D0) + 1.0D0
C
C     SUBROUTINE DERIVS evaluates the derivatives:
C
C          DY(1) = Y(1) * YLAG(1) / T
C          DY(2) = YLAG(2) / (2.0D0 * SQRT(T))
C
C          (When DERIVS is called by the solver,
C          YLAG(1) contains an approximation to
C          the delayed solution Y(LOG(Y(1))); and
C          YLAG(2) contains an approximation to the
C          delayed solution Y(Y(2)-SQRT(2.0D0)+1.0D0).)
C
C     SUBROUTINE YINIT evaluates the initial function:
C
C          Y(1) = Y(2) = 1 FOR T .LE. 1
C
C     SUBROUTINE DYINIT is not needed for this problem.
C     YINIT will be passed in its place in the call to DKLAG6.
C
C     SUBROUTINE INITAL instructs the solver to use the
C     following parameters:
C
C          NGUSER    = 0        = number of user defined
C                                 root functions
C          TOLD      = 1.0D0    = initial integration time
C          TFINAL    = 3.0D0    = final integration time
C          TINC      = 0.50D0   = output print increment
C          YOLD(1)   = 1.0D0    = Y(1) at TOLD
C          YOLD(2)   = 1.0D0    = Y(2) at TOLD
C          ABSERR(1) = 1.0D-5   = integration absolute error
C                                 tolerance
C          RELERR(1) = 1.0D-5   = integration relative error
C                                 tolerance
C          IROOT     = 0          to instruct the solver to
C                                 locate times with potential
C                                 derivative discontinuities
C          HINIT     = 1.0D-5   = initial stepsize
C          HMAX      = TFINAL - TOLD
C                               = maximum allowable stepsize
C          EXTRAP    = .TRUE.   = method flag to tell the
C                                 solver this is not a
C                                 vanishing delay problem
C          NUTRAL    = .FALSE.  = problem type flag to tell
C                                 the solver whether this is
C                                 a neutral problem
C
C     SUBROUTINE DPRINT prints the solution and the errors in the
C     computed solution. (The exact solution is partially known
C     for this test problem.)
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION DW(100),IW(100),DQ(1000),RPAR(1),IPAR(1)
C
C     USER SUPPLIED PROBLEM DEPENDENT SUBROUTINES ...
C
C     EXTERNAL INITAL,DERIVS,GUSER,DPRINT,BETA,YINIT,DYINIT,CHANGE
C
C     LWDIM = DIMENSION OF DW ABOVE
C     LQDIM = DIMENSION FOR DQ
C     LIDIM = DIMENSION FOR IW
C
C     DATA LWDIM/100/,LIDIM/100/,LQDIM/1000/
C
C     LIN = UNIT NUMBER FOR INPUT.
C
C
C     DATA LIN/5/
C
C     LOUT = UNIT NUMBER FOR OUTPUT.
C
C     DATA LOUT/6/
C
C     OPEN THE INPUT AND OUTPUT FILES.
C
C     OPEN (5,FILE='demo.dat')
C     OPEN (6,FILE='demo.ans')
C
C     DEFINE THE NUMBER OF DIFFERENTIAL EQUATIONS.
C
C     N = 2
C
C     DEFINE THE LENGTHS OF THE WORK ARRAYS.
C
C     LW = LWDIM
C     LI = LIDIM
C     LQ = LQDIM
C
C     CALL THE DELAY SOLVER.
C
C     CALL DKLAG6(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,
C    +            YINIT,N,DW,LW,IW,LI,DQ,LQ,IFLAG,LIN,LOUT,
C    +            RPAR,IPAR)
C
C     WRITE (LOUT,99999) IFLAG
C
C99999 FORMAT (/,' TERMINATION FLAG = ',I2)
C
C     STOP
C     END
C
C     SUBROUTINE BETA(N,T,Y,BVAL,RPAR,IPAR)
C
C     INPUT -
C
C     N                = NUMBER OF EQUATIONS
C     T                = INDEPENDENT VARIABLE (TIME)
C     Y(I), I=1,...,N  = SOLUTION AT TIME T
C     RPAR(*),IPAR(*)  = USER ARRAYS PASSED VIA DKLAG6
C
C
C     THIS TEST SUBROUTINE DEFINES THE SYSTEM DELAYS
C     BVAL(I), I= 1 ,..., N.
C
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION BVAL(N),Y(N),RPAR(*),IPAR(*)
C
C     BVAL(1) = LOG(Y(1))
C     BVAL(2) = Y(2) - SQRT(2.0D0) + 1.0D0
C
C     RETURN
C     END
C
C     SUBROUTINE DERIVS(N,T,Y,YLAG,DYLAG,DY,RPAR,IPAR)
C
C     INPUT -
C
C     N                = NUMBER OF ODES
C     T                = INDEPENDENT VARIABLE
C     Y(*)             = SOLUTION AT TIME T
C     YLAG(*)          = DELAYED SOLUTION
C     DYLAG(*)         = DELAYED DERIVATIVE
C     RPAR(*),IPAR(*)  = USER ARRAYS PASSED VIA DKLAG6
C
C     THIS TEST SUBROUTINE DEFINES THE DERIVATIVES
C     DY(I), I=1 ,..., N.
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION Y(N),DY(N),YLAG(N),DYLAG(N),RPAR(*),IPAR(*)
C
C     DY(1) = Y(1)*YLAG(1)/T
C     DY(2) = YLAG(2)/ (2.0D0*SQRT(T))
C
C     RETURN
C     END
C
C     SUBROUTINE YINIT(N,T,Y,I,RPAR,IPAR)
C
C     INPUT -
C
C     N                = NUMBER OF EQUATIONS
C     T                = INDEPENDENT VARIABLE (TIME)
C     I                = COMPONENT NUMBER
C     RPAR(*),IPAR(*)  = USER ARRAYS PASSED VIA DKLAG6
C
C     THIS TEST SUBROUTINE DEFINES Y, THE SOLUTION AT
C     TIME T CORRESPONDING TO THE INITIAL DELAY INTERVAL
C     FOR COMPONENT I (NOT THE ENTIRE Y VECTOR, JUST
C     I-TH COMPONENT).
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION RPAR(*),IPAR(*)
C
C     IF (I.EQ.1) Y = 1.0D0
C     IF (I.EQ.2) Y = 1.0D0
C
C     RETURN
C     END
C
C     SUBROUTINE INITAL(LOUT,LIN,N,NGUSER,TOLD,TFINAL,TINC,YOLD,
C    +               IROOT,HINIT,HMAX,ITOL,ABSERR,RELERR,EXTRAP,
C    +               NUTRAL,RPAR,IPAR)
C
C     INPUT -
C
C     N                = NUMBER OF DIFFERENTIAL EQUATIONS
C     LOUT             = OUTPUT UNIT NUMBER
C     LIN              = INPUT UNIT NUMBER
C     RPAR(*),IPAR(*)  = USER ARRAYS PASSED VIA DKLAG6
C
C     THIS TEST SUBROUTINE DEFINES THE FOLLOWING PARAMETERS ...
C
C     NGUSER   = NUMBER OF USER DEFINED ROOT FUNCTIONS
C     TOLD     = INITIAL VALUE OF INDEPENDENT VARIABLE
C     TFINAL   = FINAL VALUE OF INDEPENDENT VARIABLE
C     TINC     = INDEPENDENT VARIABLE INCREMENT AT WHICH
C                THE SOLUTION AND DERIVATIVE ARE TO BE
C                INTERPOLATED
C     YOLD     = INITAL SOLUTION Y(TOLD)
C     IROOT    = ROOTFINDING FLAG. IF IROOT=0 ROOTFINDING
C                WILL BE USED TO LOCATE POINTS WITH
C                DERIVATIVE DISCONTINUITIES. IF IROOT=1
C                SUCH DISCONTINUITIES WILL BE IGNORED.
C     HINIT    = INITIAL STEPSIZE. IF HINIT = 0.0D0,
C                DKLAG6 WILL DETERMINE THE INITIAL
C                STEPSIZE.
C     HMAX     = MAXIMUM ALLOWABLE STEPSIZE
C     ITOL     = ERROR TYPE FLAG. IF ITOL = 1, ABSERR AND
C                RELERR ARE VECTORS OF LENGTH 1. IF ITOL = N
C                THEY ARE VECTORS OF LENGTH N.
C     ABSERR   = INTEGRATION ABSOLUTE ERROR TOLERANCE(S)
C     RELERR   = INTEGRATION RELATIVE ERROR TOLERANCE(S)
C     EXTRAP   = LOGICAL METHOD FLAG
C     NUTRAL   = LOGICAL PROBLEM TYPE FLAG
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     LOGICAL EXTRAP,NUTRAL
C
C     DIMENSION YOLD(N),ABSERR(*),RELERR(*),RPAR(*),IPAR(*)
C
C     NGUSER = 0
C     TOLD   = 1.0D0
C     TFINAL = 3.0D0
C     TINC   = 0.50D0
C     DO 20 I = 1 , N
C        CALL YINIT(N,TOLD,YOLD(I),I,RPAR,IPAR)
C  20 CONTINUE
C     ITOL = 1
C     ABSERR(1) = 1.0D-5
C     RELERR(1) = 1.0D-5
C     IROOT  = 0
C     HINIT  = 1.0D-5
C     HMAX   = TFINAL - TOLD
C     EXTRAP = .TRUE.
C     NUTRAL = .FALSE.
C
C     RETURN
C     END
C
C     SUBROUTINE DPRINT(N,T,Y,DY,NG,JROOT,INDEXG,ITYPE,LOUT,INDEX,
C    +                  RPAR,IPAR)
C
C     INPUT -
C
C     N                = NUMBER OF ODES
C     T                = INDEPENDENT VARIABLE (TIME)
C     Y(*)             = SOLUTION AT TIME T
C     DY(*)            = DERIVATIVE AT TIME T
C     NG               = NUMBER OF EVENT FUNCTIONS
C     JROOT(*)         = INTEGRATOR EVENT FUNCTION FLAG ARRAY
C     INDEXG(*)        = COMPONENT NUMBER FLAG ARRAY
C     LOUT             = OUTPUT UNIT NUMBER
C     INDEX            = LAST VALUE OF INDEX FROM THE INTEGRATOR
C     RPAR(*),IPAR(*)  = USER ARRAYS PASSED VIA DKLAG6
C
C     THIS TEST SUBROUTINE PROCESSES THE SOLUTION AT OUTPUT POINTS.
C
C     IF ITYPE = 0, THIS CALL TO DPRINT CORRESPONDS TO
C     A ROOT OF AN EVENT FUNCTION. OTHERWISE, IT CORRESPONDS
C     TO A MULTIPLE OF THE OUTPUT INCREMENT TINC.
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION Y(N),DY(N),JROOT(*),INDEXG(*),RPAR(*),IPAR(*)
C
C     DIMENSION EXACT(2)
C
C     WRITE THE SOLUTION AND THE DERIVATIVES.
C
C     WRITE (LOUT,99994)
C
C     WRITE (LOUT,99999) T
C     WRITE (LOUT,99998) (Y(I),DY(I),I=1,N)
C
C     DETERMINE IF A ROOT OF AN EVENT FUNCTION WAS LOCATED.
C
C     IF (ITYPE.EQ.0) THEN
C         DO 20 I = 1,NG
C             IROOT = JROOT(I)
C             IF (IROOT.EQ.1) WRITE (LOUT,99997) I,INDEXG(I)
C
C             IF (IROOT.EQ.1) THEN
C                 IF (I.EQ.2) THEN
C                     TEXACT = 2.0D0
C                 ENDIF
C                 IF (I.EQ.1) THEN
C                     TEXACT = EXP(1.0D0)
C                 ENDIF
C                 TERROR = TEXACT - T
C                 WRITE (LOUT,99995) TERROR
C             ENDIF
C
C  20     CONTINUE
C     ENDIF
C
C     CALCULATE THE MAXIMUM ERROR IN THE COMPUTED SOLUTION
C     AND DERIVATIVE FOR THIS TEST PROBLEM.
C
C     CALL TRUSOL(N,T,EXACT)
C
C     CALL YERROR(N,Y,EXACT,ERRMAX,RELMAX)
C     WRITE (LOUT,99996) ERRMAX,RELMAX
C
C     WRITE (LOUT,99994)
C
C99999 FORMAT (10H  TIME  = ,D20.10)
C99998 FORMAT (23H  SOLUTION/DERIVATIVE: ,/, (4D15.5))
C99997 FORMAT (18H  RESIDUAL NUMBER ,I3,12H HAS A ZERO.,/,
C     +        33H  THIS CORRESPONDS TO ODE NUMBER ,I3)
C99996 FORMAT (37H  MAXIMUM ABSOLUTE/RELATIVE ERRORS = ,2D15.5)
C99995 FORMAT (27H  ERROR IN COMPUTED ROOT = ,D15.5)
C99994 FORMAT (/)
C
C     RETURN
C     END
C
C     SUBROUTINE CHANGE(INDEX,JROOT,INDEXG,NG,N,YROOT,DYROOT,HINIT,
C    +                  TROOT,TINC,ITOL,ABSERR,RELERR,LOUT,
C    +                  RPAR,IPAR)
C
C     THIS TEST SUBROUTINE MAKES PROBLEM CHANGES AT ROOTS OF THE
C     EVENT FUNCTIONS.
C
C     ALL PARAMETERS HAVE THE VALUES OUTPUT FROM THE INTEGRATOR
C     WHEN A ROOT OF AN EVENT FUNCTION WAS LOCATED AT TROOT.
C     IF THIS ROUTINE RESETS INDEX TO A NEGATIVE VALUE, THE
C     DRIVER WILL TERMINATE THE INTEGRATION.
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION JROOT(NG),INDEXG(NG),YROOT(N),DYROOT(N),
C               ABSERR(ITOL),RELERR(ITOL),RPAR(*),IPAR(*)
C
C     NO CHANGES ARE NECESSARY FOR THIS TEST PROBLEM.
C
C     RETURN
C     END
C
C     SUBROUTINE GUSER (N,T,Y,DY,NGUSER,G,RPAR,IPAR)
C
C     INPUT -
C
C     N                    = number of odes
C     T                    = independent variable (time)
C     Y(I), I=1,...,N      = solution at time T
C     DY(I), I=1,...,N     = approximation to derivative at time t
C                            (not exact derivative from DERIVS)
C     NGUSER               = number of event functions
C     RPAR(*),IPAR(*)      = user arrays passed via DKLAG6
C
C
C     THIS TEST SUBROUTINE DEFINES THE RESIDUALS FOR THE
C     OPTIONAL USER EVENT FUNCTIONS,
C     G(I), I=1,...,NGUSER.
C
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION Y(N),DY(N),G(NGUSER),RPAR(*),IPAR(*)
C
C     NO USER ROOTFINDING IS NECESSARY FOR THIS TEST PROBLEM.
C
C     RETURN
C     END
C
C     The following test subroutines are required for this test
C     problem only because SUBROUTINE DPRINT calculates the
C     exact solution and the errors in the computed solution.
C
C     SUBROUTINE TRUSOL(N,T,Y)
C
C     INPUT -
C
C     N  = NUMBER OF EQUATIONS
C     T  = INDEPENDENT VARIABLE (TIME)
C
C     THIS TEST SUBROUTINE DEFINES THE EXACT SOLUTION
C     Y(I), I = 1 ,..., N . (THIS IS A DEMONSTRATION
C     SUBROUTINE CALLED BY DPRINT, NOT ONE REQUIRED
C     BY THE DKLAG6 INTEGRATOR.)
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION Y(N)
C
C     E = EXP(1.0D0)
C     IF (T.LE.E) THEN
C         Y(1) = T
C     ENDIF
C     IF (T.GT.E) THEN
C         Y(1) = EXP(T/E)
C     ENDIF
C
C     IF (T.LE.2.0D0) THEN
C         Y(2) = SQRT(T)
C     ENDIF
C     IF (T.GT.2.0D0) THEN
C         Y(2) = 0.25D0*T + 0.5D0 + (1.0D0-0.5D0*SQRT(2.0D0))*SQRT(T)
C     ENDIF
C
C     RETURN
C     END
C
C     SUBROUTINE YERROR(N,Y,EXACT,ERRMAX,RELMAX)
C
C     THIS TEST SUBROUTINE CALCULATES THE MAXIMUM ERROR IN
C     THE COMPUTED SOLUTION FOR THIS TEST PROBLEM. (THIS
C     IS A DEMONSTRATION SUBROUTINE CALLED BY DPRINT, NOT
C     ONE REQUIRED BY THE DKLAG6 INTEGRATOR.)
C
C     IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     IMPLICIT INTEGER (I-N)
C
C     DIMENSION EXACT(*),Y(*)
C
C     ERRMAX = 0.0D0
C     RELMAX = 0.0D0
C     DO 20 I = 1,N
C         ERRORI = EXACT(I) - Y(I)
C         ERRMAX = MAX(ERRMAX,ABS(ERRORI))
C         IF (Y(I).NE.0.0D0) THEN
C             ERRORI = ERRORI/Y(I)
C             RELMAX = MAX(RELMAX,ABS(ERRORI))
C         ENDIF
C  20 CONTINUE
C
C     RETURN
C     END
C
C     END OF SAMPLE PROGRAM
C
C     NONSTANDARD INTERPOLATION:
C     _________________________
C
C     In some cases it is desirable to interpolate the solution
C     or its derivative for previous values of time. For example,
C     the user may wish in subroutine DPRINT to approximate the
C     solution at previous times other than delay times, for plotting
C     purposes. Similarly, it may be desirable in subroutine GUSER
C     to approximate the solution at previous times, for rootfinding
C     purposes. Since it is not possible for DKLAG6 to anticipate
C     such tasks without requiring a very elaborate data structure,
C     DKLAG6 is not designed to accommodate these tasks directly.
C     However, it is possible to perform these tasks in the following
C     manner. The description below uses a labeled COMMON block to
C     communicate the necessary information to the  calling subroutine.
C     The same communication may be accomplished also using the
C     RPAR and IPAR arrays.
C
C     Declare the logical method flag and the problem type flag.
C
C     LOGICAL EXTRAP, NUTRAL
C
C     Dimension the necessary arrays of length N. (The Y array will
C     usually be a parameter to the subroutine doing the interpolation.
C     The latter four arrays will be local to the subroutine.)
C
C     DIMENSION Y(*), BVAL(*), YTEMP(*), YVAL(*), DYVAL(*), RPAR(*),
C               IPAR(*)
C
C     In the calling program to DKLAG6, include the work arrays
C     in a common block and make this common block accessible to
C     the subroutine that performs the interpolation.
C
C     COMMON /WORKS/ W(*), IW(*), QUEUE(*), LQUEUE
C
C     Declare the subroutines in which the initial solution and
C     derivative for T .le. TINIT are calculated.
C
C     EXTERNAL YINIT, DYINIT
C
C     Supply the following parameters:
C
C     N       = number of differential equations
C
C     NG      = total number of root functions (including those
C               corresponding to IROOT=0 in subroutine INITAL)
C               Note:
C               NG = IW(30) where IW is described below.
C
C     T       = current value of the independent variable
C
C     Y       = solution at time T (T and Y will usually be
C               an input parameter to the subroutine which
C               calls D6INT1 to perform the interpolation.)
C
C     HNEXT   =  1.0d0 if the integration goes left to right
C             = -1.0d0 if the integration goes right to left
C
C     BVAL(I) = desired delay for the I-th equation, I = 1,...,N
C
C     YTEMP   = scratch work array of length N
C
C     YINIT   = external function to calculate the initial
C               delayed solution
C
C     DYINIT  = external function to calculate the initial
C               delayed derivative
C
C     QUEUE   = QUEUE array of length LQUEUE for DKLAG6
C
C     LQUEUE  = length of QUEUE array (=IW(20))
C               Note that LQUEUE may be different from
C               the value you supplied DKLG6C. It will
C               be the same value you supplied to DKLAG6,
C               however.
C
C     IW(*)   = integer work array for DKLAG6
C
C     W       = work array for DKLAG6 (QUEUE, LQUEUE, IW,
C               and W may be communicated to the calling
C               subroutine via COMMON.)
C
C     J14     = 1 + 13*N (J14 is a pointer in the work array
C               where necessary information for DKLAG6 is
C               located.)
C
C     TINIT   = starting time for the problem
C
C     EXTRAP  = logical method flag (same as in INITAL)
C
C     NUTRAL  = logical problem type flag (same as in INITAL)
C
C     RPAR(*), IPAR(*) = user arrays passed via D6INT1
C
C     Perform the desired interpolation.
C
C     CALL D6INT1(N,NG,T,Y,HNEXT,BVAL,YTEMP,YVAL,DYVAL,YINIT,
C    *            DYINIT,QUEUE,LQUEUE,IW,DW(J14),TINIT,EXTRAP,
C    *            INDEX,1,0,RPAR,IPAR)
C
C     The user must be careful not to alter the values of any
C     of the above DKLAG6 parameters.
C
C     The delayed solution at time BVAL(I) will be returned
C     in YVAL(I), I = 1,...,N.
C
C     The derivative of the delayed solution at time BVAL(I)
C     will be returned in DYVAL(I), I = 1,...,N.
C
C     Caution:
C
C     DYVAL(I) will not be defined for any component for which
C     BETA(Y,T) = T.
C
C     If INDEX is not 2 after the return from subroutine D6INT1,
C     the interpolation was not successful. In this case, the
C     contents of YVAL are not reliable approximations.
C
C     Note:
C
C     If it is known that the values of BVAL(*) do not fall outside
C     the interval spanned by the the history queue, a second
C     interpolation subroutine D6INT2, with a simpler calling list,
C     may be used in lieu of D6INT1. The call sequence for D6INT2
C     is as follows (where the parameters have the same meaning as
C     for D6INT1):
C
C     CALL D6INT2(N,NG,BVAL,YWORK,QUEUE,LQUEUE,IW,YLAG,DYLAG,
C     +           HNEXT,DW(J14),INDEX,1,0,EXTRAP)
C
C     D6INT1 is the core subroutine used by DKLAG6. The primary
C     purpose for D6INT2 is to provide the capability to solve
C     a problem, save W, IW, and QUEUE arrays, and then use this
C     information in new YINIT/DYINIT subroutines to define the
C     initial delay solution/derivative information for another
C     problem. It also provides a way to continue the same problem
C     to a new value of TFINAL although this is not recommended
C     due to the expense involved and the need for separate DW, IW,
C     and QUEUE arrays during the continuation.
C
C     D6INT1 and D6INT2 are not intended for casual use.
C
C     Note:
C
C     Refer to the documentation prologue for D6INT2 for a
C     description of the situations in which D6INT2 or D6INT1
C     cannot perform requested interpolations.
C
C     There are actually five user callable interpolation modules
C     D6INT1, D6INT2, D6INT3, D6INT4, and D6INT5, as described
C     below; refer to the documentation prologues for detailed
C     usage requirements.
C
C     Further notes on the various interpolation modules within
C     DKLAG6:
C
C     (The first five modules are user callable.)
C
C     Module   Called By  Calls         Purpose/Comments
C
C     D6INT1    D6DRV2    D6INT6    user callable interpolation
C               D6STP1              of solution history queue
C                                   for nonvanishing delay
C                                   problems (EXTRAP = .FALSE.);
C                                   called by the interval
C                                   oriented driver D6DRV2 and
C                                   the core integrator D6STP1
C
C     D6INT3    D6STP2    D6INT9    user callable interpolation
C                         D6INT7    of solution history queue
C                                   for vanishing delay problems
C                                   (EXTRAP = .TRUE.); called by
C                                   the core integrator D6STP2
C
C     D6INT2      -       D6POLY    user callable simplified
C                                   version of D6INT1 which
C                                   is applicable if all delay
C                                   times are spanned by the
C                                   solution history queue;
C                                   not called by DKLAG6
C
C     D6INT4      -       D6POLY    user callable simplified
C                                   version of D6INT3 which
C                                   is applicable if all delay
C                                   times are spanned by the
C                                   solution history queue;
C                                   not called by DKLAG6
C
C     D6INT5      -       D6POLY    user callable version of
C                                   D6INT4 which may be used
C                                   to interpolate only selected
C                                   solution components; not
C                                   called by DKLAG6
C
C     D6INT8    D6DRV2    D6POLY    interpolate the solution and
C                                   derivative at an output point
C                                   following a successful
C                                   integration step; called only
C                                   by the interval oriented
C                                   driver D6DRV2
C
C     D6INT6    D6INT1    D6POLY    subsidiary module for D6INT1
C
C     D6INT7    D6INT3    D6INT9    subsidiary module for D6INT3
C                         D6POLY
C
C     D6INT9    D6INT3      -       interpolate the solution and
C               D6INT7              derivative using any of the
C               D6HST2              imbedded family methods of
C                                   orders 1, ..., 5; used only
C                                   for vanishing delay problems
C                                   (EXTRAP = .TRUE.)
C
C     D6HST2    D6HST1    D6INT9    subsidiary module for the
C                                   initial stepsize subroutine
C                                   D6HST1
C
C     D6POLY    D6INT2      -       lowest level solution and
C               D6INT4              derivative interpolation
C               D6INT8              module
C               D6INT6
C               D6INT7
C               D6GRT1
C               D6DRV3
C
C     D6POLY    D6INT5      -       version of D6POLY used by
C                                   D6INT5; not called by DKLAG6
C
C     D6WSET    D6POLY      -       subroutine which defines the
C                                   solution and/or derivative
C                                   interpolation coefficients
C                                   for D6POLY
C
C     Notes on the various binary search modules:
C
C     D6SRC1 is the basic search module. The D6SUR* modules are
C     used to find bracketing intervals within the solution
C     history queue and to call D6SRC1. D6SRC2 is used for
C     nonvanishing delay problems (EXTRAP = .FALSE.). D6SRC3 is
C     used for vanishing delay problems (EXTRAP = .TRUE.).
C     D6SRC4 is used by the user callable interpolation modules
C     D6INT4 and D6INT5.
C
C     ERROR MESSAGE SUMMARY:
C     _____________________
C
C     The following is a summary of the error messages generated
C     by DKLAG6. Each error message is preceded by the following
C     message:
C
C         AN ERROR WAS ENCOUNTERED IN THE DKLAG6 SOLVER.   
C         REFER TO THE FOLLOWING MESSAGE ... .  
C
C     Error number = 1
C
C         Message:
C
C         INDEX IS LESS THAN 1 OR GREATER THAN 5.   
C
C         Meaning:
C
C         The program or subroutine which called the core
C         driver D6DRV3 specified an illegal value for
C         the status flag INDEX. This should never occur
C         if you are using the DKLAG6 driver.
C
C         Suggested actions:
C
C         Check to see that you are not inadvertently
C         redefining INDEX when one of your user provided
C         subroutines is called.
C
C     Error number = 2 
C
C         Message:
C
C         NG IS NEGATIVE.   
C
C         Meaning:
C
C         NG is the number of functions for which root
C         finding is desired. NG cannot be less than 0.
C
C         Suggested actions:
C
C         Check the value of NG that is defined in your
C         INITAL subroutine. Also check that NG is not
C         being clobbered in one of your subroutines
C         (e.g., GUSER, DPRINT, CHANGE).
C
C     Error number = 3 
C
C         Message:
C
C         N IS LESS THAN ONE.   
C
C         Meaning:
C
C         N is the number of differential equations. N
C         cannot be less than 1.
C
C         Suggested actions:
C
C         Check the value of N that is defined in your
C         calling program to DKLAG6. Also check that N is
C         not being clobbered in one of your subroutines.
C
C     Error number = 4 
C
C         Message:
C
C         RELERR IS NEGATIVE.   
C
C         Meaning:
C
C         RELERR is the relative error tolerance for the
C         integration. RELERR cannot be less than zero.
C
C         Suggested actions:
C
C         Check the definition of RELERR in your INITAL
C         subroutine.
C
C     Error number = 5 
C
C         Message:
C
C         ABSERR IS NEGATIVE.   
C
C         Meaning:
C
C         ABSERR is the absolute error tolerance for the
C         integration. ABSERR cannot be less than zero.
C
C         Suggested actions:
C
C         Check the definition of ABSERR in your INITAL
C         subroutine.
C
C     Error number = 6 
C
C         Message:
C
C         BOTH ABSERR AND RELERR ARE ZERO.   
C
C         Meaning:
C
C         ABSERR and RELERR are the absolute and relative
C         error tolerances for the integration. Although
C         either one of them can be zero, at least one of
C         them must be positive.
C
C         Suggested actions:
C
C         Check the definition of ABSERR and RELERR
C         in your INITAL subroutine.
C
C     Error number = 7 
C
C         Message:
C
C         NSIG MUST BE AT LEAST ZERO.   
C
C         Meaning:
C
C         NSIG is the number of significant digits to be
C         used in the rootfinding. This error will not
C         occur if the DKLAG6 driver is used.
C
C         Suggested actions:
C
C         If you are using the DKLAG6 driver, this cannot 
C         occur. Otherwise, check the value of NSIG that
C         is being supplied to the D6DRV3 driver.
C
C     Error number = 8 
C
C         Message:
C
C         TFINAL AND TOLD MUST BE DIFFERENT.   
C
C         Meaning:
C
C         You have told the integrator that the beginning
C         and final integration times are the same. They
C         must be different.
C
C         Suggested actions:
C
C         Check the definitions of TOLD and TFINAL in
C         your INITAL subroutine.
C
C     Error number = 9 
C
C         Message:
C
C         THE LENGTH OF THE WORK(W) ARRAY IS TOO SMALL.   
C
C         Meaning:
C
C         The length of the W work array you supplied to
C         DKLAG6 needs to be increased.
C
C         Suggested actions:
C
C         Increase the size of the W array and check that
C         the value of LW that you are supplying to D6DRV2
C         is correct.
C
C     Error number = 10
C
C         Message:
C
C         A COMPONENT OF THE SOLUTION VANISHED MAKING   
C         IT IMPOSSIBLE TO CONTINUE WITH A PURE   
C         RELATIVE ERROR TEST.   
C
C         Meaning:
C
C         You requested pure relative error control by setting
C         ABSERR = 0. At some point the computed solution
C         passed through zero, making it inadvisable to
C         continue the integration. The actual test which
C         generates this message is whether the solution at
C         the latest mesh point is less than 10 units of
C         machine unit roundoff in magnitude or a sign change
C         in the solution has occurred during the last
C         integration step.
C
C         Suggested actions:
C
C         If you know this should not happen, check the
C         coding in your YINIT subroutine. Otherwise,
C         consider using a nonzero value of ABSERR in
C         your INITAL subroutine.
C
C     Error number = 11
C
C         Message:
C
C         THE REQUIRED STEPSIZE IS TOO SMALL. THIS   
C         USUALLY INDICATES A SUDDEN CHANGE IN THE   
C         EQUATIONS OR A VANISHING DELAY PROBLEM.   
C
C         Meaning:
C
C         The integrator encountered difficulties and
C         subsequently was forced to reduce the integration
C         stepsize almost to zero (about 25 times machine
C         unit roundoff). This can occur if one of the
C         differential equations has an abrupt discontinuity
C         at some point. It can also occur if you use the
C         nonvanishing delay option (by defining EXTRAP to
C         be .TRUE.) and a delay vanishes at some point in
C         the integration.
C
C         Suggested actions:
C
C         Check the coding in your user provided subroutines,
C         particularly DERIVS. Consider using the vanishing
C         delay option (EXTRAP = .TRUE.).
C
C     Error number = 12
C
C         Message:
C
C         THE ROOTFINDER WAS NOT SUCCESSFUL.   
C
C         Meaning:
C
C         This occurs if the rootfinder does not converge
C         within the maximum number of iterations. Since
C         this number is set large enough to guarantee
C         convergence to within the tolerance specified by
C         DKLAG6, it indicates your root function has a
C         discontinuity at some point and the machine you
C         are using isn't handling roundoff cleanly.
C
C         Suggested actions:
C
C         Check the coding in your GUSER subroutine. Print
C         the values of time and the residuals to find out
C         which one is causing the problem. Look for a
C         discontinuity in the residuals. Alternatively,
C         you may wish to replace subroutine D6RTOL and
C         require less accuracy in the computed roots.
C
C     Error number = 13
C
C         Message:
C
C         A TERMINAL ERROR OCCURRED IN D6DRV3.   
C         REFER TO THE OTHER PRINTED MESSAGES.   
C
C         Meaning:
C
C         D6DRV3 is the core third level driver for the
C         integration. If it encounters an error or one
C         is encountered in a subroutine it calls, the
C         above message will be generated.
C
C         Suggested actions:
C
C         Take the suggested action(s) associated with
C         the other error messages that are generated.
C
C     Error number = 14
C
C         This error number is inactive in this version
C         of DKLAG6.
C
C     Error number = 15
C
C         This error number is inactive in this version
C         of DKLAG6.
C
C     Error number = 16
C
C         This error number is inactive in this version
C         of DKLAG6.
C
C     Error number = 17
C
C         This error number is inactive in this version
C         of DKLAG6.
C
C     Error number = 18
C
C         This error number is inactive in this version
C         of DKLAG6.
C
C     Error number = 19
C
C         AN ABNORMAL ERROR WAS ENCOUNTERED IN D6DRV2.   
C         REFER TO THE OTHER PRINTED MESSAGES.   
C
C         Meaning:
C         D6DRV2 is a second level driver for the
C         integration. If it encounters an error
C         or one is encountered in a subroutine it
C         calls, the above message will be generated.
C
C         Suggested actions:
C
C         Take the suggested action(s) associated with
C         the other error messages that are generated.
C
C     Error number = 20
C
C         Message:
C
C         THE LENGTH OF THE QUEUE ARRAY IS TOO SMALL.   
C
C         Meaning:
C
C         The length of the QUEUE solution history array
C         you supplied to DKLAG6 needs to be increased.
C
C         Suggested actions:
C
C         If possible, increase the size of the QUEUE array
C         and check that the parameter LQUEUE you are
C         supplying to DKLAG6 is correct. If it is not
C         possible to increase the size of the QUEUE array,
C         consider using a larger error tolerance. Also
C         check that the system delays in your BETA
C         subroutine are being defined correctly. If the
C         integrator is using very small steps, check the
C         coding in your DERIVS subroutine.
C
C     Error number = 21
C
C         Message:
C
C         THE LENGTH OF THE IWORK (IW) ARRAY IS TOO SMALL.   
C
C         Meaning:
C
C         The length of the IW integer work array you
C         supplied to DKLAG6 needs to be increased.
C
C         Suggested actions:
C
C         Increase the size of the IW array and check that
C         the value of LIW that you are supplying to D6DRV2
C         is correct.
C
C     Error number = 22
C
C         Message:
C
C         THE OUTPUT PRINT INCREMENT MUST BE NONZERO.   
C
C         Meaning:
C
C         The output point increment which you requested
C         must be nonzero.
C
C         Suggested actions:
C
C         Check the definition of TINC in your INITAL
C         subroutine.
C
C     Error number = 23
C
C         Message:
C
C         IROOT MUST BE SET TO 0 OR 1 IN INITAL.   
C
C         Meaning:
C
C         IROOT is the flag used to tell DKLAG6 whether
C         or not derivative discontinuity rootfinding
C         is to be performed. It must be 0 (in which case
C         the rootfinding will be performed) or 1 (in
C         which case it will not be performed).
C
C         Suggested actions:
C
C         Check the definition of IROOT in your INITAL
C         subroutine.
C
C     Error number = 24
C
C         Message:
C
C         A SYSTEM DELAY IS BEYOND THE CURRENT TIME.   
C
C         Meaning:
C
C         A system delay is not allowed to exceed the
C         corresponding time if the integration proceeds
C         left to right and is not allowed to be less
C         than the corresponding time if the integration
C         proceeds right to left.
C
C         Suggested actions:
C
C         If you know this should not happen, check the
C         coding in your BETA subroutine. If it is okay,
C         and the delay is state dependent, you might
C         consider using a smaller error tolerance.
C
C     Error number = 25
C
C         Message:
C
C         THE INTEGRATION WAS TERMINATED BY THE USER.
C
C         Meaning:
C
C         You chose to terminate the integration when
C         one of your subroutines was called.
C
C         Suggested actions:
C
C         If you intended to terminate the integration,
C         you know why you did this. If you did not
C         intend to terminate the integration, check
C         the coding in your subroutines.
C
C     Error number = 26
C
C         Message:
C
C         THE INTEGRATION WAS TERMINATED TO AVOID  
C         EXTRAPOLATING PAST THE QUEUE.   
C
C         Meaning:
C
C         A delay time fell outside the solution history
C         queue (before the oldest point in the queue).
C         This indicates the size of the queue is not
C         large enough.
C
C         Suggested actions:
C
C         See the actions suggested for error number = 20.
C
C     Error number = 27
C
C         Message:
C
C         THE INTEGRATION WAS TERMINATED TO AVOID  
C         EXTRAPOLATING BEYOND THE QUEUE.   
C
C         Meaning:
C
C         A delay time fell outside the solution history
C         queue (after the most recent point in the queue).
C         This indicates the size of the queue is not
C         large enough or the stepsize is now larger than
C         one of the system delays. If the non-vanishing
C         delay option is used (EXTRAP = .FALSE.), the step
C         integration routine will attempt to circumvent
C         the problem by reducing the integration stepsize.
C         The problem will not occur if the vanishing
C         delay option is used (EXTRAP = .TRUE.).
C
C         Suggested actions:
C
C         Consider increasing the size of the solution
C         history QUEUE. Also consider using the vanishing
C         delay option (EXTRAP = .TRUE.).
C
C     Error number = 28
C
C         Message:
C
C         THE SOLVER CANNOT START. IN D6INT1  THE DELAY   
C         TIME DOES NOT FALL IN THE INITIAL INTERVAL.   
C
C         Meaning:
C
C         You are using the non-vanishing delay option
C         (EXTRAP = .FALSE.) and a system delay is
C         greater than the initial time on the first
C         integration step (e.g., you might be solving
C         a problem like y'(t) = y(t/2)) before any
C         interpolation information is available.
C
C         Suggested actions:
C
C         Consider using the vanishing delay option
C         (EXTRAP = .TRUE.).
C
C     Error number = 29
C
C         Message:
C
C         THE STEPSIZE WAS HALVED TEN TIMES IN D6STP2   
C         DUE TO ILLEGAL DELAYS. THE INTEGRATION WILL   
C         BE TERMINATED.   
C
C         Meaning:
C
C         You are using the non-vanishing delay option.
C         When the step integrator encountered a system
C         delay that fell beyond the most recent point
C         in the solution history queue, it attempted to
C         continue the integration by bisecting the
C         integration stepsize several (ten) times but
C         was unsuccessful. This usually indicates a
C         system delay vanished at some point.
C
C         Suggested actions:
C
C         Consider using the vanishing delay option
C         (EXTRAP = .TRUE.).
C
C     Error number = 30
C
C         Message:
C
C         SUBROUTINE CHANGE CANNOT ALTER N.   
C
C         Meaning:
C
C         When the CHANGE subroutine you supplied was
C         called after a zero of a root function was
C         located, it attempted to change N, the number
C         of differential equations. This is not allowed
C         in this version of DKLAG6.
C
C         Suggested actions:
C
C         Check the definition of N in your CHANGE
C         subroutine.
C
C     Error number = 31
C
C         Message:
C
C         SUBROUTINE CHANGE CANNOT ALTER NG.   
C
C         Meaning:
C
C         When the CHANGE subroutine you supplied was
C         called after a zero of a root function was
C         located, it attempted to change NG, the number
C         of root functions. This is not allowed in this
C         version of DKLAG6.
C
C         Suggested actions:
C
C         Check the definition of NG in your CHANGE
C         subroutine.
C
C     Error number = 32
C
C         Message:
C
C         A TERMINAL ERROR OCCURRED IN D6HST2.   
C
C         INACTIVE IN THIS VERSION.
C
C         Meaning:
C
C         D6HST2 is a self-starting Runge-Kutta subroutine
C         used for the first integration step for vanishing
C         delay problems (EXTRAP = .TRUE.). It encountered
C         a system delay which was beyond the corresponding
C         time.
C
C         Suggested actions:
C
C         If you know this should not happen, check the
C         coding in your BETA subroutine. If it is okay,
C         and the delay is state dependent, consider
C         using a smaller error tolerance.
C
C     Error number = 33
C
C         Message:
C
C         A - B IS EQUAL TO ZERO IN D6HST1.   
C
C         Meaning:
C
C         D6HST1 is the subroutine which calculates the
C         initial stepsize to be attempted. It requires
C         A, the initial time, and B, a subsequent value
C         of time that is used to limit the stepsize it
C         calculates. A and B cannot be the same. This
C         should not occur if you are using the DKLAG6
C         driver and have defined parameters correctly
C         in subroutine INITAL.
C
C         Suggested actions:
C
C         Check the definitions of TOLD and TFINAL in
C         your INITAL subroutine.
C
C     Error number = 34
C
C         Message:
C
C         A SYSTEM DELAY TIME IS BEYOND THE    
C         CURRENT TIME IN SUBROUTINE D6HST1.   
C
C         Meaning:
C
C         D6HST1 is the subroutine which calculates the
C         initial stepsize to be attempted. It encountered
C         a system delay which was beyond the corresponding
C         time.
C
C         Suggested actions:
C
C         If you know this should not happen, check the
C         coding in your BETA subroutine. If it is okay,
C         and the delay is state dependent, consider
C         using a smaller error tolerance.
C
C     Error number = 35
C
C         Message:
C
C         THERE IS NOT ENOUGH INFORMATION IN THE   
C         SOLUTION HISTORY QUEUE TO INTERPOLATE FOR    
C         THE VALUE OF T REQUESTED WHEN D6INT2 WAS   
C         LAST CALLED.   
C
C         Meaning:
C
C         You have called the interpolation subroutine D6INT2
C         and requested that it interpolate for a value of time
C         that is not spanned by the contents of the solution
C         history queue. This usually occurs if you solve one
C         problem and save the queue information for use in the
C         solution of a second problem. Probably what has happened
C         is that during the first solution, it was not possible
C         to save all of the solution information due to the size
C         of the queue you provided. (A circular queue is used and
C         the oldest information is over-written by more recent
C         information when the queue is full.)
C
C         Suggested actions:
C
C         Check the values in BVAL(*) you are supplying to D6INT2.
C         If they fall in the range spanned by the first problem,
C         consider increasing the size of the QUEUE solution history
C         array. Also consider using a larger error tolerance.
C
C     Error number = 36 
C
C         Message:
C
C         YOU MUST DEFINE ITOL TO BE EITHER 1 OR N
C         IN SUBROUTINE INITAL.
C
C         Meaning:
C
C         ITOL is used to tell the solver whether the same
C         absolute and relative error tolerances should be
C         used for each differential equation (ITOL = 1) or
C         if possibly different tolerances should be used
C         for different components.
C
C         Suggested actions:
C
C         Define ITOL to be 1 or N. Be sure to define ABSERR(*)
C         and RELERR(*) also.
C
C     Error number = 37 
C
C         Message:
C
C         YOU HAVE CHANGED THE VALUE OF ITOL TO A VALUE
C         OTHER THAN 1 OR N IN YOUR CHANGE SUBROUTINE.
C
C         Meaning:
C
C         See the description for error number 37.
C
C         Suggested actions:
C
C         Define ITOL to be 1 or N. Be sure to define ABSERR(*)
C         and RELERR(*) also.
C
C     Error number = 38 
C
C         Message:
C
C         YOU HAVE SPECIFIED A NONZERO RELATIVE ERROR WHICH
C         IS SMALLER THAN THE FLOOR VALUE. IT WILL BE
C         REPLACED WITH THE FLOOR VALUE DEFINED IN
C         SUBROUTINE D6RERS. EXECUTION WILL CONTINUE.
C
C         Meaning:
C
C         You have probably specified a relative error
C         tolerance which is too small for the machine
C         you are using.
C
C         Suggested actions:
C
C         Consider using a larger relative error tolerance.
C
C     Error number = 39
C
C         Message:
C
C         YOU CHANGED THE STEPSIZE TO 0.0D0 WHEN YOUR SUBROUTINE
C         CHANGE WAS CALLED. THE INTEGRATION WILL BE TERMINATED.
C
C         Meaning:
C
C         Although you may change the stepsize when CHANGE is
C         called, you must give it a nonzero value if you do.
C
C         Suggested actions:
C
C         Change the stepsize to a nonzero value or allow the
C         integrator to continue with the stepsize it passes
C         to CHANGE.
C
C     Error number = 40
C
C         Message:
C
C         THE STEPSIZE WAS HALVED THIRTY TIMES IN D6STP2
C         DUE TO CONVERGENCE FAILURES. THE INTEGRATION
C         WILL BE TERMINATED.
C
C         Meaning:
C
C         A predictor corrector iteration is used in subroutine
C         D6STP2 for vanishing delay problems (EXTRAP = .TRUE.).
C         The code could not obtain convergence of the iteration
C         even after halving the stepsize 30 times.
C
C         Suggested actions:
C
C         If this message occurs, there is a derivative discontinuity
C         which the code cannot get past. Please contact the author.
C
C     Error number = 41
C
C         Message:
C
C         AFTER A ZERO OF AN EVENT FUNCTION WAS LOCATED AND
C         THE INTEGRATION STEP WAS REPEATED UP TO THE ZERO,
C         THE CORRECTOR ITERATION IN D6STP2 DID NOT CONVERGE.
C
C         Meaning:
C
C         A predictor corrector iteration is used in subroutine
C         D6STP2 for vanishing delay problems (EXTRAP = .TRUE.).
C         If the code locates a zero following a successful
C         integration step it then repeats the step up to the
C         zero. The iteration did not converge when the step
C         was repeated using the smaller stepsize.
C
C         Suggested actions:
C
C         It should not encounter non-convergence of the iteration
C         since the iteration just converged when a larger step
C         size was used. If this message occurs, please contact
C         the author.
C
C     Error number = 42
C
C         Message:
C
C         AN ADVANCED DELAY WILL BE ALLOWED IN D6BETA.
C
C         Meaning:
C
C         To handle state dependent vanishing delay problems, it
C         is sometimes necessary to allow the delay to slightly
C         exceed the corresponding time.
C
C         Suggested actions:
C
C         If this is not acceptable and it is desirable to treat
C         this condition as a terminal error, remove the statement
C         from subroutine D6BETA which allows advanced delays.
C
C***REFERENCES  CORWIN, S. P., AND THOMPSON, S., DKLA6: SOLUTION
C                 OF SYSTEMS OF FUNCTIONAL DIFFERENTIAL EQUATIONS
C                 WITH STATE DEPENDENT DELAYS, TR-96-001,
C                 COMPUTER SCIENCE TECHNICAL REPORT SERIES,
C                 RADFORD UNIVERSITY, RADFORD, VIRGINIA, 24142.
C
C                NEVES, K. W., AND THOMPSON, S., SOFTWARE FOR THE
C                 NUMERICAL SOLUTION OF SYSTEMS OF FUNCTIONAL 
C                 DIFFERENTIAL EQUATIONS WITH STATE DEPENDENT
C                 DELAYS, APPLIED NUMERICAL MATHEMATICS,
C                 VOLUME 9, 1992, PP. 385-401.
C
C               NEVES, K. W., AND THOMPSON, S., DRKLAG: SOLUTION
C                 OF SYSTEMS OF FUNCTIONAL DIFFERENTIAL EQUATIONS
C                 WITH STATE DEPENDENT DELAYS, TR-92-003,
C                 COMPUTER SCIENCE TECHNICAL REPORT SERIES,
C                 RADFORD UNIVERSITY, RADFORD, VIRGINIA, 24142.
C
C               THOMPSON, S., IMPLEMENTATION AND EVALUATION OF
C                 SEVERAL CONTINUOUSLY IMBEDDED METHODS OF
C                 SARAFYAN, ORNL/TM-10434, SEPTEMBER 1987.
C
C***ROUTINES CALLED  D6DRV2, D6MESG, J4SAVE
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  DKLAG6
C
C                     PRIMARY FLOW OF CONTROL
C
C                               *
C                               *
C                               *
C     DKLAG6 (first level driver; allocates arrays)
C                               *
C                               *
C                               *
C     D6DRV2 (second level driver; resembles a typical driver
C             program for an ode solver; initializes problem,
C             calls the solver for solution at integration
C             mesh points, interpolates the solution at
C             output points, untangles the rootfinding and
C             non-rootfinding solver returns, calls the user's
C             routine to process the solution)
C                               *
C                               *
C                               *
C     D6DRV3 (third level driver; controls the step integration,
C             rootfinding, error control, stepsize selection,
C             management of the solution history queue)
C         *              *              *                *
C         *              *              *                *
C         *              *              *                *
C       D6STP1,3       D6DRV3       D6ERR1/D6HST3      D6QUE1
C
C        step      driver for   integration error  update history
C     integrator   rootfinder   control/stepsize       queue
C                                   selection
C
      IMPLICIT NONE
C
      INTEGER LIW,LW,LQUEUE,IFLAG,LIN,LOUT,IW,J4SAVE,IPAR,
     +        INEED,J1,J2,J3,J4,J5,J6,J7,J8,J9,J10,J11,J12,
     +        J13,J14,K1,K2,K3,K4,LWORK,LIWORK,N,IDUM
      DOUBLE PRECISION W,QUEUE,D6OUTP,RPAR,RDUM
C
      DIMENSION IW(LIW),IPAR(*)
      DIMENSION W(LW),QUEUE(LQUEUE),RPAR(*)
C
C     DECLARE THE OUTPUT INCREMENT FUNCTION.
C
      EXTERNAL D6OUTP
C
C     DECLARE THE USER SUPPLIED PROBLEM DEPENDENT SUBROUTINES.
C
      EXTERNAL INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,DYINIT
C
C     LWORK IS THE LENGTH OF THE WORK ARRAY (FOR D6DRV2; NEED
C     13*N MORE IF GO THROUGH D6KLAG).
C     LWORK MUST BE AT LEAST 21*N + 1.
C
C     LQUEUE IS THE LENGTH OF THE QUEUE HISTORY ARRAY.
C     LQUEUE SHOULD BE APPROXIMATELY
C         (12*(N+1)) * (TFINAL-TINIT)/HAVG ,
C     WHERE TINIT AND TFINAL ARE THE BEGINNING AND FINAL
C     INTEGRATION TIMES AND HAVG IS THE AVERAGE STEPSIZE
C     THAT WILL BE USED BY THE INTEGRATOR.
C
C***FIRST EXECUTABLE STATEMENT  DKLAG6
C
C     RESET THE OUTPUT UNIT NUMBER TO THE ONE SPECIFIED
C     BY THE USER.
C
      IDUM = J4SAVE(3,LOUT,.TRUE.)
C
      CALL D6STAT(N, 1, 0.0D0, RPAR, IPAR, IDUM, RDUM)
C
C     THE DW ARRAY IS PARTITIONED HERE AND IN D6PNTR (FOR
C     D6DRV2) AS FOLLOWS:
C
C     IN D6KLAG:
C                                      STARTS IN DW:
C
C     YOLD       : J1  = 1              = 1
C     DYOLD      : J2  = J1  + N        = 1 +    N
C     YNEW       : J3  = J2  + N        = 1 +  2*N
C     DYNEW      : J4  = J3  + N        = 1 +  3*N
C     YROOT      : J5  = J4  + N        = 1 +  4*N
C     DYROOT     : J6  = J5  + N        = 1 +  5*N
C     YOUT       : J7  = J6  + N        = 1 +  6*N
C     DYOUT      : J8  = J7  + N        = 1 +  7*N
C     YLAG       : J9  = J8  + N        = 1 +  8*N
C     DYLAG      : J10 = J9  + N        = 1 +  9*N
C     BVAL       : J11 = J10 + N        = 1 + 10*N
C     ABSERR     : J12 = J11 + N        = 1 + 11*N
C     RELERR     : J13 = J12 + N        = 1 + 12*N
C     WORK       : J14 = J13 + N        = 1 + 13*N
C
C     IN D6DRV2 WORK(1) = DW(J14) = DW(1+13*N):
C
C                  STARTS IN WORK:     STARTS IN DW:
C
C     K0 , KPRED0: I1    = 1            = 1 + 13*N
C     K1 , KPRED1: I2    = I1  + 2*N    = 1 + 15*N
C     K2 , KPRED2: I3    = I2  + 2*N    = 1 + 17*N
C     K3 , KPRED3: I4    = I3  + 2*N    = 1 + 19*N
C     K4 , KPRED4: I5    = I4  + 2*N    = 1 + 21*N
C     K5 , KPRED5: I6    = I5  + 2*N    = 1 + 23*N
C     K6 , KPRED6: I7    = I6  + 2*N    = 1 + 25*N
C     K7 , KPRED7: IK7   = I7  + 2*N    = 1 + 27*N
C     K8 , KPRED8: IK8   = IK7 + 2*N    = 1 + 29*N
C     K9 , KPRED9: IK9   = IK8 + 2*N    = 1 + 31*N
C     R          : I8    = IK9 + 2*N    = 1 + 33*N
C     TINIT      : I9-1  = I8  + N      = 1 + 34*N
C     G(TOLD)    : I9    = I8  + N + 1  = 2 + 34*N
C     G(TNEW)    : I10   = I9  + NG     = 2 + 34*N +   NG
C     G(TROOT)   : I11   = I10 + NG     = 2 + 34*N + 2*NG
C     GA         : I12   = I11 + NG     = 2 + 34*N + 3*NG
C     GB         : I13   = I12 + NG     = 2 + 34*N + 4*NG
C     G2(TROOT)  : I14   = I13 + NG     = 2 + 34*N + 5*NG
C     TG         : I15   = I14 + NG     = 2 + 34*N + 6*NG
C     TG ENDS IN :                        1 + 34*N + 7*NG
C
C     DEFINE THE LOCAL ARRAY POINTERS INTO THE WORK ARRAY.
C
C     J1   -  YOLD
C     J2   -  DYOLD
C     J3   -  YNEW
C     J4   -  DYNEW
C     J5   -  YROOT
C     J6   -  DYROOT
C     J7   -  YOUT
C     J8   -  DYOUT
C     J9   -  YLAG
C     J10  -  DYLAG
C     J11  -  BVAL
C     J12  -  ABSERR
C     J13  -  RELERR
C     J14  -  WORK
C
      J1 = 1
      J2 = J1 + N
      J3 = J2 + N
      J4 = J3 + N
      J5 = J4 + N
      J6 = J5 + N
      J7 = J6 + N
      J8 = J7 + N
      J9 = J8 + N
      J10 = J9 + N
      J11 = J10 + N
      J12 = J11 + N
      J13 = J12 + N
      J14 = J13 + N
      LWORK = LW - (J14-1)
C
      K1 = 1
      K2 = K1 + 34 
      LIWORK = LIW - 34 
C     THE REMAINING INTEGER STORAGE WILL BE SPLIT EVENLY
C     BETWEEN THE JROOT AND INDEXG ARRAYS.
      LIWORK = LIWORK/3
      K3 = K2 + LIWORK
      K4 = K3 + LIWORK
C
C     PRELIMINARY CHECK FOR ILLEGAL INPUT ERRORS.
C
      IFLAG = 0
C
      IF (N.LT.1) THEN
          CALL D6MESG(3,'DKLAG6')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      INEED = 12*(N+1)
      IF (INEED.GT.LQUEUE) THEN
          CALL D6MESG(20,'DKLAG6')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LIWORK.LT.0 .OR. LIW .LT. 35) THEN
          CALL D6MESG(21,'DKLAG6')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LWORK.LT.21*N+1) THEN
          CALL D6MESG(9,'DKLAG6')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      CALL D6DRV2(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,
     +            DYINIT,N,W(J1),W(J2),W(J3),W(J4),W(J5),W(J6),
     +            W(J7),W(J8),W(J9),W(J10),W(J11),W(J12),W(J13),
     +            W(J14),QUEUE,IW(K1),IW(K2),IW(K3),IW(K4),LWORK,
     +            LIWORK,LQUEUE,IFLAG,D6OUTP,LIN,LOUT,RPAR,IPAR)
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6DRV2
      SUBROUTINE D6DRV2(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,
     +                  YINIT,DYINIT,N,YOLD,DYOLD,YNEW,DYNEW,
     +                  YROOT,DYROOT,YOUT,DYOUT,YLAG,DYLAG,BVAL,
     +                  ABSERR,RELERR,WORK,QUEUE,IPOINT,JROOT,
     +                  INDEXG,LEVEL,LWORK,LIWORK,LQUEUE,IFLAG,
     +                  D6OUTP,LIN,LOUT,RPAR,IPAR)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6DRV2
C***SUBSIDIARY
C***PURPOSE  The function of subroutine D6DRV2 is to solve N
C            ordinary differential equations with state dependent
C            delays. D6DRV2 is applicable to the integration of
C            systems of differential equations of the form
C
C            dY(I)/dT = F(T, Y(I), Y(BETA(T,Y)), Y'(BETA(T,Y)))
C
C            It is a second level interval oriented driver called 
C            by subroutine DKLAG6.
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
C     REFER TO THE PROLOGUE FOR DKLAG6 FOR DESCRIPTIONS OF
C     THE FIRST EIGHT PARAMETERS FOR THIS SUBROUTINE.
C     THESE ARE THE PROBLEM DEPENDENT SUBROUTINES REQUIRED
C     BY D6DRV2.
C
C     N = NUMBER OF ODES
C
C     LOCAL ARRAYS OF LENGTH N:
C
C     YOLD    - OLD SOLUTION AT TIME TOLD
C
C     DYOLD   - OLD DERIVATIVE AT TIME TOLD
C
C     YNEW    - NEW SOLUTION AT TIME TNEW = TOLD + H
C
C     DYNEW   - NEW DERIVATIVE AT TIME TNEW = TOLD + H
C
C     YROOT   - SOLUTION AT ROOT OF EVENT FUNCTION
C
C     DYROOT  - DERIVATIVE AT ROOT OF EVENT FUNCTION
C
C     YOUT    - SOLUTION AT OUTPUT POINT
C
C     DYOUT   - DERIVATIVE AT OUTPUT POINT
C
C     YLAG    - SOLUTION AT DELAY TIMES
C
C     DYLAG   - DERIVATIVE AT DELAY TIMES
C
C     BVAL    - DELAY TIMES
C
C     ABSERR  - ABSOLUTE ERROR TOLERANCES
C
C     RELERR  - RELATIVE ERROR TOLERANCES
C
C     WORK    - WORK ARRAY OF LENGTH LWORK
C
C     QUEUE   - SOLUTION HISTORY ARRAY OF LENGTH LQUEUE
C
C     IPOINT  - INTEGER POINTER ARRAY OF LENGTH AT
C               LEAST 35 
C
C     JROOT   - INTEGER POINTER ARRAY OF LENGTH LIWORK
C
C     INDEXG  - INTEGER POINTER ARRAY OF LENGTH LIWORK
C
C     LEVEL   - INTEGER POINTER ARRAY OF LENGTH LIWORK
C
C     LWORK   - LENGTH OF THE WORK ARRAY. LWORK MUST BE
C               AT LEAST 21*N + 1.
C
C     LIWORK  - LENGTH OF THE JROOT AND INDEXG ARRAYS.
C               LIWORK MUST BE AT LEAST AS LARGE AS NG
C               WHERE NG IS THE LARGEST NUMBER OF ROOT
C               FUNCTIONS THAT WILL BE USED AT ANY TIME
C               (SEE DKLAG6 PROLOGUE FOR A DISCUSSION
C               OF NG.)
C
C     LQUEUE  - LENGTH OF THE QUEUE ARRAY. SEE DKLAG6
C               PROLOGUE FOR A DESCRIPTION OF LQUEUE.
C
C     IFLAG   - ERROR FLAG
C
C               = 0 INDICATES SUCCESSFUL RETURN
C               = 1 INDICATES UNSUCCESSFUL RETURN
C
C     LIN     - NUMBER FOR INPUT UNIT
C
C     LOUT    - NUMBER FOR OUTPUT UNIT
C
C     RPAR(*) AND IPAR(*) ARE USER ARRAYS PASSED VIA DKLAG6.
C
C***END PROLOGUE  D6DRV2
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  DCOPY, D6BETA, D6DRV3, D6QUE2, D6GRT3, D6GRT4,
C                    D6HST1, D6INT8, D6OUTP, D6PNTR, D6RERS, D6MESG,
C                    D6INT1, D1MACH, D6RTOL, D6ESTF, D6DISC
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6DRV2
C
      IMPLICIT NONE
C
      INTEGER IFLAG,IHFLAG,LIWORK,LWORK,LIN,LOUT,LQUEUE,MORDER,N,
     +        INDEXG,IPOINT,JROOT,IPAR,IDCOMP,JCOMP,IDISC,IWANT,J,
     +        IDIR,IER,IERRUS,IFIRST,IFOUND,INX1,INDEX,INEED,IOK,
     +        IORDER,IQUEUE,ITYPE,I,I1,I2,I3,I4,I5,I6,I7,I8,I9,
     +        I10,I11,I12,I13,I14,I15,I9M1,I10M1,I11M1,JNDEX,JQUEUE,
     +        KQUEUE,KROOT,NFEVAL,NG,NGEVAL,NSIG,NGUSER,NGTEMP,
     +        NTEMP,ITOL,ICOMP,NGEMAX,NGOLD,NDISC,IG,IDUM,IK7,IK8,
     +        IK9,IPAIR,OFFRT,RSTRT,AUTORF,NGUSP1,LEVEL,LEVMAX,ANCESL
      DOUBLE PRECISION BVAL,DYLAG,DYNEW,DYOLD,DYOUT,DYROOT,QUEUE,YLAG,
     +                 YNEW,YOLD,YOUT,YROOT,WORK,RPAR,TDISC,
     +                 ABSERR,D1MACH,HDUMMY,HINIT,HINITM,HLEFT,HMAX,
     +                 HNEXT,RATIO,RELERR,TFINAL,TINC,TINIT,TNEW,
     +                 TOLD,TOUT,TROOT,TVAL,U,U13,U26,BIG,SMALL,
     +                 D6OUTP,TEMP,RER,U10,RDUM,U65,TROOT2
      LOGICAL EXTRAP,NUTRAL
C     LOGICAL VANISH
C
      DIMENSION YOLD(*),DYOLD(*),YNEW(*),DYNEW(*),YROOT(*),DYROOT(*),
     +          WORK(*),JROOT(*),YOUT(*),DYOUT(*),IPOINT(*),YLAG(*),
     +          BVAL(*),INDEXG(*),QUEUE(*),DYLAG(*),ABSERR(*),
     +          RELERR(*),RPAR(*),IPAR(*),LEVEL(*)
C
C     DECLARE THE USER SUPPLIED PROBLEM DEPENDENT SUBROUTINES.
C
      EXTERNAL INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,DYINIT
C
C     DECLARE THE OUTPUT INCREMENT FUNCTION.
C
      EXTERNAL D6OUTP
C
C     DECLARE THE SUBROUTINE IN WHICH THE ROOT FUNCTION
C     RESIDUALS WILL BE EVALUATED.
C
      EXTERNAL D6GRT3
C
C     DEFINE THE ROUNDOFF FACTOR FOR THE TERMINATION
C     CRITERIA. THE INTEGRATION WILL BE TERMINATED
C     WHEN IT IS WITHIN U26 OF THE FINAL INTEGRATION
C     TIME. ALSO DEFINE THE FLOOR VALUE RER FOR THE
C     RELATIVE ERROR TOLERANCES.
C
C***FIRST EXECUTABLE STATEMENT  D6DRV2
C
      U = D1MACH(4)
      U10 = 10.0D0*U
      U13 = 13.0D0*U
      U26 = 26.0D0*U
      U65 = 65.0D0*U
      CALL D6RERS(RER)
C
C     INITIALIZE THE PROBLEM.
C
      EXTRAP = .FALSE.
      NUTRAL = .FALSE.
      CALL INITAL(LOUT,LIN,N,NGUSER,TOLD,TFINAL,TINC,YOLD,AUTORF,
     +            HINIT,HMAX,ITOL,ABSERR,RELERR,EXTRAP,NUTRAL,
     +            RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
C
C     AVOID PROBLEMS WITH THE APPROXIMATION OF DYLAG(I) FOR
C     NEUTRAL PROBLEMS WHEN BVAL(I) = T.
C
      IF (NUTRAL) EXTRAP = .TRUE.
C
      TINC = D6OUTP(TOLD,TINC)
C
      TINIT = TOLD
C
      IQUEUE = 0
C
C     CHECK FOR SOME COMMON ERRORS. ALSO RESET THE RELATIVE
C     ERROR TOLERANCES TO THE FLOOR VALUE IF NECESSARY.
C
      IF (ITOL.NE.1 .AND. ITOL.NE.N) THEN
         CALL D6MESG(36,'D6DRV2')
         GO TO 8995
      ENDIF
      IF (N.LT.1) THEN
          CALL D6MESG(3,'D6DRV2')
          GO TO 8995
      ENDIF
      IF (TFINAL.EQ.TOLD) THEN
          CALL D6MESG(8,'D6DRV2')
          GO TO 8995
      ENDIF
      DO 20 I = 1 , N
         IF (I.GT.ITOL) GO TO  20
         IF (RELERR(I).NE.0.0D0 .AND. RELERR(I).LT.RER) THEN
            RELERR(I) = RER
            CALL D6MESG(38,'D6DRV2')
         ENDIF
         IF (ABSERR(I).LT.0.0D0) THEN
             CALL D6MESG(5,'D6DRV2')
             GO TO 8995
         ENDIF
         IF (RELERR(I).LT.0.0D0) THEN
             CALL D6MESG(4,'D6DRV2')
             GO TO 8995
         ENDIF
         IF ((ABSERR(I).EQ.0.0D0).AND.(RELERR(I).EQ.0.0D0)) THEN
             CALL D6MESG(6,'D6DRV2')
             GO TO 8995
         ENDIF
   20 CONTINUE
      IF (TINC.EQ.0.0D0) THEN
          CALL D6MESG(22,'D6DRV2')
          GO TO 8995
      ENDIF
      IF ((AUTORF.LT.0) .OR. (AUTORF.GT.1)) THEN
          CALL D6MESG(23,'D6DRV2')
          GO TO 8995
      ENDIF
C
C     START WITH EITHER N+NGUSER OR NGUSER ROOT FUNCTIONS UNLESS
C     THE USER WISHES TO INPUT ADDITIONAL DISCONTINUITY INFORMATION.
C
      IF (NGUSER.LT.0) NGUSER = 0
      NG = 0
      IF (AUTORF.EQ.0) NG = N
      NG = NG + NGUSER
      NGOLD = NG
      NDISC = 0
      IF (AUTORF.EQ. 0) THEN
C        DETERMINE THE NUMBER OF ADDITIONAL DISCONTINUITY ROOT
C        FUNCTIONS THE USER WISHES TO SPECIFY.
         DO 40 I = 1 , N
            IWANT = 1
            IDCOMP = I
            JCOMP = 0
            IDISC = 0
            TDISC = TOLD
            CALL D6DISC(IWANT,IDCOMP,IDISC,TDISC,JCOMP,RPAR,IPAR)
            IF (IDISC .GT. 0) THEN
               NDISC = NDISC + IDISC
            ENDIF
   40   CONTINUE
      ENDIF
      NG = NG + NDISC
      IPOINT(30) = NG
C
      IF ((NG.GT.0) .AND. (LIWORK.LT.NG)) THEN
          CALL D6MESG(21,'D6DRV2')
          GO TO 8995
      ENDIF
C
C     FORCE AN INTEGRATION RESTART AT ALL ROOTS (INCLUDING ROOTS
C     OF THE USER PROVIDED EVENT FUNCTIONS).
C
      RSTRT = 0
      IPOINT(29) = RSTRT
C
C     INITIALIZE THE INDEXG, JROOT, G-WORK ARRAYS, IF
C     ROOTFINDING IS TO BE DONE.
C
      IF (NG.GT.0) THEN
          CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,1)
          CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,2)
          INX1 = I9 + 6*NG - 1
          DO 60 IG = 1,NGOLD
              INDEXG(IG) = 0
              IF (IG.GT.NGUSER) INDEXG(IG) = IG - NGUSER
C             INITIALIZE TG(*) = TOLD
C             TG STARTS IN GWORK(6NG+1) WHERE GWORK(1)
C             IS THE SAME AS WORK(I9)
              WORK(INX1+IG) = TOLD
              JROOT(IG) = 0
   60     CONTINUE
C         ALLOW THE USER TO INPUT DISCONTINUITY INFORMATION FOR
C         THE INITIAL FUNCTION IF DISCONTINUITY ROOTFINDING IS
C         BEING USED.
          IF (NDISC.GT.0) THEN
             IG = NGOLD
             DO 100 I = 1 , N
                IWANT = 1
                IDCOMP = I
                JCOMP = 0
                IDISC = 0
                TDISC = TOLD
                CALL D6DISC(IWANT,IDCOMP,IDISC,TDISC,JCOMP,RPAR,IPAR)
                IF (IDISC .GT. 0) THEN
                   DO 80 J = 1 , IDISC
                      IG = IG + 1
                      IWANT = 2
                      IDCOMP = I
                      JCOMP = J
                      TDISC = TOLD
                      CALL D6DISC(IWANT,IDCOMP,IDISC,TDISC,JCOMP,
     +                            RPAR,IPAR)
                      INDEXG(IG) = IDCOMP
C                     TG STARTS IN GWORK(6NG+1) WHERE GWORK(1)
C                     IS THE SAME AS WORK(I9)
                      WORK(INX1+IG) = TDISC
                      JROOT(IG) = 0
   80              CONTINUE
                ENDIF
  100        CONTINUE
          ENDIF
      ENDIF
C
C     INITIALIZE THE LEVEL TRACKING ARRAY.
C
      CALL D6LEVL (LEVMAX,NUTRAL)
      LEVMAX = MAX(LEVMAX,0)
      IF (LEVMAX .GT. 0 .AND. NG .GT. 0) THEN
         IF (NGUSER .GT. 0) THEN
            DO 105 I = 1 , NGUSER
               LEVEL(I) = -1
  105       CONTINUE
         ENDIF
         IF (NG .GT. NGUSER) THEN
            NGUSP1 = NGUSER + 1
            DO 110 I = NGUSP1, NG
               LEVEL(I) = 0
  110       CONTINUE
         ENDIF
      ENDIF
C
C     CALCULATE THE INITIAL DELAYS.
C
      CALL BETA(N,TOLD,YOLD,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
C
C     CHECK FOR ILLEGAL DELAYS.
C
      HDUMMY = TFINAL - TOLD
      CALL D6BETA(N,TOLD,HDUMMY,BVAL,IOK)
C
      IF (IOK.EQ.1) THEN
          INDEX = -11
          GO TO 8995
      ENDIF
C
C     GIVE TINC THE CORRECT SIGN.
C
      TINC = SIGN(ABS(TINC),TFINAL-TOLD)
C
C     CALCULATE THE INITIAL DERIVATIVES.
C
      DO 120 I = 1,N
          TVAL = BVAL(I)
          CALL YINIT(N,TVAL,YLAG(I),I,RPAR,IPAR)
          IF (N.LE.0) GO TO 8985
          CALL DYINIT(N,TVAL,DYLAG(I),I,RPAR,IPAR)
          IF (N.LE.0) GO TO 8985
  120 CONTINUE
      CALL DERIVS(N,TOLD,YOLD,YLAG,DYLAG,DYOLD,RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
C
C     FINISH THE INITIALIZATION.
C
      TINC = D6OUTP(TOLD,TINC)
C
      TOUT = TOLD + TINC
      INDEX = 1
      NSIG = 0
      NGEMAX = 500
      CALL D6RTOL(NSIG,NGEMAX,U10)
C     IDIR =  1 MEANS INTEGRATION TO THE RIGHT.
C     IDIR = -1 MEANS INTEGRATION TO THE LEFT.
      IDIR = 1
      IF (TFINAL.LT.TOLD) IDIR = -1
C
      IPOINT(28) = IDIR
C
C     OUTPUT THE INITIAL SOLUTION.
C
      ITYPE = 1
      CALL DPRINT(N,TOLD,YOLD,DYOLD,NG,JROOT,INDEXG,ITYPE,LOUT,INDEX,
     +            RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
C
C     TURN OFF THE ROOTFINDING UNTIL THE SECOND STEP IS COMPLETED.
C
      IFIRST = 0
      IPOINT(26) = IFIRST
      IFOUND = 0
      IPOINT(27) = IFOUND
C
C     CALCULATE THE INITIAL STEPSIZE IF THE USER CHOSE NOT TO
C     DO SO.
C
      IF (HINIT .EQ. 0.0D0) THEN
C
         DO 140 I = 1 , N
            ICOMP = I 
            IF (I.GT.ITOL) ICOMP = 1
            WORK(I) = 0.5D0 * (ABSERR(ICOMP)
     +                + RELERR(ICOMP) * ABS(YOLD(I)))
            IF (WORK(I) .EQ. 0.0D0) THEN
               CALL D6MESG(10,'D6DRV2')
               GO TO 8995
            ENDIF
  140    CONTINUE
         MORDER = 6
         SMALL = D1MACH(4)
         BIG = SQRT(D1MACH(2))
         CALL D6HST1(DERIVS, N, TOLD, TOUT, YOLD, DYOLD, WORK(1),
     +               MORDER, SMALL, BIG, WORK(N+1), WORK(2*N+1),
     +               WORK(3*N+1), WORK(4*N+1), WORK(5*N+1),
     +               YLAG, DYLAG, BETA, YINIT, DYINIT,
     +               IHFLAG, HINIT, IPAIR, RPAR, IPAR)
         IF (N.LE.0) GO TO 8985
         IF (IHFLAG .NE. 0) GO TO 8995
C
      ENDIF
C
C     CHECK IF THE INITIAL STEPSIZE IS TOO SMALL.
C 
      TEMP = MAX(ABS(TOLD),ABS(TOUT))
      TEMP = U13*TEMP
      IF (ABS(HINIT) .LE. TEMP) THEN
         CALL D6MESG(11,'D6INT1')
         GO TO 8995
      ENDIF
C
      CALL D6STAT(N, 2, HINIT, RPAR, IPAR, IDUM, RDUM)
C
C     LIMIT THE INITIAL STEPSIZE TO BE NO LARGER THAN
C     ABS(TFINAL-TOLD) AND NO LARGER THAN HMAX.
C
      HINITM = MIN(ABS(HINIT),ABS(TFINAL-TOLD))
      HINIT = SIGN(HINITM,TFINAL-TOLD)
      HMAX = SIGN(ABS(HMAX),TFINAL-TOLD)
      HNEXT = HINIT
C
C     CALL TO THE INTEGRATOR TO ADVANCE THE INTEGRATION
C     ONE STEP FROM TOLD. REFER TO THE DOCUMENTATION
C     FOR D6DRV3 FOR DESCRIPTIONS OF THE FOLLOWING
C     PARAMETERS.
C
  160 CONTINUE
C
      CALL D6DRV3(N,TOLD,YOLD,DYOLD,TNEW,YNEW,DYNEW,TROOT,YROOT,DYROOT,
     +            TFINAL,DERIVS,ITOL,ABSERR,RELERR,NG,NGUSER,D6GRT3,
     +            GUSER,NSIG,JROOT,INDEXG,WORK,LWORK,IPOINT,NFEVAL,
     +            NGEVAL,RATIO,HNEXT,HMAX,YLAG,DYLAG,BETA,BVAL,YINIT,
     +            DYINIT,QUEUE,LQUEUE,IQUEUE,INDEX,EXTRAP,NUTRAL,
     +            RPAR,IPAR,IER,TROOT2,LOUT)
C
C     CHECK FOR USER REQUESTED TERMINATION.
C
      IF (N.LE.0) GO TO 8985
      IF (INDEX.EQ.-14) GO TO 8985
C
C     TERMINATE THE INTEGRATION IF AN ABNORMAL RETURN
C     WAS MADE FROM THE INTEGRATOR.
C
      IF (INDEX.LT.2) GO TO 8995
C
C     MAKE ANY NECESSARY CHANGES IF A ROOT OCCURRED AT
C     THE INITIAL POINT. THEN CALL THE INTEGRATOR AGAIN.
C
      IF (INDEX.EQ.5) THEN
C
         OFFRT = 1
         NTEMP = N
         NGTEMP = NG
         HINIT = HNEXT
         CALL CHANGE(INDEX,JROOT,INDEXG,NG,N,YROOT,DYROOT,HINIT,TROOT,
     +               TINC,ITOL,ABSERR,RELERR,LOUT,RPAR,IPAR)
C
         TINC = D6OUTP(TROOT,TINC)
C
C        TERMINATE IF THE USER WISHES TO.
C
         IF (N.LE.0) GO TO 8985
         IF (HINIT.EQ.0.0D0) THEN
            CALL D6MESG(39,'D6DRV2')
            GO TO 8995
         ENDIF
C
C        CHECK IF ANY ILLEGAL CHANGES WERE MADE.
C        ALSO RESET THE RELATIVE ERROR TOLERANCES
C        IF NECESSARY.
C
         IF (ITOL.NE.1 .AND. ITOL.NE.N) THEN
            CALL D6MESG(37,'D6DRV2')
            GO TO 8995
         ENDIF
         IF (NTEMP.NE.N) THEN
             CALL D6MESG(30,'D6DRV2')
             GO TO 8995
         ENDIF
C
C        IF THE USER TURNED OFF THE AUTOMATIC ROOTFINDING ...
C
         IF (NG.EQ.-1) THEN
            NG = NGUSER
            NGTEMP = NG
            OFFRT = 0
            CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                  I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                  IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                  JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,1)
            CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                  I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                  IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                  JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,2)
         ENDIF
         IF (NGTEMP.NE.NG) THEN
             CALL D6MESG(31,'D6DRV2')
             GO TO 8995
         ENDIF
         DO 180 I = 1 , N
            IF (I.GT.ITOL) GO TO 180
            IF (RELERR(I).NE.0.0D0 .AND. RELERR(I).LT.RER)
     +      RELERR(I) = RER
            IF (ABSERR(I).LT.0.0D0) THEN
                CALL D6MESG(5,'D6DRV2')
                GO TO 8995
            ENDIF
            IF (RELERR(I).LT.0.0D0) THEN
                CALL D6MESG(4,'D6DRV2')
                GO TO 8995
            ENDIF
            IF ((ABSERR(I).EQ.0.0D0).AND.(RELERR(I).EQ.0.0D0)) THEN
                CALL D6MESG(6,'D6DRV2')
                GO TO 8995
            ENDIF
  180    CONTINUE
         IF (TINC.EQ.0.0D0) THEN
             CALL D6MESG(22,'D6DRV2')
             GO TO 8995
         ENDIF
C
         GO TO 160
C
      ENDIF
C
C     TURN ON THE ROOTFINDING.
C
      IFIRST = 1
      IPOINT(26) = IFIRST
C
C     ----------------------------------------------------------
C
C     IS THIS A ROOT RETURN?
C
      IF ((INDEX.EQ.3) .OR. (INDEX.EQ.4)) THEN
C
C   **************************
C   * THIS IS A ROOT RETURN. *
C   **************************
C
  200     CONTINUE
C
C         HAVE WE PASSED AN OUTPUT POINT?
C
          IDIR = IPOINT(28)
          IF (((TROOT.GT.TOUT).AND. (IDIR.EQ.1)) .OR.
     +        ((TROOT.LT.TOUT).AND. (IDIR.EQ.-1))) THEN
C
C             WE HAVE PASSED THE NEXT OUTPUT POINT.
C             (TROOT IS BEYOND TOUT.)
C
C             INTERPOLATE AT TOUT.
C
              CALL D6INT8(N,NG,TOLD,YOLD,TNEW,WORK,TOUT,YOUT,DYOUT,
     +                    IPOINT)
C
C             MAKE A PRINT RETURN AT TOUT.
C
              ITYPE = 1
              CALL DPRINT(N,TOUT,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,LOUT,
     +                    INDEX,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C             UPDATE THE PRINT RETURN OUTPUT POINT.
C
              TINC = D6OUTP(TOUT,TINC)
C
              TOUT = TOUT + TINC
C
C             LOOP UNTIL THE OUTPUT POINT TOUT LIES BEYOND
C             THE ROOT TROOT.
C
              GO TO 200
C
          ENDIF
C
C         THE ROOT TROOT DOES NOT LIE BEYOND THE NEXT OUTPUT POINT.
C
C         MAKE A PRINT RETURN AT THE ROOT.
C
          ITYPE = 0
          CALL DPRINT(N,TROOT,YROOT,DYROOT,NG,JROOT,INDEXG,ITYPE,LOUT,
     +                INDEX,RPAR,IPAR)
          IF (N.LE.0) GO TO 8985
C
C         DO WE RESTART THE INTEGRATION AT THE ROOT OR
C         CONTINUE THE INTEGRATION FROM TNEW?
C
          RSTRT = IPOINT(29)
C
C         NOTE:
C         IF RSTRT = 0, A CALL TO SUBROUTINE CHANGE WILL BE MADE
C         AT THE INITIAL POINT.
C
          IF (RSTRT.EQ.0) THEN
C
C      -------------------------------------
C        RESTART THE INTEGRATION AT TROOT.
C      -------------------------------------
C
C             FIRST CHECK IF WE ARE CLOSE ENOUGH TO THE FINAL
C             INTEGRATION TIME TO TERMINATE THE INTEGRATION.
C
              HLEFT = TFINAL - TROOT
C
              IF (ABS(HLEFT).LE.U26*MAX(ABS(TROOT),
     +            ABS(TFINAL))) THEN
C
C                 IF WE ARE CLOSE ENOUGH, PERFORM A FORWARD EULER
C                 EXTRAPOLATION TO TFINAL.
C
                  DO 220 I = 1,N
                      YOUT(I) = YROOT(I) + HLEFT*DYROOT(I)
  220             CONTINUE
C
C                 CALCULATE THE DERIVATIVES FOR THE FINAL
C                 EXTRAPOLATED SOLUTION.
C
C                 NOTE:
C                 THE ERROR VECTOR WORK(I8) WILL BE USED
C                 AS SCRATCH STORAGE.
C
                  I8 = IPOINT(8)
                  CALL BETA(N,TFINAL,YOUT,BVAL,RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
                  CALL D6BETA(N,TFINAL,HNEXT,BVAL,IOK)
C
                  IF (IOK.EQ.1) THEN
                      INDEX = -11
                      GO TO 8995
                  ENDIF
C
C                 CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
C    +                        YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
C    +                        IPOINT,WORK,TINIT,EXTRAP,JNDEX,1,0,
C    +                        RPAR,IPAR)
                  CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
     +                        YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
     +                        IPOINT,WORK,TINIT,EXTRAP,JNDEX,2,1,
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C                 D6INT1 DOES NOT APPROXIMATE DYLAG(I) IF BVAL(I)=T
                  IF (NUTRAL) THEN
                     DO 240 I = 1,N
                        IF (BVAL(I).EQ.TFINAL) THEN
C                          EXTRAPOLATE FOR DYLAG.
                           DYLAG(I) = DYROOT(I)
                        ENDIF
  240                CONTINUE
                  ENDIF
C
                  IF (JNDEX.LT.0) THEN
                      INDEX = JNDEX
                      GO TO 8995
                  ENDIF
C
                  CALL DERIVS(N,TFINAL,YOUT,YLAG,DYLAG,DYOUT,
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C
C                 MAKE A PRINT RETURN FOR THE FINAL INTEGRATION TIME.
C
                  ITYPE = 1
                  CALL DPRINT(N,TFINAL,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,
     +                        LOUT,INDEX,RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C
C                 ADJUST THE FINAL MESH TIME IN THE HISTORY QUEUE.
C
                  CALL D6QUE2(N,TFINAL,IPOINT,QUEUE)
C
C                 TERMINATE THE INTEGRATION.
C
                  GO TO 8990
C
              ENDIF
C
C             IT IS NOT YET TIME TO TERMINATE THE INTEGRATION SO
C             PREPARE FOR THE RESTART AT THE ROOT.
C
C             RESET THE INTEGRATION FLAG.
C
              INDEX = 1
C
C             TURN OFF THE ROOTFINDING UNTIL THE SECOND STEP
C             IS COMPLETED.
C
              IFIRST = 0
              IPOINT(26) = IFIRST
              IFOUND = 0
              IPOINT(27) = IFOUND
C
C             MAKE ANY NECESSARY CHANGES IN THE PROBLEM.
C
              OFFRT = 1
              NTEMP = N
              NGTEMP = NG
              HINIT = HNEXT
              HINIT = SIGN(MAX(ABS(HINIT),U65),HINIT)
              CALL CHANGE(INDEX,JROOT,INDEXG,NG,N,YROOT,DYROOT,HINIT,
     +                    TROOT,TINC,ITOL,ABSERR,RELERR,LOUT,
     +                    RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
              HINIT = SIGN(MAX(ABS(HINIT),U65),HINIT)
C
              TINC = D6OUTP(TOUT,TINC)
C
C             TERMINATE IF THE USER WISHES TO.
C
              IF (INDEX.LT.0 .OR. HINIT.EQ.0.0D0) THEN
                  CALL D6MESG(25,'D6DRV2')
                  GO TO 8990
              ENDIF
C
C             CHECK IF ANY ILLEGAL CHANGES WERE MADE.
C             ALSO RESET THE RELATIVE ERROR TOLERANCES
C             IF NECESSARY.
C
              IF (ITOL.NE.1 .AND. ITOL.NE.N) THEN
                 CALL D6MESG(37,'D6DRV2')
                 GO TO 8995
              ENDIF
              IF (NTEMP.NE.N) THEN
                  CALL D6MESG(30,'D6DRV2')
                  GO TO 8995
              ENDIF
C
C             IF THE USER TURNED OFF THE AUTOMATIC ROOTFINDING ...
C
              IF (NG.EQ.-1) THEN
                 NG = NGUSER
                 NGTEMP = NG
                 OFFRT = 0
                 CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,
     +                       I10,I11,I12,I13,I14,I15,I9M1,I10M1,
     +                       I11M1,IORDER,IFIRST,IFOUND,IDIR,RSTRT,
     +                       IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +                       TINIT,WORK,IK7,IK8,IK9,IPAIR,1)
                 CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,
     +                       I10,I11,I12,I13,I14,I15,I9M1,I10M1,
     +                       I11M1,IORDER,IFIRST,IFOUND,IDIR,RSTRT,
     +                       IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +                       TINIT,WORK,IK7,IK8,IK9,IPAIR,2)
              ENDIF
C
              IF (NGTEMP.NE.NG) THEN
                  CALL D6MESG(30,'D6DRV2')
                  GO TO 8995
              ENDIF
              DO 260 I = 1 , N
                 IF (I.GT.ITOL) GO TO 260
                 IF (RELERR(I).NE.0.0D0 .AND. RELERR(I).LT.RER)
     +           RELERR(I) = RER
                 IF (ABSERR(I).LT.0.0D0) THEN
                     CALL D6MESG(5,'D6DRV2')
                     GO TO 8995
                 ENDIF
                 IF (RELERR(I).LT.0.0D0) THEN
                     CALL D6MESG(4,'D6DRV2')
                     GO TO 8995
                 ENDIF
                 IF ((ABSERR(I).EQ.0.0D0).AND.(RELERR(I).EQ.0.0D0)) THEN
                     CALL D6MESG(6,'D6DRV2')
                     GO TO 8995
                 ENDIF
  260         CONTINUE
              IF (TINC.EQ.0.0D0) THEN
                  CALL D6MESG(22,'D6DRV2')
                  GO TO 8995
              ENDIF
C
           IF (OFFRT .NE. 0) THEN
C
C             SHUFFLE THE STORAGE AND INITIALIZE THE NEW LOCATIONS
C             TO ACCOMMODATE THE ADDITIONAL ROOT FUNCTIONS.
C
C             DETERMINE HOW MANY OF THE ROOT FUNCTIONS HAVE
C             A ZERO AT TROOT.
C
              KROOT = 0
              ANCESL = -1
              IF (NG.GT.NGUSER) THEN
                 ANCESL = 10000
                 NGUSP1 = NGUSER + 1
                 DO 280 I = NGUSP1,NG
                    IF (JROOT(I).EQ.1) THEN
                       KROOT = KROOT + 1
C                      MINIMUM LEVEL FROM WHICH THIS ROOT CAME.
                       IF (LEVMAX.GT.0) ANCESL = MIN(ANCESL,LEVEL(I))
                    ENDIF
  280            CONTINUE
                 IF (KROOT .NE. 0) KROOT = N
                 IF (LEVMAX.GT.0) THEN
C                   DO NOT ADD THIS ROOT IF HAVE REACHED THE
C                   MAXIMUM TRACKING LEVEL.
                    IF (ANCESL .GE. LEVMAX) THEN
                       KROOT = 0
                    ENDIF
                 ENDIF
              ENDIF
C
C             CHECK IF THERE IS ENOUGH STORAGE TO CONTINUE.
C
              INEED = 21*N + 1 + 7 * (NG+KROOT)
              IF (INEED.GT.LWORK) THEN
                  CALL D6MESG(9,'D6DRV2')
                  GO TO 8995
              ENDIF
C
              INEED = 2 * (NG + KROOT)
              IF (INEED.GT.LIWORK) THEN
                  CALL D6MESG(21,'D6DRV2')
                  GO TO 8995
              ENDIF
C
C             SHUFFLE THE OLD WORK LOCATIONS.
C
              IF (KROOT.GT.0)
     +        CALL D6GRT4(WORK,NG,KROOT,JROOT,INDEXG,LEVEL,
     +                    TROOT,IPOINT,LEVMAX,ANCESL)
C
C             DEFINE THE NEW NUMBER OF ROOT FUNCTIONS.
C
              NG = NG + KROOT
              IF (OFFRT .EQ. 0) NG = NGUSER
              IPOINT(30) = NG
C
           ENDIF
C
C             RE-SET THE INITIAL TIME.
C
              TOLD = TROOT
C
C             DEFINE THE INITIAL STEPSIZE FOR THE RESTART.
C
              HNEXT = HINIT
C
C             IF A SECOND ROOT WAS LOCATED IN (TROOT,TNEW)
C             LIMIT THE STEPSIZE SO AS NOT TO MISS IT ON
C             RESTARTING.
C
              IF (TROOT2 .NE. TROOT) THEN
                 HLEFT = 0.25D0 * (TROOT2 -TROOT)
                 HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFINAL-TNEW)
              ENDIF
C
C             USE THE INTERPOLATED SOLUTION AT THE ROOT TO
C             UPDATE THE LEFT SOLUTION.
C
              CALL DCOPY(N,YROOT,1,YOLD,1)
C
C             UPDATE THE LEFT DERIVATIVES WITH THE SYSTEM
C             DERIVATIVES FOR THE NEW LEFT SOLUTION (NOT
C             THE INTERPOLATED DERIVATIVES IN DYROOT).
C
C             NOTE:
C             THE ERROR VECTOR WORK(I8) WILL BE USED
C             AS SCRATCH STORAGE.
C
              I8 = IPOINT(8)
              CALL BETA(N,TOLD,YOLD,BVAL,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
              CALL D6BETA(N,TOLD,HNEXT,BVAL,IOK)
C
              IF (IOK.EQ.1) THEN
                  INDEX = -11
                  GO TO 8995
              ENDIF
C
C             CALL D6INT1(N,NG,TNEW,YNEW,HNEXT,BVAL,WORK(I8),YLAG,
C    +                    DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,IPOINT,
C    +                    WORK,TINIT,EXTRAP,JNDEX,1,0,RPAR,IPAR)
              CALL D6INT1(N,NG,TNEW,YNEW,HNEXT,BVAL,WORK(I8),YLAG,
     +                    DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,IPOINT,
     +                    WORK,TINIT,EXTRAP,JNDEX,2,1,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C             D6INT1 DOES NOT APPROXIMATE DYLAG(I) IF BVAL(I)=T
              IF (NUTRAL) THEN
                 DO 300 I = 1,N
                    IF (BVAL(I).EQ.TNEW) THEN
C                      EXTRAPOLATE FOR DYLAG.
                       DYLAG(I) = DYROOT(I)
                    ENDIF
  300            CONTINUE
              ENDIF
C
              IF (JNDEX.LT.0) THEN
                  INDEX = JNDEX
                  GO TO 8995
              ENDIF
C
              CALL DERIVS(N,TOLD,YOLD,YLAG,DYLAG,DYOLD,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C             RESET THE STEPSIZE IF IT WILL THROW US BEYOND
C             THE FINAL INTEGRATION TIME ON THE NEXT STEP.
C
              HLEFT = TFINAL - TROOT
              HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFINAL-TROOT)
C
C             CALL THE INTEGRATOR AGAIN.
C
              GO TO 160
C
          ENDIF
C
          RSTRT = IPOINT(29)
          IF (RSTRT.NE.0) THEN
C
C      ---------------------------------------
C        CONTINUE THE INTEGRATION FROM TNEW.
C      ---------------------------------------
C
C        CAUTION:
C        THIS SECTION IS CURRENTLY INACTIVE. (IN THIS VERSION,
C        AN INTEGRATION RESTART IS PERFORMED AT EACH ROOT OF
C        AN EVENT FUNCTION.)
C
C             TURN OFF THE ROOTFINDING UNTIL THE SECOND STEP
C             IS COMPLETED.
C
              IFIRST = 0
              IPOINT(26) = IFIRST
              IFOUND = 0
              IPOINT(27) = IFOUND
C
C             CHECK IF WE ARE CLOSE ENOUGH TO THE FINAL INTEGRATION
C             TIME TO TERMINATE THE INTEGRATION.
C
              HLEFT = TFINAL - TNEW
C
              IF (ABS(HLEFT).LE.U26*MAX(ABS(TNEW),
     +            ABS(TFINAL))) THEN
C
C                 IF WE ARE CLOSE ENOUGH, PERFORM A FORWARD EULER
C                 EXTRAPOLATION TO TFINAL.
C
                  DO 320 I = 1,N
                      YOUT(I) = YNEW(I) + HLEFT*DYNEW(I)
  320             CONTINUE
C
C                 CALCULATE THE SYSTEM DERIVATIVES FOR THE FINAL
C                 EXTRAPOLATED SOLUTION.
C
                  I8 = IPOINT(8)
                  CALL BETA(N,TFINAL,YOUT,BVAL,RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
                  CALL D6BETA(N,TFINAL,HNEXT,BVAL,IOK)
C
                  IF (IOK.EQ.1) THEN
                      INDEX = -11
                      GO TO 8995
                  ENDIF
C
C                 CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
C    +                        YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
C    +                        IPOINT,WORK,TINIT,EXTRAP,JNDEX,1,0,
C    +                        RPAR,IPAR)
                  CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
     +                        YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
     +                        IPOINT,WORK,TINIT,EXTRAP,JNDEX,2,1,
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C                 D6INT1 DOES NOT APPROXIMATE DYLAG(I) IF BVAL(I)=T
                  IF (NUTRAL) THEN
                     DO 340 I = 1,N
                        IF (BVAL(I).EQ.TFINAL) THEN
C                          EXTRAPOLATE FOR DYLAG.
                           DYLAG(I) = DYNEW(I)
                        ENDIF
  340                CONTINUE
                  ENDIF
C
                  IF (JNDEX.LT.0) THEN
                      INDEX = JNDEX
                      GO TO 8995
                  ENDIF
C
                  CALL DERIVS(N,TFINAL,YOUT,YLAG,DYLAG,DYOUT,
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C
C                 MAKE A PRINT RETURN FOR THE FINAL INTEGRATION TIME.
C
                  ITYPE = 1
                  CALL DPRINT(N,TFINAL,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,
     +                        LOUT,INDEX,RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C
C                 ADJUST THE FINAL MESH TIME IN THE HISTORY QUEUE.
C
                  CALL D6QUE2(N,TFINAL,IPOINT,QUEUE)
C
C                 TERMINATE THE INTEGRATION.
C
                  GO TO 8990
C
              ENDIF
C
C        IT IS NOT YET TIME TO TERMINATE THE INTEGRATION
C        SO CONTINUE THE INTEGRATION FROM TNEW.
C
C             UPDATE THE INDEPENDENT VARIABLE.
C
              TOLD = TNEW
C
C             UPDATE THE LEFT SOLUTION.
C
              CALL DCOPY(N,YNEW,1,YOLD,1)
C
C             UPDATE THE LEFT DERIVATIVES WITH THE SYSTEM
C             DERIVATIVES FOR THE NEW LEFT SOLUTION.
C
              CALL DCOPY(N,DYNEW,1,DYOLD,1)
C
C             RESET THE STEPSIZE IF IT WILL THROW US BEYOND
C             THE FINAL INTEGRATION TIME ON THE NEXT STEP.
C
              HLEFT = TFINAL - TNEW
              HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFINAL-TNEW)
C
C             CALL THE INTEGRATOR AGAIN.
C
              GO TO 160
C
          ENDIF
C
C     ----------------------------------------------------------
C
C   ******************************
C   * THIS IS NOT A ROOT RETURN. *
C   ******************************
C
      ELSE
C
  360     CONTINUE
C
C         HAVE WE REACHED AN OUTPUT POINT?
C
          IDIR = IPOINT(28)
          IF (((TNEW.GE.TOUT).AND. (IDIR.EQ.1)) .OR.
     +        ((TNEW.LE.TOUT).AND. (IDIR.EQ.-1))) THEN
C
C             WE HAVE PASSED THE NEXT OUTPUT POINT.
C             (TNEW IS BEYOND TOUT.)
C
C             INTERPOLATE AT TOUT.
C
              CALL D6INT8(N,NG,TOLD,YOLD,TNEW,WORK,TOUT,YOUT,DYOUT,
     +                    IPOINT)
C
C             MAKE A PRINT RETURN AT TOUT.
C
              ITYPE = 1
              CALL DPRINT(N,TOUT,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,LOUT,
     +                    INDEX,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C             UPDATE THE PRINT RETURN OUTPUT POINT.
C
              TINC = D6OUTP(TOUT,TINC)
C
              TOUT = TOUT + TINC
C
C             LOOP UNTIL THE PRINT OUTPUT POINT TOUT LIES BEYOND
C             THE INTEGRATOR OUTPUT POINT TNEW.
C
              GO TO 360
C
          ENDIF
C
C         TNEW DOES NOT LIE BEYOND THE NEXT OUTPUT POINT.
C         PREPARE FOR ANOTHER CALL TO THE INTEGRATOR OR
C         TERMINATE THE INTEGRATION.
C
C         EXTRAPOLATE THE SOLUTION TO TFINAL AND TERMINATE
C         THE INTEGRATION IF CLOSE ENOUGH TO TFINAL.
C
          HLEFT = TFINAL - TNEW
C
          IF (ABS(HLEFT).LE.U26*MAX(ABS(TNEW),ABS(TFINAL))) THEN
C
C             IF WE ARE CLOSE ENOUGH, PERFORM A FORWARD EULER
C             EXTRAPOLATION TO TFINAL.
C
              DO 380 I = 1,N
                  YOUT(I) = YNEW(I) + HLEFT*DYNEW(I)
  380         CONTINUE
C
C             CALCULATE THE DERIVATIVES FOR THE FINAL
C             EXTRAPOLATED SOLUTION.
C
C             NOTE:
C             THE ERROR VECTOR WORK(I8) WILL BE USED
C             AS SCRATCH STORAGE.
C
              I8 = IPOINT(8)
              CALL BETA(N,TFINAL,YOUT,BVAL,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
              CALL D6BETA(N,TFINAL,HNEXT,BVAL,IOK)
C
              IF (IOK.EQ.1) THEN
                  INDEX = -11
                  GO TO 8995
              ENDIF
C
C             CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
C    +                    YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
C    +                    IPOINT,WORK,TINIT,EXTRAP,JNDEX,1,0,
C    +                    RPAR,IPAR)
              CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
     +                    YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
     +                    IPOINT,WORK,TINIT,EXTRAP,JNDEX,2,1,
     +                    RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C             D6INT1 DOES NOT APPROXIMATE DYLAG(I) IF BVAL(I)=T
              IF (NUTRAL) THEN
                 DO 400 I = 1,N
                    IF (BVAL(I).EQ.TFINAL) THEN
C                      EXTRAPOLATE FOR DYLAG.
                       DYLAG(I) = DYNEW(I)
                    ENDIF
  400            CONTINUE
              ENDIF
C
              IF (JNDEX.LT.0) THEN
                  INDEX = JNDEX
                  GO TO 8995
              ENDIF
C
              CALL DERIVS(N,TFINAL,YOUT,YLAG,DYLAG,DYOUT,
     +                    RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C
C             MAKE A PRINT RETURN FOR THE FINAL INTEGRATION TIME.
C
C
              ITYPE = 1
              CALL DPRINT(N,TFINAL,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,
     +                    LOUT,INDEX,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C             TERMINATE THE INTEGRATION.
C
              GO TO 8990
C
          ENDIF
C
C         IT IS NOT YET TIME TO TERMINATE THE INTEGRATION.
C         PREPARE TO CALL THE INTEGRATOR AGAIN TO CONTINUE
C         THE INTEGRATION FROM TNEW.
C
C         UPDATE THE INDEPENDENT VARIABLE.
C
          TOLD = TNEW
C
C         UPDATE THE LEFT SOLUTION.
C
          CALL DCOPY(N,YNEW,1,YOLD,1)
C
C         UPDATE THE LEFT DERIVATIVES WITH THE SYSTEM
C         DERIVATIVES FOR THE NEW LEFT SOLUTION.
C
          CALL DCOPY(N,DYNEW,1,DYOLD,1)
C
C         RESET THE STEPSIZE IF IT WILL THROW US BEYOND
C         THE FINAL INTEGRATION TIME ON THE NEXT STEP.
C
          HLEFT = TFINAL - TNEW
          HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFINAL-TNEW)
C
C         CALL THE INTEGRATOR AGAIN.
C
          GO TO 160
C
      ENDIF
C
C     ----------------------------------------------------------
C
 8985 CONTINUE
C
C     USER TERMINATION.
C
      CALL D6MESG(25,'D6DRV2')
      IFLAG = 1
      GO TO 9005
C
 8990 CONTINUE
C
C     NORMAL TERMINATION.
C
      IFLAG = 0
      GO TO 9005
C
 8995 CONTINUE
C
C     ABNORMAL TERMINATION.
C
      CALL D6MESG(19,'D6DRV2')
      IFLAG = 1
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6DRV3
      SUBROUTINE D6DRV3(N,TOLD,YOLD,DYOLD,TNEW,YNEW,DYNEW,TROOT,YROOT,
     +                  DYROOT,TFINAL,DERIVS,ITOL,ABSERR,RELERR,NG,
     +                  NGUSER,GRESID,GUSER,NSIG,JROOT,INDEXG,WORK,
     +                  LWORK,IPOINT,NFEVAL,NGEVAL,RATIO,HNEXT,HMAX,
     +                  YLAG,DYLAG,BETA,BVAL,YINIT,DYINIT,QUEUE,
     +                  LQUEUE,IQUEUE,INDEX,EXTRAP,NUTRAL,
     +                  RPAR,IPAR,IER,TROOT2,LOUT)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6DRV3
C***SUBSIDIARY
C***PURPOSE  The function of subroutine D6DRV3 is to solve N
C            ordinary differential equations with state dependent
C            delays. D6DRV3 is a third-level step oriented driver
C            called by subroutine D6DRV2.
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
C     D6DRV3 IS A STEP-ORIENTED RUNGE-KUTTA-SARAFYAN INTEGRATOR.
C     IT HAS FACILITIES TO HANDLE INTERPOLATION AND ROOTFINDING.
C     IT IS INTENDED PRIMARILY FOR PROBLEMS FOR WHICH OUTPUT IS
C     IS DESIRED AT A LARGE NUMBER OF POINTS OR FOR WHICH IT IS
C     DESIRABLE TO LOCATE THE ZEROES OF FUNCTIONS WHICH DEPEND
C     ON THE SOLUTION. THIS VERSION IS MODIFIED TO SOLVE DELAY
C     DIFFERENTIAL EQUATIONS. IT SHOULD NOT BE USED DIRECTLY BUT
C     SHOULD BE ACCESSED VIA SUBROUTINES DKLAG6 AND D6DRV2.
C
C     PARAMETERS -
C
C     N          - NUMBER OF DIFFERENTIAL EQUATIONS (INPUT).
C                  N MUST BE GREATER THAN 0.
C
C     TOLD       - CURRENT VALUE OF INDEPENDENT VARIABLE (INPUT).
C                  D6DRV3 WILL ATTEMPT TO STEP FROM TOLD TO
C                  TOLD + HNEXT. (IT MAY BE NECESSARY TO TAKE A
C                  SHORTER STEP IN ORDER TO SATISFY THE ERROR
C                  TEST.) ON OUTPUT, TNEW WILL CONTAIN THE POINT
C                  TO WHICH THE STEP WAS ACTUALLY TAKEN.)
C
C     YOLD       - DOUBLE PRECISION ARRAY OF LENGTH N
C                  CONTAINING THE SOLUTION AT TOLD (INPUT).
C                  YOLD IS NOT ALTERED BY D6DRV3.
C
C     DYOLD      - DOUBLE PRECISION ARRAY OF LENGTH N
C                  CONTAINING THE DERIVATIVES AT TOLD (INPUT).
C                  NOTE:
C                  WHEN D6DRV3 RETURNS TO THE CALLING PROGRAM
C                  WITH INDEX = 2, YNEW AND DYNEW CONTAIN
C                  THE SOLUTION AND DERIVATIVE AT TNEW.
C                  DYOLD IS NOT ALTERED BY D6DRV3.
C                  ON THE NEXT CALL IF TOLD IS THIS VALUE
C                  OF TNEW, SIMPLY COPY YNEW INTO YOLD
C                  AND DYNEW INTO DYOLD, RATHER THAN
C                  ACTUALLY CALLING SUBROUTINE DERIVS TO
C                  CALCULATE DYOLD.
C
C     TNEW       - THE VALUE OF THE INDEPENDENT VARIABLE TO WHICH
C                  THE INTEGRATION WAS COMPLETED (OUTPUT).
C
C     YNEW       - DOUBLE PRECISION ARRAY OF LENGTH N (INPUT).
C                  ON OUTPUT, YNEW CONTAINS THE SOLUTION AT TNEW.
C
C     DYNEW      - DOUBLE PRECISION ARRAY OF LENGTH N (INPUT).
C                  ON OUTPUT, DYNEW CONTAINS THE DERIVATIVES
C                  AT TNEW.
C
C     TROOT      - THE LEFTMOST ROOT LOCATED FOR ANY RESIDUAL
C                  EQUATION (OUTPUT). TROOT = TNEW IF NO ROOTS
C                  WERE LOCATED.
C
C     YROOT      - DOUBLE PRECISION ARRAY OF LENGTH N (INPUT).
C                  ON OUTPUT, YROOT CONTAINS THE THE INTERPOLATED
C                  SOLUTION AT TROOT.
C
C     DYROOT     - DOUBLE PRECISION ARRAY OF LENGTH N (INPUT).
C                  ON OUTPUT, DYROOT CONTAINS THE THE INTERPOLATED
C                  SOLUTION AT TROOT.
C
C     TFINAL     - THE FINAL VALUE OF THE INDEPENDENT VARIABLE
C                  TO WHICH THE PROBLEM WILL BE INTEGRATED
C                  (INPUT).
C
C     DERIVS     - NAME OF EXTERNAL USER SUPPLIED SUBROUTINE
C                  TO CALCULATE THE SYSTEM DERIVATIVES (INPUT).
C                  DERIVS MUST HAVE THE FORM
C                  SUBROUTINE DERIVS (N,T,Y,YLAG,DYLAG,DY)
C                  AND MUST CALCULATE THE DERIVATIVE ARRAY
C                  DY(I),I=1,...,N GIVEN THE INDEPENDENT VARIABLE
C                  T, AN APPROXIMATION Y(I), I=1,...,N TO THE
C                  SOLUTION AT TIME T, AN APPROXIMATION
C                  YLAG(I), I=1,...,N TO THE DELAYED SOLUTION
C                  YLAG, AND AN APPROXIMATION TO THE DELAYED
C                  DERIVATIVE DYLAG(I), I=1,...,N. DERIVS MUST
C                  BE DECLARED EXTERNAL IN THE CALLING PROGRAM.
C                  CAUTION:
C                  FOR ANY COMPONENT WITH BETA(T,Y) = T, DYLAG(I)
C                  WILL NOT BE SUPPLIED WHEN DERIVS IS CALLED.
C                  THE INTEGRATION MAY BE TERMINATED BY SETTING
C                  N TO A NEGATIVE VALUE WHEN THIS SUBROUTINE
C                  IS CALLED.
C
C     ITOL       - ABSERR(*) AND RELERR(*) WILL BE TREATED
C                  AS VECTORS OF LENGTH ITOL. ITOL MUST BE
C                  1 OR N.
C
C     ABSERR     - DOUBLE PRECISION VECTOR OF LENGTH ITOL.
C                  ABSOLUTE ERROR TOLERANCE(S) FOR THE
C                  INTEGRATOR (INPUT). ABSERR(I) MUST BE
C                  NONNEGATIVE FOR EACH I.
C
C     RELERR     - DOUBLE PRECISION VECTOR OF LENGTH ITOL.
C                  RELATIVE ERROR TOLERANCE(S) FOR THE
C                  INTEGRATOR (INPUT). RELERR(I) MUST BE
C                  NONNEGATIVE FOR EACH I. NOT BOTH ABSERR(I)
C                  AND RELERR(I) CAN BE ZERO FOR ANY I.
C
C     NG         - NUMBER OF RESIDUAL FUNCTIONS FOR
C                  WHICH ROOTS WILL BE SOUGHT (INPUT).
C                  NG MUST BE NONNEGATIVE.
C
C     NGUSER     - NUMBER OF USER DEFINED RESIDUAL FUNCTIONS
C                  FOR WHICH ROOTS WILL BE SOUGHT (INPUT).
C                  NGUSER MUST BE NONNEGATIVE.
C
C     GRESID     - NAME OF EXTERNAL SUBROUTINE
C                  TO CALCULATE THE RESIDUALS FOR THE
C                  ROOT FUNCTIONS (INPUT). GRESID MUST HAVE
C                  THE FORM
C                  SUBROUTINE GRESID(N,T,Y,DY,NG,NGUSER,
C                                    GRESID,GVAL,BETA,BVAL,
C                                    TG,INDEXG)
C                  AND MUST CALCULATE THE RESIDUALS
C                  GVAL(I), I=1,...,NG, GIVEN THE INDEPENDENT
C                  VARIABLE T, AN APPROXIMATION Y(I), I=1,...,N
C                  TO THE SOLUTION Y, AND AN APPROXIMATION
C                  DY(I), I=1,...,N TO THE DERIVATIVE.
C                  TG IS AN ARRAY OF LENGTH NG WHICH CONTAINS
C                  ALL PREVIOUSLY LOCATED ROOTS. INDEXG IS AN
C                  ARRAY OF LENGTH NG WHICH CONTAINS THE ODE
C                  COMPONENT NUMBERS CORRESPONDING TO EACH OF
C                  THE ROOT FUNCTIONS. THESE ARRAYS MUST BE
C                  INITIALIZED AS IN SUBROUTINE D6DRV2 BEFORE
C                  CALLING D6DRV2.
C                  NGUSER OF THE NUMBER OF USER DEFINED ROOT
C                  FUNCTIONS. GUSER IS THE NAME OF THE EXTERNAL
C                  USER DEFINED ROOT FUNCTION RESIDUAL SUBROUTINE.
C                  NOTE:
C                  GRESID MAY CALL DERIVS TO CALCULATE
C                  THE EXACT DERIVATIVES CORRESPONDING
C                  TO Y. IF THIS IS DONE, GRESID MAY OVERWRITE
C                  DY WITH THE CALCULATED DERIVATIVES.
C                  GRESID MUST BE DECLARED EXTERNAL IN THE
C                  CALLING PROGRAM.
C                  THE INTEGRATION MAY BE TERMINATED BY SETTING
C                  N TO A NEGATIVE VALUE WHEN THIS SUBROUTINE
C                  IS CALLED.
C                  NOTE:
C                  ----
C                  IN THIS IMPLEMENTATION, THE NAME D6GRT3 IS
C                  PASSED TO D6DRV3 BY SUBROUTINE D6DRV2.
C
C     GUSER      - NAME OF USER SUPPLIED EXTERNAL SUBROUTINE
C                  TO CALCULATE THE RESIDUALS FOR THE
C                  ROOT FUNCTIONS (INPUT). GUSER MUST HAVE
C                  THE FORM
C                  SUBROUTINE GUSER(N,T,Y,DY,NGUSER,GVAL)
C                  AND MUST CALCULATE THE RESIDUALS
C                  GVAL(I), I=1,...,NGUSER, GIVEN THE INDEPENDENT
C                  VARIABLE T, AN APPROXIMATION Y(I), I=1,...,N
C                  TO THE SOLUTION Y, AND AN APPROXIMATION
C                  DY(I), I=1,...,N TO THE DERIVATIVE.
C                  NOTE:
C                  GUSER MAY CALL DERIVS TO CALCULATE
C                  THE EXACT DERIVATIVES CORRESPONDING
C                  TO Y. IF THIS IS DONE, GUSER MAY OVERWRITE
C                  DY WITH THE CALCULATED DERIVATIVES.
C                  GUSER MUST BE DECLARED EXTERNAL IN THE
C                  CALLING PROGRAM.
C                  THE INTEGRATION MAY BE TERMINATED BY SETTING
C                  N TO A NEGATIVE VALUE WHEN THIS SUBROUTINE
C                  IS CALLED.
C
C     NSIG       - NUMBER OF SIGNIFICANT DIGITS DESIRED
C                  IN THE ROOTFINDING CALCULATIONS (INPUT).
C                  THE TOLERANCE USED BY D6DRV3 IS
C                  MAX(ABS(TOLD),ABS(TOLD + H)) TIMES
C                  THE LARGER OF 10.0D0 ** (-NSIG) AND
C                  10.0D0 * DROUND WHERE DROUND IS THE
C                  MACHINE UNIT ROUNDOFF. NSIG WILL BE
C                  IGNORED IF IT IS 0.
C
C     JROOT      - INTEGER ARRAY OF LENGTH NG INDICATING WHICH
C                  RESIDUAL EQUATIONS HAVE ROOTS AT TROOT
C                  (OUTPUT). IF JROOT(I)=1, THE I-TH
C                  EQUATION HAS A ROOT AT TROOT. OTHERWISE,
C                  JROOT(I)=0.
C
C     INDEXG     - INTEGER ARRAY OF LENGTH NG INDICATING THE
C                  ODE COMPONENT NUMBERS FOR EACH OF THE ROOT
C                  FUNCTIONS(INPUT). THIS ARRAY MUST BE INITIALIZED
C                  BEFORE CALLING D6DRV3.
C
C     WORK       - DOUBLE PRECISION WORK ARRAY OF LENGTH
C                  LWORK (INPUT). DO NOT ALTER THE CONTENTS
C                  OF WORK BETWEEN CALLS TO D6DRV3.
C
C     LWORK      - LENGTH OF THE WORK ARRAY (INPUT). LWORK
C                  MUST BE AT LEAST
C                              21*N + 7NG + 1.
C
C     IPOINT     - INTEGER ARRAY OF LENGTH 35 (INPUT). ON OUTPUT,
C                  IPOINT CONTAINS THE VALUES OF VARIOUS POINTERS
C                  WHICH WILL BE USED BY D6DRV3 ON SUBSEQUENT
C                  CALLS. THE CONTENTS OF IPOINT MUST NOT BE
C                  ALTERED BETWEEN CALLS TO D6DRV3.
C
C     NFEVAL     - NUMBER OF CALLS MADE TO DERIVS (OUTPUT).
C
C     NGEVAL     - NUMBER OF CALLS MADE TO GRESID (OUTPUT).
C
C     RATIO      - THE ESTIMATED ERROR RATIO DEFINED BY
C                     RATIO = MAX (E(I),I=1,...,N)
C                  WHERE E(I) IS THE ESTIMATED ERROR PER
C                  STEP FOR COMPONENT I DIVIDED BY
C                     RELERR * ABS(YNEW(I)) + ABSERR
C                  (OUTPUT).
C
C     HNEXT      - THE STEPSIZE USED BY D6DRV3. ON INPUT,
C                  HNEXT IS THE STEPSIZE THAT WILL BE
C                  ATTEMPTED. ON OUTPUT, IT WILL BE THE
C                  STEPSIZE THAT WAS ACTUALLY USED
C                  SUCCESSFULLY. HNEXT MAY BE CHANGED
C                  BEFORE CALLING D6DRV3 AGAIN.
C
C     HMAX       - ON INPUT, THE MAXIMUM ALLOWABLE STEPSIZE
C                  FOR D6DRV3. THE STEPSIZE IS LIMITED BY
C                  D6DRV3 TO BE NO LARGER THAN THE SMALLEST
C                  SYSTEM DELAY SO ITS VALUE MAY BE CHANGED
C                  ON OUTPUT.
C
C     YLAG       - DOUBLE PRECISION ARRAY OF LENGTH N USED
C                  AS SCRATCH STORAGE BY D6DRV3 WHEN THE
C                  DELAY SOLUTION IS APPROXIMATED (INPUT).
C
C     DYLAG      - DOUBLE PRECISION ARRAY OF LENGTH N USED
C                  AS SCRATCH STORAGE BY D6DRV3 WHEN THE
C                  DELAY DERIVATIVE IS APPROXIMATED (INPUT).
C
C     BETA       - NAME OF USER SUPPLIED SUBROUTINE TO EVALUATE
C                  THE SYSTEM DELAYS (INPUT). BETA MUST BE DECLARED
C                  EXTERNAL IN THE CALLING PROGRAM. BETA MUST
C                  HAVE THE FORM
C                               BETA(N,T,Y,BVAL)
C                  AND MUST EVALUATE THE DELAYS BVAL(I), I=1,...,N,
C                  AT TIME T, GIVEN N, T, AND AN APPROXIMATION
C                  TO THE SOLUTION Y(I), I=1,...,N. FOR EACH
C                  T, BVAL(I) MUST BE LESS THAN OR EQUAL TO
C                  T IF THE INTEGRATION GOES FROM LEFT TO RIGHT
C                  AND GREATER THAN OR EQUAL TO T IF IT GOES
C                  FROM RIGHT TO LEFT.
C                  THE INTEGRATION MAY BE TERMINATED BY SETTING
C                  N TO A NEGATIVE VALUE WHEN THIS SUBROUTINE
C                  IS CALLED.
C
C     BVAL       - DOUBLE PRECISION ARRAY OF LENGTH N USED
C                  AS SCRATCH STORAGE BY D6DRV3 WHEN THE
C                  SYSTEM DELAYS ARE EVALUATED (INPUT).
C
C     YINIT      - NAME OF USER SUPPLIED SUBROUTINE TO EVALUATE
C                  THE DELAYED SOLUTION, ONE COMPONENT AT A TIME,
C                  FOR VALUES OF T IN THE INITIAL DELAY INTERVALS
C                  (INPUT). YINIT MUST BE DECLARED EXTERNAL IN THE
C                  CALLING PROGRAM. YINIT MUST HAVE THE FORM
C                       SUBROUTINE YINIT(N,T,YLAG,I)
C                  AND MUST EVALUATE THE I-TH COMPONENT YLAG(I)
C                  OF THE DELAYED SOLUTION AT TIME T GIVEN N, T,
C                  AND I. YINIT WILL BE CALLED ANY TIME D6DRV3
C                  NEEDS TO KNOW THE DELAYED SOLUTION FOR T IF
C                  BVAL(I) .LE. T .LE. TINIT, WHERE TINIT IS
C                  THE VALUE OF TOLD ON THE FIRST CALL TO
C                  D6DRV3.
C                  THE INTEGRATION MAY BE TERMINATED BY SETTING
C                  N TO A NEGATIVE VALUE WHEN THIS SUBROUTINE
C                  IS CALLED.
C
C     DYINIT     - NAME OF USER SUPPLIED SUBROUTINE TO EVALUATE
C                  THE DELAYED DERIVATIVE, ONE COMPONENT AT A TIME,
C                  FOR VALUES OF T IN THE INITIAL DELAY INTERVALS
C                  (INPUT). DYINIT MUST BE DECLARED EXTERNAL IN THE
C                  CALLING PROGRAM. DYINIT MUST HAVE THE FORM
C                       SUBROUTINE DYINIT(N,T,DYLAG,I)
C                  AND MUST EVALUATE THE I-TH COMPONENT DYLAG(I)
C                  OF THE DELAYED DERIVATIVE AT TIME T GIVEN N, T, AND
C                  I. YINIT WILL BE CALLED ANY TIME D6DRV3
C                  NEEDS TO KNOW THE DELAYED DERIVATIVE FOR T IF
C                  BVAL(I) .LE. T .LE. TINIT, WHERE TINIT IS
C                  THE VALUE OF TOLD ON THE FIRST CALL TO
C                  D6DRV3.
C                  THE INTEGRATION MAY BE TERMINATED BY SETTING
C                  N TO A NEGATIVE VALUE WHEN THIS SUBROUTINE
C                  IS CALLED.
C
C     QUEUE      - DOUBLE PRECISION ARRAY OF LENGTH LQUEUE (INPUT).
C                  QUEUE IS USED BY D6DRV3 TO STORE THE HISTORY
C                  INFORMATION USED BY D6DRV3 TO INTERPOLATE THE
C                  DELAYED SOLUTION AT PREVIOUS TIMES.
C
C     LQUEUE     - LENGTH OF THE QUEUE HISTORY ARRAY (INPUT). THE
C                  PREVIOUS SOLUTION INFORMATION CAN BE STORED
C                  FOR AS MANY AS LQUEUE/(12*(N+1)) POINTS. LQUEUE
C                  SHOULD BE ROUGHLY 12*(N+1)*(TFINAL-TINIT)/HAVG
C                  WHERE TFINAL IS THE FINAL INTEGRATION TIME,
C                  TINIT IS THE INITIAL INTEGRATION TIME, AND HAVG
C                  IS THE AVERAGE STEPSIZE USED BY D6DRV3.
C
C     IQUEUE     - QUEUE ARRAY POINTER. IQUEUE MUST BE SET TO
C                  0 BEFORE THE FIRST CALL TO D6DRV3 AND MUST
C                  NOT BE CHANGED ON SUCCEEDING CALLS.
C
C     INDEX      - INTEGRATION STATUS FLAG.
C
C                  ON INPUT, USE INDEX = 1 TO START (OR RESTART)
C                  AN INTEGRATION. TO CONTINUE AN INTEGRATION,
C                  USE THE LAST VALUE OF INDEX RETURNED BY D6DRV3.
C
C                  ON OUTPUT, INDEX WILL HAVE ONE OF THE FOLLOWING
C                  VALUES -
C
C                   =  2  INDICATES NORMAL RETURN.
C                         THE INTEGRATION WAS
C                         COMPLETED TO TNEW AND
C                         NO ROOTS WERE LOCATED.
C                   =  3  INDICATES THE INTEGRATION
C                         WAS COMPLETED TO TNEW AND
C                         A ROOT WAS LOCATED
C                         AT TROOT.
C                   =  4  INDICATES THE INTEGRATION
C                         WAS COMPLETED TO TNEW,
C                         GRESID HAS A ZERO AT TNEW, AND
C                         THERE ARE NO SIGN CHANGES
C                         ON THE INTERVAL (TOLD,TNEW).
C                   =  5  INDICATES GRESID HAS A ROOT AT
C                         TOLD AND THAT THE INTEGRATION
C                         WAS THEREFORE NOT STARTED.
C                         CALL D6DRV3 TO CONTINUE.
C                   = -1  NG IS LESS THAN 0.
C                   = -2  N IS LESS THAN 1.
C                   = -3  RELERR IS LESS THAN 0.0D0,
C                         ABSERR IS LESS THAN
C                         0.0D0, OR BOTH RELERR AND
C                         ABSERR ARE 0.0D0.
C                   = -4  THE STEPSIZE IS TOO SMALL
C                         I.E.,
C                         ABS(TNEW - TOLD) IS NOT
C                         LARGER THAN
C                         13.0D0 * DROUND
C                         * MAX(ABS(TOLD),ABS(TNEW)),
C                         WHERE DROUND IS UNIT ROUNDOFF
C                         AND TNEW = TOLD + HNEXT.
C                   = -5  LWORK IS TOO SMALL.
C                   = -6  NSIG IS LESS THAN 0.
C                   = -7  INDEX IS LESS THAN 1 OR
C                         GREATER THAN 5.
C                   = -8  A COMPONENT OF THE SOLUTION
C                         VANISHED AND ABSERR = 0.0D0. IT
C                         IS IMPOSSIBLE TO CONTINUE
C                         WITH A PURE RELATIVE ERROR
C                         TEST.
C                   = -9  TFINAL IS EQUAL TO TOLD.
C                   =-10  THE ROOTFINDER FAILED TO CONVERGE.
C                   =-11  AN ILLEGAL DELAY WAS ENCOUNTERED.
C                   =-12  AN EXTRAPOLATION PAST THE OLDEST
C                         DATA IN THE SOLUTION HISTORY QUEUE
C                         WAS REQUESTED.
C                   =-13  ITOL IS NOT EQUAL TO 1 OR N.
C                   =-14  THE USER TERMINATED THE INTEGRATION.
C                   =-15  CONVERGENCE COULD NOT BE OBTAINED
C                         IN D6STP2.
C
C     EXTRAP     - LOGICAL METHOD FLAG. IF EXTRAP = .FALSE.,
C                  THE SOLVER WILL USE METHODS DESIGNED FOR
C                  PROBLEMS WHICH DO NOT HAVE ISOLATED
C                  VANISHING DELAYS. OTHERWISE, IT WILL USE
C                  METHODS DESIGNED FOR PROBLEMS WITH
C                  ISOLATED VANISHING DELAYS.
C
C     NUTRAL     - LOGICAL METHOD FLAG. IF NUTRAL = .TRUE.,
C                  THE SOLVER WILL ASSUME THE CALCULATION OF
C                  DY/DT INVOLVES A DELAYED DERIVATIVE.
C
C     RPAR       - DOUBLE PRECISION USER ARRAY PASSED VIA
C                  DKLAG6
C
C     IPAR       - INTEGER ARRAY PASSED VIA DKLAG6
C
C     IER        - ERROR PARAMETER (OUTPUT).
C                   =  0 IF INDEX = 2 OR 3 (NORMAL).
C                   =  128 - INDEX IF INDEX IS
C                      NEGATIVE (ABNORMAL).
C
C     TROOT2     - IF A SECOND INTERPOLATORY ROOT IS FOUND
C                  IT IS RETURNED AS TROOT2. OTHERWUSE
C                  TROOT = TROOT.
C
C     LOUT       - OUTPUT UNIT NUMBER FOR ERROR MESSAGES
C
C     NOTE:
C
C     FOLLOWING A SUCCESSFUL RETURN FROM D6DRV3, IF THE INTERPOLATED
C     SOLUTION YOUT IS DESIRED AT A TIME TOUT, YOUT MAY BE CALCULATED
C     USING SUBROUTINE D6INT8 AS FOLLOWS -
C
C            CALL D6INT8(N, NG, TOLD, YOLD, TNEW, WORK, TOUT, YOUT,
C                        DYOUT, IPOINT)
C
C     AUXILIARY SUBROUTINES -
C
C     D6STAT  -  DO NOTHING SUBROUTINE WHICH IS CALLED
C                FOLLOWING SUCCESSFUL STEPS. MAY BE
C                REPLACED BY A USER SUBROUTINE FOR
C                COLLECTING INTEGRATION STATISTICS
C                (USER REPLACEABLE)
C
C     D6OUTP  -  DEFINE THE OUTPUT PRINT INCREMENT
C                (USER REPLACEABLE, IF OUTPUT AT
C                UNEQUALLY SPACED POINTS IS DESIRED)
C
C     D6RERS  -  DEFINE THE FLOOR VALUE FOR THE RELATIVE
C                ERROR TOLERANCES (USER REPLACEABLE)
C                (USER REPLACEABLE)
C
C     D6RTOL  -  SET CONVERGENCE PARAMETERS FOR THE ROOTFINDER
C                (USER REPLACEABLE)
C
C     D6BETA  -  CHECK FOR ILLEGAL DELAYS
C                (USER REPLACEABLE)
C
C     D6ESTF  -  DEFINE THE TYPE OF ERROR ESTIMATE TO BE
C                USED BY DKLAG6 (USER REPLACEABLE)
C
C     D6DISC  -  SUPPLY DISCONTINUITY INFORMATION FOR
C                POINTS PRIOR TO THE INITIAL TIME
C                (USER REPLACEABLE)
C
C     D6AFTR  -  CALLED FOLLOWING EACH SUCCESSFUL
C                INTEGRATION STEP (USER REPLACEABLE)
C
C     D1MACH  -  DEFINE MACHINE CONSTANTS
C
C     DAXPY   -  FORM A LINEAR COMBINATION OF TWO VECTORS
C                (FROM LINPACK)
C
C     DCOPY   -  COPY A VECTOR TO ANOTHER VECTOR
C                (FROM LINPACK)
C
C     D6ERR1  -  CALCULATE THE ERROR RATIO
C
C     D6QUE2  -  ADJUST SOLUTION HISTORY QUEUE IF AN
C                EXTRAPOLATION IS PERFORMED NEAR THE
C                FINAL INTEGRATION TIME
C
C     D6GRT3  -  GRESID SUBROUTINE USED TO DO ROOTFINDING
C                FOR DERIVATIVE DISCONTINUITIES
C
C     D6GRT4  -  SHUFFLE STORAGE IN THE EVENT ROOTFINDING
C                IS DONE FOR DERIVATIVE DISCONTINUITIES
C
C     D6HST1  -  CALCULATE THE INITIAL STEPSIZE
C                (VARIANT OF WATTS' ODEPACK INITIAL
C                STEPSIZE SUBROUTINE)
C
C     D6HST4  -  LIMIT THE STEPSIZE NOT TO EXCEED THE
C                MAXIMUM STEPSIZE
C
C     D6HST3  -  CALCULATE THE INTEGRATION STEPSIZE
C
C     D6INT8  -  INTERPOLATE THE SOLUTION AND DERIVATIVE
C
C     D6VEC3  -  CALCULATE MAXIMUM NORM FOR D6HST1
C
C     D6INT2  -  INTERPOLATE THE INTERIOR OF THE QUEUE
C                SOLUTION HISTORY ARRAY
C
C     D6POLY  -  EVALUATE THE INTERPOLATION POLYNOMIALS
C
C     D6WSET  -  DEFINE THE SOLUTION AND DERIVATIVE
C                INTERPOLATION COEFFICIENTS FOR D6POLY
C
C     D6PNTR  -  CALCULATE, SAVE, AND RESTORE POINTERS
C
C     D6GRT1  -  DIRECT THE ROOTFINDER
C
C     D6GRT2  -  DO THE NECESSARY ROOTFINDING
C
C     D6SRC1  -  BINARY SEARCH SUBROUTINE CALLED BY D6SRC3
C
C     D6STAV  -  INTERPOLATE DURING THE FIRST INTEGRATION STEP
C                FOR VANISHING DELAY PROBLEMS
C
C     D6STP1  -  APPLY THE BASIC INTEGRATION METHOD FOR
C                NONVANISHING DELAY PROBLEMS
C
C     D6STP2  -  APPLY THE BASIC INTEGRATION METHOD FOR
C                VANISHING DELAY PROBLEMS
C
C     D6SRC2  -  SEARCH SUBROUTINE CALLED BY D6INT1 FOR
C                NONVANISHING DELAY PROBLEMS
C
C     D6SRC3  -  SEARCH SUBROUTINE CALLED BY D6INT1 FOR
C                VANISHING DELAY PROBLEMS
C
C     D6VEC2  -  MULTIPLY A VECTOR BY A CONSTANT
C
C     D6QUE1  -  UPDATE THE SOLUTION HISTORY QUEUE
C
C     D6INT1  -  INTERPOLATE THE SOLUTION HISTORY QUEUE
C
C     D6INT6  -  INTERPOLATION SUBROUTINE CALLED BY D6INT1
C
C     D6MESG  -  DEFINE AND PROCESS ERROR MESSAGES
C
C     D6VEC1  -  LOAD A CONSTANT INTO A VECTOR
C
C     D6INT4  -  INTERPOLATE THE INTERIOR OF THE QUEUE
C                SOLUTION HISTORY ARRAY (VARIANT OF D6INT2)
C
C     D6SRC4  -  SEARCH SUBROUTINE CALLED BY D6INT4 FOR
C                NONVANISHING DELAY PROBLEMS (VARIANT OF
C                D6SRC2)
C
C     D6INT5  -  VARIANT OF D6INT4 WHICH ALLOWS INTERPOLATION
C                OF SELECTED SOLUTION COMPONENTS RATHER THAN
C                THE ENTIRE SOLUTION VECTOR (NOT CALLED BY
C                DKLAG6) 
C
C     D6HST2  -  SUBROUTINE CALLED TO USE A FAMILY OF IMBEDDED
C                METHODS TO GENERATE AN INTERPOLANT FOR
C                VANISHING DELAY PROBLEMS
C
C     D6INT9  -  SUBROUTINE TO INTERPOLATE THE SOLUTION AND THE
C                DERIVATIVE USING A FAMILY OF IMBEDDED METHODS
C
C     FORTRAN FUNCTIONS USED - DBLE, LOG10, SIGN, ABS, MIN,
C                              MAX, SQRT, FLOAT, MOD, REAL
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  DCOPY, D6BETA, D6ERR1, D6GRT3, D6HST4, D6HST3,
C            D6POLY, D6PNTR, D6GRT1, D6STP1, D6STP2, D6MESG,
C            D6QUE1, D6INT1, D1MACH, D6RTOL, D6ESTF
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6DRV3
C
      IMPLICIT NONE
C
      INTEGER IER,INDEX,IQUEUE,LQUEUE,LWORK,N,NFEVAL,NG,NGEVAL,
     +        NSIG,NGUSER,IVAN,ICVAL,ITOL,ICOMP,NGEMAX,JCVAL,I,IDUM,
     +        I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IBEGIN,IDIR,IERRUS,IERTST,IFIRST,
     +        IFOUND,INDEXO,INEED,IOK,IORDER,RSTRT,IROOTL,
     +        JNDEX,JQUEUE,KQUEUE,NUMPT,MAXTRY,ITRY,ICONV,
     +        INDEXG,IPOINT,JROOT,IPAR,IK7,IK8,IK9,IPAIR,
     +        JCOMP
      DOUBLE PRECISION BVAL,QUEUE,WORK,YLAG,YNEW,YOLD,YROOT,DYNEW,DYOLD,
     +                 DYROOT,DYLAG,RPAR,ABSERR,DROUND,D1MACH,GRTOL,H,
     +                 HMAX,HNEXT,HNM1,RATIO,RELERR,TEMP,TFINAL,TINIT,
     +                 TNEW,TNM1,TOLD,TROOT,U10,U13,RDUM,TROOT2
C
C     DOUBLE PRECISION HVAN
      INTEGER ITELLE,ITELLI,LOUT
C
      LOGICAL EXTRAP,EXTRPT,NUTRAL
C
      DIMENSION YOLD(*),DYOLD(*),YNEW(*),DYNEW(*),YROOT(*),DYROOT(*),
     +          JROOT(*),WORK(*),IPOINT(*),YLAG(*),BVAL(*),QUEUE(N+1,*),
     +          INDEXG(*),DYLAG(*),ABSERR(*),RELERR(*),RPAR(*),IPAR(*),
     +          JCOMP(1)
C
      EXTERNAL DERIVS,GUSER,GRESID,BETA,YINIT,DYINIT
C
C     CHECK THE TERMINATION FLAG.
C
C***FIRST EXECUTABLE STATEMENT  D6DRV3
C
      IF (INDEX.EQ.0) INDEX = 2
      IF (INDEX.LT.1) INDEX = -7
      IF (INDEX.GT.5) INDEX = -7
      IF (INDEX.EQ.-7) THEN
          CALL D6MESG(1,'D6DRV3')
          GO TO 220
      ENDIF
C
C     ---------------
C     INITIALIZATION.
C     ---------------
C
C     BE SURE THE FOLLOWING MACHINE DEPENDENT PARAMETER
C     IS DEFINED APPROPRIATELY. IT SHOULD BE DEFINED AS
C     DROUND = 2.0D0 ** (1-K) WHERE K IS THE NUMBER OF BITS
C     IN THE MANTISSA FOR A DOUBLE PRECISION FLOATING
C     POINT NUMBER.
C
      DROUND = D1MACH(4)
C
      U10 = 10.0D0*DROUND
      U13 = 13.0D0*DROUND
C
      IER = 0
      TNEW = TOLD
      TROOT = TOLD
      TROOT2 = TOLD
      NFEVAL = 0
      NGEVAL = 0
      NGEMAX = 500
      CALL D6RTOL(NSIG,NGEMAX,U10)
      GRTOL = 0.0D0
      IF (NSIG.GT.0) GRTOL = 10.0D0** (-NSIG)
      GRTOL = MAX(GRTOL,U10)
      RATIO = 0.0D0
      HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HMAX)),TFINAL-TOLD)
      H = HNEXT
      IVAN = 0
C     HVAN = H
C
C     CHECK THE VALIDITY OF THE PARAMETERS.
C
      IF (ITOL.NE.1 .AND. ITOL.NE.N) THEN
         CALL D6MESG(36,'D6DRV3')
         INDEX = -13
      ENDIF
      IF (NG.LT.0) THEN
          INDEX = -1
          CALL D6MESG(2,'D6DRV3')
      ENDIF
      IF (NGUSER.LT.0) THEN
          INDEX = -1
          CALL D6MESG(2,'D6DRV3')
      ENDIF
      IF (N.LT.1) THEN
          INDEX = -2
          CALL D6MESG(3,'D6DRV3')
      ENDIF
      DO 40 I = 1 , N
         IF (I.GT.ITOL) GO TO 40
         IF (ABSERR(I).LT.0.0D0) THEN
             CALL D6MESG(5,'D6DRV3')
             INDEX = -3
         ENDIF
         IF (RELERR(I).LT.0.0D0) THEN
             CALL D6MESG(4,'D6DRV3')
             INDEX = -3
         ENDIF
         IF ((ABSERR(I).EQ.0.0D0).AND.(RELERR(I).EQ.0.0D0)) THEN
             CALL D6MESG(6,'D6DRV3')
             INDEX = -3
         ENDIF
   40 CONTINUE
      IF (NSIG.LT.0) THEN
          INDEX = -6
          CALL D6MESG(7,'D6DRV3')
      ENDIF
      IF (TFINAL.EQ.TOLD) THEN
          INDEX = -9
          CALL D6MESG(8,'D6DRV3')
      ENDIF
      IF (HNEXT.EQ.0.0D0) THEN
          INDEX = -4
          CALL D6MESG(11,'D6DRV3')
      ENDIF
      INEED = 12*(N+1)
      IF (INEED.GT.LQUEUE) THEN
          INDEX = -5
          CALL D6MESG(20,'D6DRV3')
      ENDIF
C
      IF (INDEX.LT.0) GO TO 220
C
C     IF THIS IS THE FIRST CALL OR AN INTEGRATION RESTART ...
C
      IF (INDEX.EQ.1) GO TO 60
C
C     OTHERWISE CONTINUE THE INTEGRATION ....
C
      GO TO 140
C
   60 CONTINUE
C
C     ONCE AN EXTRAPOLATORY ROOT HAS BEEN FOUND, THE
C     ROOTFINDING WILL BE TURNED OFF UNTIL AN
C     INTERPOLATORY ROOT HAS BEEN FOUND AND THE
C     THE INTEGRATION HAS BEEN RESTARTED.
C
      IFOUND = 1
      IPOINT(27) = IFOUND
C
C     DEFINE THE POINTERS AND INTEGRATION FLAGS.
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TOLD,WORK,IK7,IK8,IK9,IPAIR,1)
C
C     THE FOLLOWING VARIABLES ARE NOT PROTECTED BY THE D6DRV3
C     PARAMETER LIST.
C
      IDIR = IPOINT(28)
      RSTRT = IPOINT(29)
      IF (IQUEUE .NE. 0) THEN
         JQUEUE = IPOINT(22)
         KQUEUE = IPOINT(23)
         IFIRST = IPOINT(26)
         IFOUND = IPOINT(27)
      ENDIF
C
C     SAVE THE POINTERS AND INTEGRATION FLAGS.
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,2)
C
C     CHECK IF ENOUGH STORAGE HAS BEEN ALLOCATED
C     FOR THE WORK ARRAY.
C
      INEED = I15 + NG - 1
      IF (INEED.GT.LWORK) THEN
          CALL D6MESG(9,'D6DRV3')
          INDEX = -5
          GO TO 220
      ENDIF
C
C     CALCULATE THE INITIAL RESIDUALS IF NECESSARY.
C
      IF (NG.EQ.0) GO TO 140
C
      CALL GRESID(N,TOLD,YOLD,DYOLD,NG,NGUSER,GUSER,WORK(I9),
     +            BETA,BVAL,INDEXG,WORK(I15),RPAR,IPAR)
      NGEVAL = NGEVAL + 1
      IF (N.LE.0) GO TO 90
      CALL DCOPY(NG,WORK(I9),1,WORK(I11),1)
C
C     CHECK FOR A ZERO AT THE LEFT ENDPOINT.
C
      DO 80 I = 1,NG
          JROOT(I) = 0
          IF (WORK(I9M1+I).EQ.0.0D0) JROOT(I) = 1
   80 CONTINUE
      DO 100 I = 1,NG
          IF (JROOT(I).EQ.1) GO TO 120
  100 CONTINUE
      GO TO 140
  120 CONTINUE
      INDEX = 5
      CALL DCOPY(N,YOLD,1,YROOT,1)
      GO TO 240
C
  140 CONTINUE
C
C     CONTINUATION OF THE INTEGRATION ...
C
C     RESTORE THE POINTERS AND INTEGRATION FLAGS.
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
C     ----------------------------------
C     EXTRAPOLATORY ROOTFINDING SECTION.
C     ----------------------------------
C
C     IF ROOTFINDING IS BEING DONE AND THIS IS NOT THE FIRST
C     INTEGRATION STEP AND AN EXTRAPOLATORY ROOT HAS NOT
C     ALREADY BEEN FOUND ...
C
      IFOUND = IPOINT(27)
      IF ((NG.GT.0.).AND.(IFOUND.EQ.1)) THEN
C
          IFIRST = IPOINT(26)
          IF (IFIRST.EQ.1) THEN
C
C             IF IROOTL = 1, THE EXTRAPOLATORY ROOTFINDING WILL
C             BE BYPASSED.
              IROOTL = 1
              CALL D6EXTP (IROOTL)
C
              IF (IROOTL.EQ.0) THEN
C
C                 CHECK IF THE MOST RECENT POLYNOMIAL PREDICTS
C                 A ROOT WILL OCCUR BETWEEN TOLD AND TNEW. IF
C                 SO, APPROXIMATE THE ROOT USING EXTRAPOLATION
C                 AND REDUCE THE STEPSIZE TO MAKE THE APPROXIMATE
C                 ROOT AN INTEGRATION MESHPOINT.
C
                  INDEXO = IQUEUE
                  NUMPT = LQUEUE/ (12*(N+1))
                  IBEGIN = NUMPT + (INDEXO-1)*11
                  TNEW = TOLD + H
                  HNM1 = QUEUE(2,INDEXO) - QUEUE(1,INDEXO)
                  TNM1 = QUEUE(1,INDEXO)
C
C                 EXTRAPOLATE THE MOST RECENT SOLUTION POLYNOMIAL
C                 TO TNEW.
C
                  CALL D6POLY(N,TNM1,QUEUE(1,IBEGIN+1),HNM1,
     +                        QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+8),
     +                        QUEUE(1,IBEGIN+4),QUEUE(1,IBEGIN+5),
     +                        QUEUE(1,IBEGIN+6),QUEUE(1,IBEGIN+7),
     +                        QUEUE(1,IBEGIN+9),QUEUE(1,IBEGIN+10),
     +                        QUEUE(1,IBEGIN+11),YNEW,DYNEW,TNEW,
     +                        IORDER,JCOMP,IPAIR,3,0,0.0D0,0.0D0,0,
     +                        1)
C
C                 LOAD THE RESIDUALS WORK(I11) FOR TOLD AND
C                 CALCULATE THE RESIDUALS WORK(I10) FOR THE
C                 EXTRAPOLATED SOLUTION AT TNEW.
C
                  CALL GRESID(N,TNEW,YNEW,DYNEW,NG,NGUSER,GUSER,
     +                        WORK(I10),BETA,BVAL,INDEXG,WORK(I15),
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 90
C
C                 UPDATE THE FUNCTION COUNTER.
C
                  NGEVAL = NGEVAL + 1
C
C                 CALL THE ROOTFINDER TO USE THE EXTRAPOLATED
C                 POLYNOMIAL TO LOCATE THE ROOT.
C
                  CALL D6GRT1(N,TOLD,QUEUE(1,IBEGIN+1),YNEW,DYNEW,TROOT,
     +                        YROOT,DYROOT,TNEW,NG,NGUSER,GRESID,GUSER,
     +                        JROOT,HNM1,INDEX,IER,NGEVAL,GRTOL,
     +                        QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+8),
     +                        QUEUE(1,IBEGIN+4),QUEUE(1,IBEGIN+5),
     +                        QUEUE(1,IBEGIN+6),QUEUE(1,IBEGIN+7),
     +                        QUEUE(1,IBEGIN+9),QUEUE(1,IBEGIN+10),
     +                        QUEUE(1,IBEGIN+11),WORK(I9),WORK(I10),
     +                        WORK(I14),WORK(I12),WORK(I13),WORK(I15),
     +                        INDEXG,BETA,BVAL,TNM1,IORDER,U10,IPAIR,
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 90
C
C                 CHECK IF THE ROOTFINDER ENCOUNTERED AN ERROR.
C
                  IF (INDEX.LT.0) THEN
                      CALL D6MESG(12,'D6DRV3')
                      GO TO 220
                  ENDIF
C
C                 PROCEED TO THE INTEGRATION STEP CALL TO D6STP*
C                 IF NO SIGN CHANGE WAS DETECTED. OTHERWISE ...
C
                  IF (INDEX.NE.2) THEN
C
C                     THE PURPOSE OF THE NEXT PRINT IS TO INFORM THE
C                     USER EACH TIME THE CODE LOCATES AN EXTRAPOLATORY
C                     ROOT. IT CAN BE DISABLED BY SETTING ITELLE = 1.
C
                      ITELLE = 0
                      IF (ITELLE.EQ.0) THEN
                          WRITE(LOUT,111) INDEX,TROOT,H
  111                     FORMAT(' AN EXTRAPOLATORY ROOT WAS FOUND',
     +                           ' IN D6DRV3.',/,
     +                           ' INDEX, TROOT, H =:',/, I3,2D20.10)
C
                      ENDIF
C
C                     PROCESS THE ROOT ONLY IF IT IS NOT
C                     TOO CLOSE TO TOLD.
C
                      TEMP = MAX(ABS(TOLD),ABS(TROOT))
                      TEMP = U13*TEMP
                      IF (ABS(TROOT-TOLD).GT.TEMP) THEN
C
C                     SET THE FLAG TO TELL THIS SECTION THAT
C                     AN EXTRAPOLATORY ROOT HAS BEEN FOUND.
C                     IT WILL NOT BE RE-SET AGAIN UNTIL AN
C                     INTERPOLATORY ROOT HAS BEEN FOUND.
C
                      IFOUND = 0
                      IPOINT(27) = IFOUND
C
C                     RESET THE STEPSIZE TO HIT THE EXTRAPOLATED
C                     ROOT.
C
                      H = TROOT - TOLD
C
                  ENDIF
C
                  ENDIF
C
              ENDIF
C
          ENDIF
C
      ENDIF
C
C     ----------------------------------------------
C     INTEGRATION STEP FROM TOLD TO TNEW = TOLD + H.
C     ----------------------------------------------
C
C     CALL D6STP* TO CALCULATE THE NEW APPROXIMATION YNEW.
C
C     NOTE:
C     THE ERROR VECTOR WORK(I8) WILL BE USED AS
C     SCRATCH STORAGE.
C
C     DO THE STEP INTEGRATION.
C
      IF (.NOT. EXTRAP)
     +CALL D6STP1(N,NG,TOLD,YOLD,DYOLD,H,DERIVS,TNEW,INDEX,YNEW,
     +            WORK(I1),WORK(I2),WORK(I3),WORK(I4),WORK(I5),WORK(I6),
     +            WORK(I7),WORK(IK7),WORK(IK8),WORK(IK9),
     +            NFEVAL,IORDER,YLAG,DYLAG,BETA,BVAL,WORK(I8),YINIT,
     +            DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,2,IVAN,EXTRAP,
     +            U13,IPAIR,RPAR,IPAR)
       IF (N.LE.0) GO TO 90
C
      MAXTRY = 30
      ITRY = 0
      IF (EXTRAP) THEN
  160 CONTINUE
      ITRY = ITRY + 1
      CALL D6STP2(N,NG,TOLD,YOLD,DYOLD,H,DERIVS,TNEW,INDEX,YNEW,
     +            WORK(I1),WORK(I2),WORK(I3),WORK(I4),WORK(I5),WORK(I6),
     +            WORK(I7),WORK(IK7),WORK(IK8),WORK(IK9),
     +            NFEVAL,IORDER,YLAG,DYLAG,BETA,BVAL,WORK(I8),YINIT,
     +            DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,2,IVAN,EXTRAP,
     +            NUTRAL,RPAR,IPAR,ITOL,ABSERR,RELERR,ICONV,WORK(I1+N),
     +            WORK(I2+N),WORK(I3+N),WORK(I4+N),WORK(I5+N),
     +            WORK(I6+N),WORK(I7+N),WORK(IK7+N),WORK(IK8+N),
     +            WORK(IK9+N),IPAIR)
       IF (N.LE.0) GO TO 90
       IF (ICONV .EQ. 1) THEN
          IF (ITRY .GT. MAXTRY) THEN
             CALL D6MESG(40,'D6STP2')
             INDEX = -15
             GO TO 220
          ENDIF
          H = 0.5D0 * H
          GO TO 160
       ENDIF
       ENDIF
C
C     CHECK FOR ERROR RETURN.
C
      IF (INDEX.LT.0) GO TO 220
C
C     ---------------------------------------------
C     ERROR TESTING AND STEPSIZE SELECTION SECTION.
C     ---------------------------------------------
C
C     CHECK IF IT IS ADVISABLE TO CONTINUE THE INTEGRATION
C     WITH A PURE RELATIVE ERROR TEST.
C
      DO 180 I = 1 , N
         ICOMP = I
         IF (I.GT.ITOL) ICOMP = 1
         IF (ABSERR(ICOMP).NE.0.0D0) GO TO 180
         IF (YNEW(I)*YOLD(I).GT.0.0D0) GO TO 180
         INDEX = -8
         CALL D6MESG(10,'D6DRV3')
         GO TO 220
  180 CONTINUE
C
C     DETERMINE THE TYPE OF ERROR ESTIMATE TO BE USED.
C
      JCVAL = 2
      EXTRPT = EXTRAP
      CALL D6ESTF(EXTRPT,JCVAL)
      EXTRAP = EXTRPT
      ICVAL = JCVAL
      IF (ICVAL .LT.  1) ICVAL = 2
      IF (ICVAL .EQ.  5) ICVAL = 2
      IF (ICVAL .EQ.  6) ICVAL = 2
      IF (ICVAL .GT. 10) ICVAL = 2
C
C     CALCULATE THE MAXIMUM LOCAL ERROR ESTIMATE RATIO.
C
      CALL D6ERR1(N,ABSERR,RELERR,INDEX,RATIO,YNEW,WORK(I1),WORK(I3),
     +            WORK(I4),WORK(I5),WORK(I6),WORK(I7),WORK(IK7),
     +            WORK(IK8),WORK(IK9),WORK(I8),ICVAL,ITOL,IPAIR)
C
C     CHECK IF THE ESTIMATED ERROR IS ACCEPTABLE
C     AND CALCULATE AN ESTIMATED STEPSIZE WITH WHICH
C     TO REDO THE STEP OR WITH WHICH TO TAKE
C     THE NEXT STEP.
C
      CALL D6HST3(IERTST,RATIO,H,HNEXT,IERRUS,IORDER,ICVAL)
C
      CALL D6STAT(N, IERTST+3, H, RPAR, IPAR, IDUM, RDUM)
C
C     CHECK FOR ILLEGAL DELAYS FOR THE NEW SOLUTION.
C
      CALL BETA(N,TNEW,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 90
      CALL D6BETA(N,TNEW,HNEXT,BVAL,IOK)
      IF (IOK.EQ.1) THEN
          INDEX = -11
          GO TO 220
      ENDIF
C
C     LIMIT THE STEPSIZE TO BE NO LARGER THAN
C     THE MAXIMUM STEPSIZE.
C
      CALL D6HST4(HNEXT,HMAX)
C
C     IF THE ESTIMATED ERROR IS SMALL ENOUGH, CONTINUE.
C
      IF (IERTST.EQ.0) GO TO 200
C
C     IF THE STEP NEEDS TO BE REDONE, CHECK IF THE
C     ESTIMATED STEPSIZE IS TOO SMALL.
C
      TEMP = MAX(ABS(TOLD),ABS(TNEW))
      TEMP = U13*TEMP
      IF (ABS(HNEXT).LE.TEMP) THEN
          INDEX = -4
          CALL D6MESG(11,'D6DRV3')
          GO TO 220
      ENDIF
C
C     IF THE STEPSIZE IS NOT TOO SMALL, REDO THE STEP
C     WITH THE ESTIMATED STEPSIZE AS THE NEW STEPSIZE.
C
      H = HNEXT
      GO TO 140
C
  200 CONTINUE
C
C     K8 CONTAINS THE DERIVATIVE AT T+H SO THERE IS NO NEED TO
C     CALCULATE THE DERIVATIVES FOR THE APPROXIMATE SOLUTION
C     AT THIS POINT.
C
      CALL DCOPY(N,WORK(IK8),1,DYNEW,1)
C
C     ----------------------------------
C     INTERPOLATORY ROOTFINDING SECTION.
C     ----------------------------------
C
      IF (NG.GT.0) THEN
C
C         CALCULATE THE RESIDUALS AT TNEW, USING
C         EXACT DERIVATIVES.
C
          CALL GRESID(N,TNEW,YNEW,DYNEW,NG,NGUSER,GUSER,WORK(I10),
     +                BETA,BVAL,INDEXG,WORK(I15),RPAR,IPAR)
          NGEVAL = NGEVAL + 1
          IF (N.LE.0) GO TO 90
C
C         DO NOT MAKE ANOTHER RETURN FOR A PREVIOUSLY
C         REPORTED ROOT.
C
C         DO THE INTERPOLATORY ROOTFINDING.
C
          INDEX = 2
          IF (IFIRST .EQ. 1)
     +    CALL D6GRT5(N,TOLD,YOLD,YNEW,DYNEW,TROOT,YROOT,DYROOT,TNEW,NG,
     +                NGUSER,GRESID,GUSER,JROOT,H,INDEX,IER,NGEVAL,
     +                GRTOL,WORK(I1),WORK(I3),WORK(I4),WORK(I5),
     +                WORK(I6),WORK(I7),WORK(IK7),WORK(IK8),WORK(IK9),
     +                WORK(I9),WORK(I10),WORK(I11),WORK(I12),WORK(I13),
     +                WORK(I15),INDEXG,BETA,BVAL,TOLD,IORDER,U10,IPAIR,
     +                RPAR,IPAR,TROOT2)
          IF (N.LE.0) GO TO 90
C
C         TURN THE EXTRAPOLATORY ROOTFINDING BACK ON
C         FOR SUCCEEDING STEPS.
C
          IFOUND = 1
          IPOINT(27) = IFOUND
C
C         CHECK IF THE ROOTFINDER WAS SUCCESSFUL.
C
          IF (INDEX.LT.0) THEN
              CALL D6MESG(12,'D6DRV3')
              GO TO 220
          ENDIF
C
C         IF A ROOT WAS FOUND THEN ...
C
          IF ((INDEX.EQ.3) .OR. (INDEX.EQ.4)) THEN
C
C             THE PURPOSE OF THE NEXT PRINT IS TO INFORM THE
C             USER EACH TIME THE CODE LOCATES AN INTERPOLATORY
C             ROOT. IT CAN BE DISABLED BY SETTING ITELLI = 1.
C
              ITELLI = 0
              IF (ITELLI.EQ.0) THEN
                  WRITE(LOUT,112) INDEX,TROOT,H
  112             FORMAT(' AN INTERPOLATORY ROOT WAS FOUND',
     +            ' IN D6DRV3.',
     +            ' INDEX, TROOT, H =:',/, I3,2D20.10)
                  IF (TROOT2 .NE. TROOT) THEN
                     WRITE(LOUT,113) TROOT2
  113                FORMAT(' IN ADDITION, A SECOND ROOT WAS',
     +                      ' FOUND AT TROOT = ', D20.10)
                  ENDIF
              ENDIF
C
          ENDIF
C
C         IF THE ROOT OCCURRED AT TNEW THEN  DO NOT INCREASE THE
C         STEPSIZE FOR THE NEXT STEP.
C
          IF (INDEX .EQ. 4)
     +    HNEXT = SIGN(MIN(ABS(HNEXT),ABS(H)),H)
C
C         IF THE ROOT DID NOT OCCUR AT TNEW THEN ...
C
          IF (INDEX.EQ.3) THEN
C
C             D6DRV2 WILL RESTART THE INTEGRATION AT TROOT.
C
              H = TROOT - TOLD
C             HNEXT = H
C
C             FOR THE NEXT STEP CHOOSE THE LARGER OF THE DISTANCES
C             FROM THE ROOT TO THE ENDPOINTS.
C
              HNEXT = SIGN(MAX(ABS(TROOT-TOLD),ABS(TNEW-TROOT)),H)
C
C             CALL D6STP* TO CALCULATE THE NEW APPROXIMATION YNEW
C             AT TROOT. USE AN HEURISTIC AND ASSUME THAT THE ERROR
C             TEST WILL PASS SINCE IT PASSED FOR THE ORIGINAL TNEW.
C             THE PURPOSE OF THIS CALL IS TO GET NEW RUNGE-KUTTA
C             WEIGHTS THAT CORRESPOND TO THE INTERVAL (TOLD,TROOT)
C             RATHER THAN TO THE ORIGINAL INTERVAL (TOLD,TNEW).
C
C             NOTE:
C             THE ERROR VECTOR WORK(I8) WILL BE USED AS
C             SCRATCH STORAGE.
C
              IVAN = 0
C
              IF (.NOT. EXTRAP)
     +        CALL D6STP1(N,NG,TOLD,YOLD,DYOLD,H,DERIVS,TNEW,JNDEX,YNEW,
     +                    WORK(I1),WORK(I2),WORK(I3),WORK(I4),WORK(I5),
     +                    WORK(I6),WORK(I7),WORK(IK7),WORK(IK8),
     +                    WORK(IK9),NFEVAL,IORDER,YLAG,DYLAG,
     +                    BETA,BVAL,WORK(I8),YINIT,DYINIT,QUEUE,LQUEUE,
     +                    IPOINT,WORK,TINIT,1,IVAN,EXTRAP,U13,IPAIR,
     +                    RPAR,IPAR)
              IF (N.LE.0) GO TO 90
C
              MAXTRY = 30
              ITRY = 0
              IF (EXTRAP) THEN
              ITRY = ITRY + 1
              CALL D6STP2(N,NG,TOLD,YOLD,DYOLD,H,DERIVS,TNEW,JNDEX,YNEW,
     +            WORK(I1),WORK(I2),WORK(I3),WORK(I4),WORK(I5),WORK(I6),
     +            WORK(I7),WORK(IK7),WORK(IK8),WORK(IK9),
     +            NFEVAL,IORDER,YLAG,DYLAG,BETA,BVAL,WORK(I8),YINIT,
     +            DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,2,IVAN,EXTRAP,
     +            NUTRAL,RPAR,IPAR,ITOL,ABSERR,RELERR,ICONV,WORK(I1+N),
     +            WORK(I2+N),WORK(I3+N),WORK(I4+N),WORK(I5+N),
     +            WORK(I6+N),WORK(I7+N),WORK(IK7+N),WORK(IK8+N),
     +            WORK(IK9+N),IPAIR)
              IF (N.LE.0) GO TO 90
              IF (ICONV .EQ. 1) THEN
                 CALL D6MESG(41,'D6STP2')
                 INDEX = -15
                 GO TO 220
              ENDIF
              ENDIF
C
C             CHECK FOR ERROR RETURN.
C
              IF (JNDEX.LT.0) THEN
                  INDEX = JNDEX
                  GO TO 220
              ENDIF
C
              TNEW = TROOT
C
C             K8 CONTAINS THE DERIVATIVE AT T+H SO THERE IS NO NEED
C             TO RECALCULATE THE DERIVATIVES FOR THE APPROXIMATE
C             SOLUTION AT THIS POINT.
C
              CALL DCOPY(N,WORK(IK8),1,DYNEW,1)
C
C             LOAD YNEW AND DYNEW INTO YROOT AND DYROOT. (YNEW AND
C             DYNEW HAVE CHANGED SINCE THE ROOTFINDING WAS DONE.)
C
              CALL DCOPY(N,YNEW,1,YROOT,1)
              CALL DCOPY(N,DYNEW,1,DYROOT,1)
C
C             DO NOT RECALCULATE THE RESIDUALS SINCE THAT WOULD
C             OVERWRITE THE EXACT ZEROES LOADED INTO G(TROOT)
C             BY D6GRT1.
C
          ENDIF
C
C         UPDATE THE RESIDUALS.
C
          CALL DCOPY(NG,WORK(I10),1,WORK(I9),1)
C
      ENDIF
C
C     ------------------------------------------
C     UPDATE THE HISTORY QUEUE BEFORE RETURNING.
C     ------------------------------------------
C
C     NOTE:
C     TNEW IS NOW THE FIRST OF EITHER THE ROOT
C     OR THE LAST SUCCESSFUL INTEGRATION POINT.
C
      CALL D6QUE1(N,QUEUE,LQUEUE,IPOINT,TOLD,YOLD,TNEW,WORK(I1),
     +            WORK(I3),WORK(I4),WORK(I5),WORK(I6),WORK(I7),
     +            WORK(IK7),WORK(IK8),WORK(IK9))
C
C     ALLOW THE USER TO UPDATE ANYTHING THAT NEEDS TO BE UPDATED
C     FOLLOWING THE SUCCESSFUL STEP SOLUTION QUEUE UPDATE.
C
      CALL D6AFTR(N, TOLD, YOLD, DYOLD, TNEW, YNEW, DYNEW,
     +            WORK(I1), WORK(I2), WORK(I3), WORK(I4),
     +            WORK(I5), WORK(I6), WORK(I7), WORK(IK7),
     +            WORK(IK8), WORK(IK9),IORDER, RPAR, IPAR)
C
C     RESTORE THE POINTERS AND INTEGRATION FLAGS.
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
      GO TO 240
C
   90 CONTINUE
C
C     THE USER TERMINATED THE INTEGRATION.
C
      INDEX = -14
C
C     ISSUE A TERMINAL ERROR MESSAGE.
C
  220 CONTINUE
C
      IER = 128 - INDEX
      CALL D6MESG(13,'D6DRV3')
C
  240 CONTINUE
C
      RETURN
      END
C*DECK D6STP1
      SUBROUTINE D6STP1(N,NG,TOLD,YOLD,DYOLD,H,DERIVS,TNEW,INDEX,YNEW,
     +                  K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,NFEVAL,IORDER,
     +                  YLAG,DYLAG,BETA,BVAL,YWORK,YINIT,DYINIT,QUEUE,
     +                  LQUEUE,IPOINT,WORK,TINIT,ICHECK,IVAN,EXTRAP,
     +                  U13,IPAIR,RPAR,IPAR)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6STP1
C***SUBSIDIARY
C***PURPOSE  The purpose of D6STP1 is to perform a single step
C            of the integration for the DKLAG6 differential
C            equation solver (for non-vanishing delay problems).
C            D6STP1 is called by the interval oriented driver
C            D6DRV3 if the logical variable EXTRAP is set to 
C            .FALSE. in the user provided subroutine INITAL.
C            It reduces the stepsize if it encounters a delay time
C            beyond the solution history queue and it does not
C            interpolate the derivative at the current time. It is
C            therefore not appropriate for vanishing delay problems
C            particularly neutral problems with vanishing delays.
C            The appropriate subroutine to use for such problems is
C            D6STP2 which extrapolates the most recent solution
C            polynomial, if necessary, and then performs a predictor-
C            corrector like iteration of the Runge-Kutta-Sarafyan
C            solution polynomials. D6STP2 is used if the logical
C            variable EXTRAP is set to .TRUE. in the user provided
C            subroutine INITAL.
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
C***ROUTINES CALLED  DAXPY, DCOPY, D6BETA, D6MESG, D6INT1,
C                    D6POLY
C***REVISION HISTORY  (YYMMDD)
C***END PROLOGUE  D6STP1
C
C     PERFORM A RUNGE-KUTTA-SARAFYAN STEP OF SIZE H
C     FROM TOLD FOR D6DRV3.
C
C     CAUTION:
C     D6STP1 WILL REDUCE H IF IT ENCOUNTERS AN ILLEGAL DELAY.
C
      IMPLICIT NONE
C
      INTEGER ICHECK,IHTRY,INDEX,IOK,IORDER,IPOINT,LQUEUE,
     +        MAXTRY,N,NFEVAL,NG,IVAN,IPAR,I,ICOMP,IPAIR
      DOUBLE PRECISION BVAL,DYOLD,H,HTRY,K0,K1,K2,K3,K4,K5,
     +                 K6,K7,K8,K9,QUEUE,TINIT,TNEW,TOLD,
     +                 TVAL,WORK,YLAG,YNEW,YOLD,YWORK,DYLAG,
     +                 U13,TEMP,RPAR,KFAC
      LOGICAL EXTRAP
C
      DIMENSION YOLD(*),DYOLD(*),YNEW(*),K0(*),K1(*),K2(*),K3(*),K4(*),
     +          K5(*),K6(*),K7(*),K8(*),K9(*),IPOINT(*),YLAG(*),
     +          BVAL(*),YWORK(*),QUEUE(*),WORK(*),DYLAG(*),RPAR(*),
     +          IPAR(*),ICOMP(1)
C
      EXTERNAL DERIVS,BETA,YINIT,DYINIT
C
C     NOTE:
C     THE K'S ARE NOT MULTIPLIED BY H.
C
C     IF AN ILLEGAL DELAY IS ENCOUNTERED, THE STEPSIZE WILL BE
C     REDUCED AND THE STEP WILL BE RETRIED UP TO MAXTRY
C     TIMES.
C
C***FIRST EXECUTABLE STATEMENT  D6STP1
C
      MAXTRY = 30
C
      HTRY = H
C
      DO 1000 IHTRY = 1 , MAXTRY
C
      H = HTRY
      INDEX = 2
      TNEW = TOLD
C
C     CHECK IF THE STEPSIZE IS TOO SMALL IF THIS
C     HAS NOT ALREADY BEEN DONE.
C
      IF (IHTRY .GT. 1) THEN
         TEMP = U13*ABS(TOLD)
         IF (ABS(H) .LE. TEMP) THEN
            INDEX = -4
            CALL D6MESG(11,'D6STP1')
            GO TO 9005
         ENDIF
      ENDIF
C
C     CALCULATE K0.
C
C     SINCE  THE SIXTH ORDER METHOD IS BEING USED, DO NOT
C     RE-COMPUTE THE DERIVATIVE AT TVAL = TOLD SINCE IT
C     IS AVAILABLE IN DYOLD.
C
      CALL DCOPY(N,YOLD,1,YNEW,1)
      CALL DCOPY(N,DYOLD,1,K0,1)
C
C     CALCULATE K1.
C
      TVAL = TOLD + H / 9.0D0
      DO 120 I = 1 , N
         YNEW(I) = YOLD(I) + H * 
     *    ( K0(I) / 9.0D0 )
  120 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K1,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K2.
C
      TVAL = TOLD + H / 6.0D0
      DO 220 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (1.0D0 / 24.0D0) * K0(I)
     *    + (3.0D0 / 24.0D0) * K1(I) )
  220 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K2,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K3.
C
      TVAL = TOLD + H / 3.0D0
      DO 320 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (1.0D0 / 6.0D0) * K0(I)
     *    - (3.0D0 / 6.0D0) * K1(I)
     *    + (4.0D0 / 6.0D0) * K2(I) )
  320 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K3,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K4.
C
      TVAL = TOLD + H / 2.0D0
      DO 420 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (1.0D0 / 8.0D0) * K0(I)
     *    + (3.0D0 / 8.0D0) * K3(I) )
  420 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K4,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K5.
C
      TVAL = TOLD + H * (2.0D0 / 3.0D0)
      KFAC = 2.0D0 / 21.0D0
      DO 520 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( -4.0D0 * KFAC * K0(I)
     *    +  3.0D0 * KFAC * K1(I)
     *    + 12.0D0 * KFAC * K2(I)
     *    - 12.0D0 * KFAC * K3(I)
     *    +  8.0D0 * KFAC * K4(I) )
  520 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K5,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K6.
C
      TVAL = TOLD + H * (5.0D0 / 6.0D0)
      KFAC = (5.0D0 / 456.0D0)
      DO 620 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *                   ( 13.0D0 * KFAC * K0(I)
     *                   +  3.0D0 * KFAC * K1(I)
     *                   - 16.0D0 * KFAC * K2(I)
     *                   + 66.0D0 * KFAC * K3(I)
     *                   - 32.0D0 * KFAC * K4(I)
     *                   + 42.0D0 * KFAC * K5(I) )
  620 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K6,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K7.
C
      TVAL = TOLD + H
      DO 720 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *   ( ( 35.0D0 / 322.0D0) * K0(I)
     *   - ( 81.0D0 / 322.0D0) * K1(I)
     *   + (240.0D0 / 322.0D0) * K2(I)
     *   - (360.0D0 / 322.0D0) * K3(I)
     *   + (680.0D0 / 322.0D0) * K4(I)
     *   - (420.0D0 / 322.0D0) * K5(I)
     *   + (228.0D0 / 322.0D0) * K6(I) )
  720 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K7,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K8.
C
      TVAL = TOLD + H
      DO 820 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( ( 161.0D0 / 3000.0D0) * K0(I)
     *    + (  57.0D0 /  250.0D0) * K2(I)
     *    + (  21.0D0 /  200.0D0) * K3(I)
     *    + (  17.0D0 /   75.0D0) * K4(I)
     *    + (  21.0D0 /  200.0D0) * K5(I)
     *    + (  57.0D0 /  250.0D0) * K6(I)
     *    + ( 161.0D0 / 3000.0D0) * K7(I) )
  820 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K8,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE K9.
C
      TVAL = TOLD + H * (2.0D0 / 3.0D0)
      DO 920 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (  7.0D0 / 135.0D0) * K0(I)
     *    + ( 32.0D0 / 135.0D0) * K2(I)
     *    + ( 12.0D0 / 135.0D0) * K3(I)
     *    + ( 32.0D0 / 135.0D0) * K4(I)
     *    + (  7.0D0 / 135.0D0) * K5(I) )
  920 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      CALL D6INT1(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,DYINIT,
     +            QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,INDEX,
     +            ICHECK,IVAN,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      IF (INDEX .LT. 0) THEN
      IF (INDEX.EQ.-12) GO TO 9005
      IF (((INDEX.EQ.-11).AND.(IHTRY.LE.MAXTRY)).AND.(ICHECK.EQ.2))
     +THEN
         HTRY = 0.5D0 * HTRY
         GO TO 1000
      ENDIF
      IF (INDEX.EQ.-11) GO TO 9005
      ENDIF
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K9,RPAR,IPAR)
      IF (N.LE.0) GO TO 1040
      NFEVAL = NFEVAL + 1
C
C     CALCULATE YNEW.
C
      TVAL = TOLD + H
      CALL D6POLY(N,TOLD,YOLD,H,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            YNEW,YNEW,TVAL,IORDER,ICOMP,IPAIR,
     +            1,0,0.0D0,0.0D0,0,2)
C
C     UPDATE THE INDEPENDENT VARIABLE.
C
      TNEW = TOLD + H
C
C     THE STEP WAS SUCCESSFUL. EXIT THE RETRY LOOP.
C
      GO TO 1020
C
 1000 CONTINUE
C
C     THE STEP WAS NOT SUCCESSFUL SINCE THE HTRY LOOP
C     WAS COMPLETED.
C
      CALL D6MESG(29,'D6STP1')
      GO TO 8995
C
 1020 CONTINUE
C
      GO TO 9005
C
 1040 CONTINUE
C
C     THE USER TERMINATED THE INTEGRATION.
C
      INDEX = -14
      CALL D6MESG(25,'D6STP1')
C
      GO TO 9005
C
 8995 CONTINUE
C
C     THE STEP WAS NOT SUCCESSFUL DUE TO AN ILLEGAL DELAY.
C
      INDEX = -11
      CALL D6MESG(27,'D6STP1')
      CALL D6MESG(24,'D6STP1')
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6STP2
      SUBROUTINE D6STP2(N,NG,TOLD,YOLD,DYOLD,H,DERIVS,TNEW,INDEX,YNEW,
     +                  K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,NFEVAL,IORDER,
     +                  YLAG,DYLAG,BETA,BVAL,YWORK,YINIT,DYINIT,QUEUE,
     +                  LQUEUE,IPOINT,WORK,TINIT,ICHECK,IVAN,EXTRAP,
     +                  NUTRAL,RPAR,IPAR,ITOL,ABSERR,RELERR,ICONV,
     +                  KPRED0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +                  KPRED6,KPRED7,KPRED8,KPRED9,IPAIR)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6STP2
C***SUBSIDIARY
C***PURPOSE  The purpose of D6STP2 is to perform a single step
C            of the integration for the DKLAG6 differential
C            equation solver (for vanishing delay problems).
C            D6STP2 is called by the interval oriented driver
C            D6DRV3 if the logical variable EXTRAP is set to 
C            .TRUE. in the user provided subroutine INITAL.
C            D6STP3 extrapolates the most recent solution
C            polynomial to obtain an initial solution if a delay
C            time falls beyond the solution history queue. It
C            then performs a predictor-corrector like iteration
C            of the Runge-Kutta-Sarafyan solution polynomials
C            (unlike D6STP1, which reduces the stepsize when a
C            delay time falls beyond the solution history queue).
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
C***ROUTINES CALLED  DAXPY, DCOPY, D6BETA, D6STAT, D6MESG, D6INT3,
C                    D6POLY
C***REVISION HISTORY  (YYMMDD)
C***END PROLOGUE  D6STP2
C
C     PERFORM A RUNGE-KUTTA-SARAFYAN STEP OF SIZE H
C     FROM TOLD FOR D6DRV3
C
      IMPLICIT NONE
C
      INTEGER ICHECK,INDEX,IOK,IORDER,IPOINT,LQUEUE,N,NFEVAL,
     +        NG,IVAN,IQUEUE,IPAR,IMETH,ITOL,ICONV,IDUM,ICOR,
     +        MAXCOR,I,IFAM,INDEXO,NUMPT,IBEGIN,ICOMP,JCOMP,
     +        IPAIR
      DOUBLE PRECISION BVAL,DYOLD,H,K0,K1,K2,K3,K4,K5,K6,K7,
     +                 K8,K9,QUEUE,TINIT,TNEW,TOLD,TVAL,
     +                 WORK,YLAG,YNEW,YOLD,YWORK,DYLAG,RPAR,
     +                 ABSERR,RELERR,KPRED0,KPRED1,KPRED2,
     +                 KPRED3,KPRED4,KPRED5,KPRED6,KPRED7,
     +                 KPRED8,KPRED9,RDUM,HNM1,TNM1,
     +                 KCFAC,KRATIO, XNUMER,XDENOM,TR,KDIFF,
     +                 KFAC
      LOGICAL EXTRAP,NUTRAL, VANISH
C
      DIMENSION YOLD(*),DYOLD(*),YNEW(*),K0(*),K1(*),K2(*),K3(*),K4(*),
     +          K5(*),K6(*),K7(*),K8(*),K9(*),IPOINT(*),YLAG(*),
     +          BVAL(*),YWORK(*),QUEUE(N+1,*),WORK(*),DYLAG(*),RPAR(*),
     +          IPAR(*),ABSERR(*),RELERR(*),KPRED0(*),KPRED1(*),
     +          KPRED2(*),KPRED3(*),KPRED4(*),KPRED5(*),KPRED6(*),
     +          KPRED7(*),KPRED8(*),KPRED9(*),JCOMP(1)
C
      EXTERNAL DERIVS,BETA,YINIT,DYINIT
C
C     NOTE:
C     THE K'S ARE NOT MULTIPLIED BY H
C
C***FIRST EXECUTABLE STATEMENT  D6STP2
C
      INDEX = 2
      TNEW = TOLD
      VANISH = .FALSE.
      ICOR = 0
      ICONV = 0
C
C     DEFINE THE MAXIMUM NUMBER OF CORRECTIONS
C
      MAXCOR = 5
C
C     DEFINE THE CONVERGENCE FACTOR FOR THE PREDICTOR CORRECTOR
C     ITERATION
C
      KCFAC = 1.0D0
C
C     IF IFAM = 0, THE IMBEDDED FAMILY WILL BE USED FOR EXTRAPOLATIONS
C     DURING THE PREDICTION (ICOR = 0), IF THE DELAY TIME FALLS BEYOND
C     THE SOLUTION HISTORY QUEUE. OTHERWISE, THE MOST RECENT SUCCESSFUL
C     Y((6,10)) POLYNOMIAL WILL BE USED FOR THE EXTRAPOLATIONS DURING
C     THE PREDICTION, IF THE DELAY FALLS BEYOND THE SOLUTION HISTORY
C     QUEUE. IN BOTH CASES, THE MOST RECENT Y((6,10)) POLYNOMIAL ITERATE
C     WILL BE USED DURING THE CORRECTOR ITERATION (ICOR > 0).
C
      IFAM = 1
C
C     IF THE FIRST STEP HAS NOT BEEN COMPLETED, USE THE IMBEDDED
C     FAMILY EVEN IF IFAM = 1 ABOVE.
C
      IQUEUE = IPOINT(21)
      IF (IQUEUE .EQ. 0) IFAM = 0
C
C     THE INFORMATION FOR THE MOST RECENT SUCCESSFUL Y((6,10))
C     IS NEEDED IF IFAM = 0
C
      IF (IQUEUE .NE. 0) THEN
         INDEXO = IQUEUE
         NUMPT = LQUEUE/ (12*(N+1))
         IBEGIN = NUMPT + (INDEXO-1)*11
         HNM1 = QUEUE(2,INDEXO) - QUEUE(1,INDEXO)
         TNM1 = QUEUE(1,INDEXO)
      ENDIF
C
C     CALCULATE K0
C
C     SINCE THE SIXTH ORDER METHOD IS BEING USED, DO NOT
C     RE-COMPUTE THE DERIVATIVE AT TVAL = TOLD; IT IS
C     AVAILABLE IN DYOLD. (TRUE ALSO FOR THE FIFTH
C     ORDER METHOD)
C
      CALL DCOPY(N,YOLD,1,YNEW,1)
      CALL DCOPY(N,DYOLD,1,K0,1)
C
   20 CONTINUE
C
C     CALCULATE K1
C
      TVAL = TOLD + H / 9.0D0
      DO 120 I = 1 , N
         YNEW(I) = YOLD(I) + H * 
     *    ( K0(I) / 9.0D0 )
  120 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K1,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K1
         KRATIO = 0.0D0
         DO 140 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K1(I) - KPRED1(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K1(I)),ABS(KPRED1(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  140    CONTINUE
      ENDIF
C
C     CALCULATE K2
C
      TVAL = TOLD + H / 6.0D0
      DO 220 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (1.0D0 / 24.0D0) * K0(I)
     *    + (3.0D0 / 24.0D0) * K1(I) )
  220 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 8
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K2,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K2
         DO 240 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K2(I) - KPRED2(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K2(I)),ABS(KPRED2(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  240    CONTINUE
      ENDIF
C
C     CALCULATE K3
C
      TVAL = TOLD + H / 3.0D0
      DO 320 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (1.0D0 / 6.0D0) * K0(I)
     *    - (3.0D0 / 6.0D0) * K1(I)
     *    + (4.0D0 / 6.0D0) * K2(I) )
  320 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 7
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K3,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K3
         DO 340 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K3(I) - KPRED3(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K3(I)),ABS(KPRED3(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  340    CONTINUE
      ENDIF
C
C     CALCULATE K4
C
      TVAL = TOLD + H / 2.0D0
      DO 420 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (1.0D0 / 8.0D0) * K0(I)
     *    + (3.0D0 / 8.0D0) * K3(I) )
  420 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 6
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K4,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K4
         DO 440 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K4(I) - KPRED4(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K4(I)),ABS(KPRED4(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  440    CONTINUE
      ENDIF
C
C     CALCULATE K5
C
      TVAL = TOLD + H * (2.0D0 / 3.0D0)
      KFAC = 2.0D0 / 21.0D0
      DO 520 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( -4.0D0 * KFAC * K0(I)
     *    +  3.0D0 * KFAC * K1(I)
     *    + 12.0D0 * KFAC * K2(I)
     *    - 12.0D0 * KFAC * K3(I)
     *    +  8.0D0 * KFAC * K4(I) )
  520 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 5
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K5,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K5
         DO 540 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K5(I) - KPRED5(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K5(I)),ABS(KPRED5(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  540    CONTINUE
      ENDIF
C
C     CALCULATE K6
C
      TVAL = TOLD + H * (5.0D0 / 6.0D0)
      KFAC = (5.0D0 / 456.0D0)
      DO 620 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *                   ( 13.0D0 * KFAC * K0(I)
     *                   +  3.0D0 * KFAC * K1(I)
     *                   - 16.0D0 * KFAC * K2(I)
     *                   + 66.0D0 * KFAC * K3(I)
     *                   - 32.0D0 * KFAC * K4(I)
     *                   + 42.0D0 * KFAC * K5(I) )
  620 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 4
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K6,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K6
         DO 640 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K6(I) - KPRED6(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K6(I)),ABS(KPRED6(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  640    CONTINUE
      ENDIF
C
C     CALCULATE K7
C
      TVAL = TOLD + H
      DO 720 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *   ( ( 35.0D0 / 322.0D0) * K0(I)
     *   - ( 81.0D0 / 322.0D0) * K1(I)
     *   + (240.0D0 / 322.0D0) * K2(I)
     *   - (360.0D0 / 322.0D0) * K3(I)
     *   + (680.0D0 / 322.0D0) * K4(I)
     *   - (420.0D0 / 322.0D0) * K5(I)
     *   + (228.0D0 / 322.0D0) * K6(I) )
  720 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 6
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K7,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K7
         DO 740 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K7(I) - KPRED7(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K7(I)),ABS(KPRED7(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  740    CONTINUE
      ENDIF
C
C     CALCULATE K8
C
      TVAL = TOLD + H
      DO 820 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( ( 161.0D0 / 3000.0D0) * K0(I)
     *    + (  57.0D0 /  250.0D0) * K2(I)
     *    + (  21.0D0 /  200.0D0) * K3(I)
     *    + (  17.0D0 /   75.0D0) * K4(I)
     *    + (  21.0D0 /  200.0D0) * K5(I)
     *    + (  57.0D0 /  250.0D0) * K6(I)
     *    + ( 161.0D0 / 3000.0D0) * K7(I) )
  820 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 5
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K8,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K8
         DO 840 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K8(I) - KPRED8(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K8(I)),ABS(KPRED8(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  840    CONTINUE
      ENDIF
C
C     CALCULATE K9
C
      TVAL = TOLD + H * (2.0D0 / 3.0D0)
      DO 920 I = 1 , N
         YNEW(I) = YOLD(I) + H *
     *    ( (  7.0D0 / 135.0D0) * K0(I)
     *    + ( 32.0D0 / 135.0D0) * K2(I)
     *    + ( 12.0D0 / 135.0D0) * K3(I)
     *    + ( 32.0D0 / 135.0D0) * K4(I)
     *    + (  7.0D0 / 135.0D0) * K5(I) )
  920 CONTINUE
      CALL BETA(N,TVAL,YNEW,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      CALL D6BETA(N,TVAL,H,BVAL,IOK)
      IF (IOK.EQ.1) GO TO 8995
      IF (ICOR .EQ. 0 .AND. IFAM .NE. 0) THEN
C        USE THE MOST RECENT SUCCESSFUL Y((6,10)) POLYNOMIAL
C        TO EXTRAPOLATE
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,HNM1,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               QUEUE(1,IBEGIN+3),QUEUE(1,IBEGIN+3),
     +               QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +               QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +               QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +               QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +               TNM1,QUEUE(1,IBEGIN+1),VANISH)
      ENDIF
      IF (ICOR .EQ. 0 .AND. IFAM .EQ. 0) THEN
C        USE THE IMBEDDED FAMILY TO EXTRAPOLATE
C        IMETH = 4
         IMETH = 9
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,K0,
     +               K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD,YOLD,VANISH)
      ENDIF
      IF (ICOR .GT. 0) THEN
C        USE THE MOST RECENT Y((6,10)) ITERATE TO CORRECT
         IMETH = 1
         CALL D6INT3(N,NG,TVAL,YNEW,H,BVAL,YWORK,YLAG,DYLAG,YINIT,
     +               DYINIT,QUEUE,LQUEUE,IPOINT,WORK,TINIT,EXTRAP,
     +               NUTRAL,INDEX,ICHECK,IVAN,IMETH,RPAR,IPAR,
     +               K0,KPRED1,KPRED2,KPRED3,KPRED4,KPRED5,
     +               KPRED6,KPRED7,KPRED8,KPRED9,TOLD,YOLD,VANISH)
         VANISH = .FALSE.
      ENDIF
      IF (N.LE.0) GO TO 3000
      IF (INDEX.EQ.-12) GO TO 9005
      CALL DERIVS(N,TVAL,YNEW,YLAG,DYLAG,K9,RPAR,IPAR)
      IF (N.LE.0) GO TO 3000
      NFEVAL = NFEVAL + 1
      IF (ICOR .GT. 0) THEN
C        FORCE CONVERGENCE OF H * K9
         DO 940 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K9(I) - KPRED9(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP) * ABS(H)
     +              * MAX(ABS(K9(I)),ABS(KPRED9(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
  940    CONTINUE
      ENDIF
C
C     CALCULATE YNEW.
C
      TVAL = TOLD + H
      CALL D6POLY(N,TOLD,YOLD,H,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            YNEW,YNEW,TVAL,IORDER,JCOMP,IPAIR,1,0,0.0D0,
     +            0.0D0,0,3)
C
      IF (ICOR .GT. 0) THEN
C
C        FORCE CONVERGENCE OF YNEW
C
         DO 1220 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = YNEW(I) - KPRED0(I)
            XNUMER = ABS(H) * ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP)
     +              * MAX(ABS(YNEW(I)),ABS(KPRED0(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
 1220    CONTINUE
C
C        FORCE CONVERGENCE OF DYNEW IF THIS IS A NEUTRAL PROBLEM

         IF (.NOT. NUTRAL) GO TO 1240
         DO 1230 I = 1 , N
            ICOMP = I
            IF (ITOL .EQ. 1) ICOMP = 1
            KDIFF = K8(I) - KPRED8(I)
            XNUMER = ABS(KDIFF)
            XDENOM = ABSERR(ICOMP) + RELERR(ICOMP)
     +              * MAX(ABS(K8(I)),ABS(KPRED8(I)))
            IF (XDENOM .LE. 0.0D0) XDENOM = 1.0D0
            TR = XNUMER / XDENOM
            KRATIO = MAX(KRATIO,TR)
 1230    CONTINUE
 1240    CONTINUE
      ENDIF
C
      IF ((ICOR .EQ. 0) .AND. (.NOT. VANISH)) THEN
C
C        A CORRECTION WILL NOT BE PERFORMED SINCE NO
C        EXTRAPOLATIONS WERE PERFORMED USING D6INT9
C
         IDUM = ICOR
         CALL D6STAT(N, 5, H, RPAR, IPAR, IDUM, RDUM)
C
         GO TO 1280
C
      ENDIF
C
C     TEST FOR CONVERGENCE OF THE DERIVATIVE APPROXIMATIONS
C
      IF (ICOR .GT. 0) THEN
C
         IF (KRATIO .LE. KCFAC) THEN
C
            IDUM = ICOR
            CALL D6STAT(N, 5, H, RPAR, IPAR, IDUM, RDUM)
C
C           DO NOT PERFORM ANY FURTHER CORRECTIONS
C
            GO TO 1280
C
         ELSE
C
            IF (ICOR .EQ. MAXCOR) THEN
C
C              REDUCE THE STEPSIZE AND TRY AGAIN
C
               ICONV = 1
C
               IDUM = ICOR
               CALL D6STAT(N, 6, H, RPAR, IPAR, IDUM, RDUM)
C
               GO TO 9005
C
            ENDIF
C
         ENDIF
C
      ENDIF
C
C     PREPARE TO DO A CORRECTION
C
      ICOR = ICOR + 1
C
      IF (ICOR .LE. MAXCOR) THEN
         DO 1260 I = 1 , N
            KPRED0(I)  = YNEW(I)
            KPRED1(I)  = K1(I) 
            KPRED2(I)  = K2(I) 
            KPRED3(I)  = K3(I) 
            KPRED4(I)  = K4(I) 
            KPRED5(I)  = K5(I) 
            KPRED6(I)  = K6(I) 
            KPRED7(I)  = K7(I) 
            KPRED8(I)  = K8(I) 
            KPRED9(I)  = K9(I) 
 1260    CONTINUE
C
C        AND DO IT
C
         GO TO 20
C
      ENDIF
C
 1280 CONTINUE
C
C     UPDATE THE INDEPENDENT VARIABLE
C
      TNEW = TOLD + H
C
C     THE STEP WAS SUCCESSFUL
C
      GO TO 9005
C
 3000 CONTINUE
C
C     THE USER TERMINATED THE INTEGRATION
C
      INDEX = -14
      CALL D6MESG(25,'D6STP2')
      GO TO 9005
C
 8995 CONTINUE
C
C     THE STEP WAS NOT SUCCESSFUL DUE TO AN ILLEGAL DELAY
C
      INDEX = -11
      CALL D6MESG(27,'D6STP2')
      CALL D6MESG(24,'D6STP2')
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6INT1
      SUBROUTINE D6INT1(N,NG,T,Y,HNEXT,BVAL,YWORK,YLAG,DYLAG,
     +                  YINIT,DYINIT,QUEUE,LQUEUE,IPOINT,WORK,
     +                  TINIT,EXTRAP,INDEX,ICHECK,IVAN,RPAR,IPAR)
C
C     SUBROUTINE D6INT1(N,NG,T,Y,HNEXT,BVAL,YWORK,YLAG,DYLAG,
C    +                  YINIT,DYINIT,QUEUE,LQUEUE,IPOINT,WORK,
C    +                  TINIT,EXTRAP,INDEX,ICHECK,IVAN,RPAR,IPAR)
C
C***BEGIN PROLOGUE  D6INT1
C***PURPOSE  The purpose of D6INT1 is to interpolate the past
C            solution history queue for the DKLAG6 differential
C            equation solver.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT1.
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
C     The following parameters must be input to D6INT1:
C
C     N       = number of differential equations
C
C     NG      = total number of root functions (including those
C               corresponding to IROOT=0 in subroutine INITAL)
C               Note:
C               NG = IPOINT(30) where IPOINT is described below.
C
C     T       = current value of the independent variable
C
C     Y       = solution at time T (T and Y will usually be
C               an input parameter to the subroutine which
C               calls D6INT1 to perform the interpolation.)
C
C     HNEXT   =  1.0d0 if the integration goes left to right
C             = -1.0d0 if the integration goes right to left
C
C     BVAL(I) = delay time at which the Ith components of the
C               solution and derivative are to be interpolated,
C               I = 1,...,N
C
C     YWORK   = scratch work array of length N
C
C     YINIT   = external function to calculate the initial
C               delayed solution
C
C     DYINIT  = external function to calculate the initial
C               delayed derivative
C
C     QUEUE   = QUEUE array of length LQUEUE for DKLAG6
C
C     LQUEUE  = length of QUEUE array
C
C     IPOINT  = integer pointer work array for DKLAG6 (starts in
C               the same location as the IW array for DKLAG6)
C
C     WORK    = work array for DKLAG6 (starts in the same location
C               as DW(1+13*N) where  DW is the work array for
C               DKLAG6 (Note: (QUEUE, LQUEUE, IPOINT, and WORK
C               may be communicated to the calling subroutine
C               via COMMON.)
C
C     TINIT   = starting time for the problem
C
C     EXTRAP  = logical method flag (same as in INITAL)
C
C     ICHECK  = DKLAG6 integration flag. Use ICHECK = 1.
C
C     IVAN    = DKLAG6 integration flag. Use IVAN = 0.
C
C     RPAR(*), IPAR(*) = user arrays passed via DKLAG6
C
C     The following will be output by D6INT1:
C
C     YVAL(*)  - The delayed solution at time BVAL(I) will be
C                returned in YVAL(I), I = 1,...,N.
C
C     DYVAL(*) - The derivative of the delayed solution at time
C                BVAL(I) will be returned in DYVAL(I), I = 1,...,N.
C
C                Note:
C                If BVAL(I) = T, DYVAL(I) will not be calculated.
C
C     INDEX    - =   2 if D6INT1 was successful
C                  -14 if the user terminates the integration
C
C     Note:
C     If it is known that the values of BVAL(*) do not fall outside
C     the interval spanned by the the history queue, a second
C     interpolation subroutine D6INT2, with a simpler calling list,
C     may be used in lieu of D6INT1. The call sequence for D6INT2
C     is as follows (where the parameters have the same meaning as
C     for D6INT1):
C
C     Note:
C     Refer to the documentation prologue for D6INT2 for a
C     description of the situations in which D6INT2 or D6INT1
C     cannot perform requested interpolations.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH, D6INT6
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT1
C
C     CALCULATE THE DELAYED SOLUTION VECTOR YLAG AND
C     THE DELAYED DERIVATIVE VECTOR.
C
C     CAUTION:
C     A BIG NUMBER IS LOADED FOR ANY COMPONENT OF DYLAG
C     FOR WHICH BETA(T,Y) = T.
C
C     CAUTION:
C     D6INT1 ASSUMES BETA WAS CALLED BEFORE IT IS CALLED.
C
      IMPLICIT NONE
C
      INTEGER I,ICHECK,INDEX,IQUEUE,LQUEUE,N,NG,IVAN,IPOINT,IPAR
      DOUBLE PRECISION BVAL,HNEXT,QUEUE,T,TINIT,TVAL,WORK,Y,YLAG,
     +                 DYLAG,YWORK,D1MACH,RPAR
      LOGICAL EXTRAP
C
      DIMENSION BVAL(*),YLAG(*),QUEUE(N+1,*),IPOINT(*),WORK(*),Y(*),
     +          YWORK(*),DYLAG(*),RPAR(*),IPAR(*)
C
      EXTERNAL YINIT,DYINIT
C
C***FIRST EXECUTABLE STATEMENT  D6INT1
C
      INDEX = 2
C
      IF (HNEXT.EQ.0.0D0) THEN
          INDEX = -4
          CALL D6MESG(11,'D6INT1')
          GO TO 9005
      ENDIF
C
C     IPOINT POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
      IQUEUE = IPOINT(21)
C
C     IF WE HAVE NOT COMPLETED THE FIRST STEP, USE THE
C     INITIAL DELAY INTERVAL INFORMATION ....
C
      IF (IQUEUE.EQ.0) THEN
C
          DO 20 I = 1,N
C
              TVAL = BVAL(I)
C
C             DO NOT INTERPOLATE IF THE DELAY VANISHES.
C
              IF (TVAL.EQ.T) THEN
                  YLAG(I) = Y(I)
C                 LOAD A BIG NUMBER IN DYLAG(I).
                  DYLAG(I) = D1MACH(2)
                  GO TO 20
              ENDIF
C
C             ERROR IF THE DELAY DOES NOT FALL IN THE
C             INITIAL INTERVAL (AND IS NOT EQUAL TO
C             CURRENT TIME T).
C
              IF ((TVAL.GT.TINIT) .AND. (HNEXT.GT.0.0D0) .OR.
     +            (TVAL.LT.TINIT) .AND. (HNEXT.LT.0.0D0)) THEN
                  INDEX = -11
                  IF (.NOT. EXTRAP)
     +            CALL D6MESG(28,'D6INT1')
                  GO TO 9005
              ENDIF
C
C             SOLUTION ...
C
              CALL YINIT(N,TVAL,YLAG(I),I,RPAR,IPAR)
              IF (N.LE.0) GO TO 9005
C
C             DERIVATIVE ...
C
              CALL DYINIT(N,TVAL,DYLAG(I),I,RPAR,IPAR)
              IF (N.LE.0) GO TO 9005
C
   20     CONTINUE
          GO TO 9005
      ENDIF
C
C     ELSE INTERPOLATE USING THE HISTORY QUEUE ....
C
      CALL D6INT6(N,NG,T,Y,BVAL,YWORK,YINIT,DYINIT,QUEUE,LQUEUE,
     +            IPOINT,YLAG,DYLAG,HNEXT,WORK,INDEX,ICHECK,
     +            IVAN,EXTRAP,RPAR,IPAR)
C
 9005 CONTINUE
C
      IF (N.LE.0) THEN
         INDEX = -14
         CALL D6MESG(25,'D6INT1')
      ENDIF
C
      RETURN
      END
C*DECK D6INT2
      SUBROUTINE D6INT2(N,NG,BVAL,YWORK,QUEUE,LQUEUE,IPOINT,
     +                  YLAG,DYLAG,HNEXT,WORK,INDEX,ICHECK,
     +                  IVAN,EXTRAP)
C
C     SUBROUTINE D6INT2(N,NG,BVAL,YWORK,QUEUE,LQUEUE,IPOINT,
C    +                  YLAG,DYLAG,HNEXT,WORK,INDEX,ICHECK,
C    +                  IVAN,EXTRAP)
C
C***BEGIN PROLOGUE  D6INT2
C***PURPOSE  The purpose of D6INT2 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT2.
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
C     The following parameters must be input to D6INT2:
C
C     N       = number of differential equations
C
C     NG      = total number of root functions (including those
C               corresponding to IROOT=0 in subroutine INITAL)
C               Note:
C               NG = IPOINT(30) where IPOINT is described below.
C
C     BVAL(I) = delay time at which the Ith components of the
C               solution and derivative are to be interpolated,
C               I = 1,...,N
C
C     YWORK   = scratch work array of length N
C
C     QUEUE   = QUEUE array of length LQUEUE for DKLAG6
C
C     LQUEUE  = length of QUEUE array
C
C     IPOINT  = integer pointer work array for DKLAG6 (starts in
C               the same location as the IW array for DKLAG6)
C
C     WORK    = work array for DKLAG6 (starts in the same location
C               as DW(1+13*N) where  DW is the work array for
C               DKLAG6 (Note: (QUEUE, LQUEUE, IPOINT, and WORK
C               may be communicated to the calling subroutine
C               via COMMON.)
C
C     ICHECK  = DKLAG6 integration flag. Use ICHECK = 1.
C
C     IVAN    = DKLAG6 integration flag. Use IVAN = 0.
C
C     EXTRAP  = logical method flag (same as in INITAL)
C
C     The following will be output by D6INT2:
C
C     YVAL(*)  - The delayed solution at time BVAL(I) will be
C                returned in YVAL(I), I = 1,...,N.
C
C     DYVAL(*) - The derivative of the delayed solution at time
C                BVAL(I) will be returned in DYVAL(I), I = 1,...,N.
C
C     INDEX    - = 2 if D6INT2 was successful
C
C     Error Conditions:
C
C     The following describes the situations in which D6INT2
C     (and D6INT1) will generate an error message when called
C     by a user subroutine. Denote by S the value of time
C     at which an interpolation for the solution or its
C     derivative is desired. An error will occur in the
C     following situations.
C
C     (1) S is less than the initial integration time. This error
C         can be avoided by checking if this is the case and
C         calling YINIT and/or DYINIT instead of calling D6INT2.
C         Alternatively, use D6INT1 which checks for this
C         condition and, if necessary, calls YINIT and DYINIT.
C
C     (2) The circular solution history queue is updated when
C         it is full by overwriting the oldest information in
C         queue with the most recent. Thus, the oldest information
C         is discarded when necessary. If S corresponds to an
C         interval for which the solution has been discarded,
C         the interpolation cannot be performed. This error can
C         be avoided by using a history queue which is large
C         enough to hold all of the history solution.
C
C     (3) Suppose the integration has been completed successfully
C         to T and the integrator attempts to step to T+H. If
C         D6INT2 is called with S > T, the interpolation will not
C         be performed since there is not enough information in
C         the history queue to accommodate it.
C
C     (4) Suppose the integrator steps successfully for T to T+H
C         and locates a zero of an event function at TROOT in
C         (T,T+H). Before updating the history queue, it then
C         repeats the step from T to TROOT (in order to force the
C         zero to be an integration meshpoint). When the step is
C         repeated, if S > T (even if S < T+H), the interpolation
C         cannot be performed, since the history queue has not yet
C         updated.
C
C     (5) There are some problems for which D6INT2 and D6INT1 are
C         not applicable. They cannot be used a problem such as
C
C                               y'(t) = y(y(t/2))
C
C         with an initial integration time of 0.0. Regardless of
C         how small an initial stepsize HINIT is attempted, when
C         say D6INT2 is called from BETA, it will need to approximate
C         the solution for values of time such as HINIT/2. An
C         error message will occur since HINIT/2 > 0. This is in
C         contrast to the use of DKLAG6 for problems such as
C
C                               y'(t) = y(t/2) .
C
C         DKLAG6 can solve such problems (if EXTRAP = .TRUE. in
C         subroutine INITAL) since it uses special methods to get
C         started, and therefore D6INT2 is not needed for such problems.
C
C         Caution:
C         If D6INT2 or D6INT1 returns a value of INDEX other than
C         2, one of these conditions has arisen. In this case, set
C         N = 0 before returning to DKLAG6 in order to terminate
C         the integration and decide what changes need to be made.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D6POLY, D6PNTR, D6SRC2, D6SRC3, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT2
C
C     INTERPOLATE THE HISTORY QUEUE.
C
      IMPLICIT NONE
C
      INTEGER I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IBEGIN,ICHECK,IDIR,IERRUS,IFIRST,IFOUND,
     +        INDEX,INDEXO,INEW,IOLD,IORDER,IPOINT,IQUEUE,RSTRT,
     +        IVAL,JQUEUE,KQUEUE,LQUEUE,N,NG,NUMPT,IVAN,IK7,IK8,IK9,
     +        IPAIR,ICOMP
      DOUBLE PRECISION BVAL,DELINT,HNEXT,QUEUE,TINIT,TN,TO,TVAL,WORK,
     +                 YLAG,YWORK,DYLAG
      LOGICAL EXTRAP
C
      DIMENSION QUEUE(N+1,*),IPOINT(*),BVAL(*),YWORK(*),YLAG(*),WORK(*),
     +          DYLAG(*),ICOMP(1)
C
      EXTERNAL D6SRC2,D6SRC3
C
C***FIRST EXECUTABLE STATEMENT  D6INT2
C
      INDEX = 2
C
      IF (HNEXT.EQ.0.0D0) THEN
          INDEX = -4
          CALL D6MESG(11,'D6INT2')
          GO TO 9005
      ENDIF
C
C     IQUEUE POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
      IQUEUE = IPOINT(21)
      JQUEUE = IPOINT(22)
      KQUEUE = IPOINT(23)
C
C     NUMPT IS THE MAXIMUM NUMBER OF POINTS FOR WHICH
C     HISTORY DATA CAN BE STORED ....
C
      NUMPT = LQUEUE/ (12*(N+1))
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
      DO 20 IVAL = 1,N
C
C         DEFINE THE DELAYED TIME FOR THIS COMPONENT.
C
          TVAL = BVAL(IVAL)
C
C         INTERPOLATE USING THE SOLUTION HISTORY QUEUE.
C
          IOLD = JQUEUE
          INEW = IQUEUE
C
C         CHECK THAT TVAL FALLS WITHIN THE INTERVAL SPANNED
C         BY THE HISTORY QUEUE
C
          IF (HNEXT .GT. 0.0D0) THEN
             IF (TVAL.LT.QUEUE(1,IOLD) .OR. TVAL.GT.QUEUE(2,INEW))
     +       THEN
               INDEX = -11
               CALL D6MESG(35,'D6INT2')
               GO TO 9005
             ENDIF
          ENDIF
          IF (HNEXT .LT. 0.0D0) THEN
             IF (TVAL.GT.QUEUE(1,IOLD) .OR. TVAL.LT.QUEUE(2,INEW))
     +       THEN
               INDEX = -11
               CALL D6MESG(35,'D6INT2')
               GO TO 9005
             ENDIF
          ENDIF
C
C         CALL THE SEARCH ROUTINE TO LOCATE THE OLD SLUG
C         OF DATA TO BE INTERPOLATED FOR EACH VALUE
C         OF T-BETA.
C
          IF (.NOT. EXTRAP)
     +    CALL D6SRC2(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX,ICHECK)
C
          IF (EXTRAP)
     +    CALL D6SRC3(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX,ICHECK,IVAN)
C
C         CHECK FOR ERROR RETURN.
C
          IF (INDEX.EQ.-11) GO TO 9005
          IF (INDEX.EQ.-12) GO TO 9005
C
C         THE DATA TO BE INTERPOLATED CORRESPONDS TO
C         TO = TOLD, TN = TNEW. INDEXO IS THE POINTER
C         FOR THE CORRESPONDING SLUG OF DATA. (THE DATA
C         FOR TOLD TO BE INTERPOLATED STARTS IN QUEUE
C         LOCATION (1,IBEGIN+1).)
C
          IBEGIN = NUMPT + (INDEXO-1)*11
C
C         DEFINE THE STEPSIZE CORRESPONDING TO THE
C         INTERPOLATION DATA.
C
          DELINT = TN - TO
C
C         INTERPOLATE THE DATA. THE YWORK VECTOR AND QUEUE(*,IBEGIN+2)
C         ARE USED FOR SCRATCH STORAGE.
C
          CALL D6POLY(N,TO,QUEUE(1,IBEGIN+1),DELINT,QUEUE(1,IBEGIN+3),
     +                QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +                QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +                QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +                QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +                YWORK,QUEUE(1,IBEGIN+2),TVAL,IORDER,
     +                ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,4)
C
          YLAG(IVAL) = YWORK(IVAL)
C
          DYLAG(IVAL) = QUEUE(IVAL,IBEGIN+2)
C
   20 CONTINUE
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6INT3
      SUBROUTINE D6INT3(N,NG,T,Y,HNEXT,BVAL,YWORK,YLAG,DYLAG,
     +                  YINIT,DYINIT,QUEUE,LQUEUE,IPOINT,WORK,
     +                  TINIT,EXTRAP,NUTRAL,INDEX,ICHECK,IVAN,
     +                  IMETH,RPAR,IPAR,K0,K1,K2,K3,K4,K5,K6,
     +                  K7,K8,K9,TOLD,YOLD,VANISH)
C
C***BEGIN PROLOGUE  D6INT3
C***PURPOSE  The purpose of D6INT3 is to interpolate the past
C            solution history queue for the DKLAG6 differential
C            equation solver.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT3.
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
C     The following parameters must be input to D6INT3:
C
C     N       = number of differential equations
C
C     NG      = total number of root functions (including those
C               corresponding to IROOT=0 in subroutine INITAL)
C               Note:
C               NG = IPOINT(30) where IPOINT is described below.
C
C     T       = current value of the independent variable
C
C     Y       = solution at time T (T and Y will usually be
C               input parameters to the subroutine which
C               calls D6INT3 to perform the interpolation).
C
C     HNEXT   =  1.0d0 if the integration goes left to right
C             = -1.0d0 if the integration goes right to left
C
C     BVAL(I) = delay time at which the Ith components of the
C               solution and derivative are to be interpolated,
C               I = 1,...,N
C
C     YWORK   = scratch work array of length N
C
C     YINIT   = external function to calculate the initial
C               delayed solution
C
C     DYINIT  = external function to calculate the initial
C               delayed derivative
C
C     QUEUE   = QUEUE array of length LQUEUE for DKLAG6
C
C     LQUEUE  = length of QUEUE array
C
C     IPOINT  = integer pointer work array for DKLAG6 (starts in
C               the same location as the IW array for DKLAG6)
C
C     WORK    = work array for DKLAG6 (starts in the same location
C               as DW(1+13*N) where  DW is the work array for
C               DKLAG6 (Note: (QUEUE, LQUEUE, IPOINT, and WORK
C               may be communicated to the calling subroutine
C               via COMMON.)
C
C     TINIT   = starting time for the problem
C
C     EXTRAP  = logical method flag (same as in INITAL)
C
C     NUTRAL  = logical problem type flag (same as in INITAL)
C
C     ICHECK  = DKLAG6 integration flag. Use ICHECK = 1.
C
C     IVAN    = DKLAG6 integration flag. Use IVAN = 0.
C
C     IMETH   = flag to indicate which member of the imbedded
C               family of methods to use for extrapolation
C
C     RPAR(*), IPAR(*) = user arrays passed via DKLAG6
C
C     The following will be output by D6INT3:
C
C     YVAL(*)  - The delayed solution at time BVAL(I) will be
C                returned in YVAL(I), I = 1,...,N.
C
C     DYVAL(*) - The derivative of the delayed solution at time
C                BVAL(I) will be returned in DYVAL(I), I = 1,...,N.
C
C     VANISH     = .TRUE. IF SUBROUTINE D6INT9 IS CALLED
C
C     INDEX    - =   2 if D6INT3 was successful
C                  -14 if the user terminates the integration
C
C     Note:
C     Unlike D6INT1 which does does not supply an approximation
C     to the delayed derivative in the event the delay time is
C     equal to the current time, D6INT3 extrapolates to obtain
C     such an approximation.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH, D6INT7
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT3
C
C     CALCULATE THE DELAYED SOLUTION VECTOR YLAG AND
C     THE DELAYED DERIVATIVE VECTOR.
C
C     CAUTION:
C     AN EXTRAPOLATION IS PERFORMED TO APPROXIMATE ANY
C     COMPONENT OF DYLAG FOR WHICH BETA(T,Y) = T.
C
C     CAUTION:
C     D6INT3 ASSUMES BETA WAS CALLED BEFORE IT IS CALLED.
C
      IMPLICIT NONE
C
      INTEGER I,ICHECK,INDEX,IQUEUE,LQUEUE,N,NG,IVAN,IPOINT,IPAR,
     +        IMETH,IPAIR,IORDER
      DOUBLE PRECISION BVAL,HNEXT,QUEUE,T,TINIT,TVAL,WORK,Y,YLAG,
     +                 DYLAG,YWORK,RPAR,K0,K1,K2,K3,K4,K5,K6,
     +                 K7,K8,K9,TOLD,YOLD
C     DOUBLE PRECISION D1MACH
      LOGICAL EXTRAP,VANISH,NUTRAL
C
      DIMENSION BVAL(*),YLAG(*),QUEUE(N+1,*),IPOINT(*),WORK(*),Y(*),
     +          YWORK(*),DYLAG(*),RPAR(*),IPAR(*),K0(*),K1(*),K2(*),
     +          K3(*),K4(*),K5(*),K6(*),K7(*),K8(*),K9(*),YOLD(*)
C
      EXTERNAL YINIT,DYINIT
C
C***FIRST EXECUTABLE STATEMENT  D6INT3
C
      INDEX = 2
C
      IF (HNEXT.EQ.0.0D0) THEN
          INDEX = -4
          CALL D6MESG(11,'D6INT3')
          GO TO 9005
      ENDIF
C
C     IPAIR INDICATES WHICH METHOD PAIR TO USE.
C
      IPAIR  = IPOINT(34)
C
C     IORDER IS THE ORDER OF THE PRIMARY METHOD.
C
      IORDER = IPOINT(17)
C
C     IQUEUE POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
      IQUEUE = IPOINT(21)
C
C     IF WE HAVE NOT COMPLETED THE FIRST STEP, USE THE
C     INITIAL DELAY INTERVAL INFORMATION ....
C
      IF (IQUEUE.EQ.0) THEN
C
          DO 20 I = 1,N
C
              TVAL = BVAL(I)
C
              IF (TVAL.EQ.T) THEN
C                 EXTRAPOLATE FOR DYLAG.
                  CALL D6INT9(N,TOLD,YOLD,HNEXT,K0,K1,K2,K3,K4,K5,K6,
     +                        K7,K8,K9,BVAL,I,YLAG,DYLAG,IMETH,IPAIR,
     +                        IORDER)
                  IF (NUTRAL) VANISH = .TRUE.
C                 DO NOT INTERPOLATE FOR YLAG.
                  YLAG(I) = Y(I)
                  GO TO 20
              ENDIF
C
C             IF THE DELAY DOES NOT FALL IN THE INITIAL INTERVAL
C             AND IS NOT EQUAL TO CURRENT TIME T, EXTRAPOLATE USING
C             THE INDICATED METHOD OF THE IMBEDDED FAMILY
C
              IF ((TVAL.GT.TINIT) .AND. (HNEXT.GT.0.0D0) .OR.
     +            (TVAL.LT.TINIT) .AND. (HNEXT.LT.0.0D0)) THEN
                  CALL D6INT9(N,TOLD,YOLD,HNEXT,K0,K1,K2,K3,K4,K5,K6,
     +                        K7,K8,K9,BVAL,I,YLAG,DYLAG,IMETH,IPAIR,
     +                        IORDER)
                  VANISH = .TRUE.
              ELSE
C                 SOLUTION ...
                  CALL YINIT(N,TVAL,YLAG(I),I,RPAR,IPAR)
                  IF (N.LE.0) GO TO 9005
C                 DERIVATIVE ...
                  CALL DYINIT(N,TVAL,DYLAG(I),I,RPAR,IPAR)
                  IF (N.LE.0) GO TO 9005
              ENDIF
   20     CONTINUE
          GO TO 9005
      ENDIF
C
C     ELSE INTERPOLATE USING THE HISTORY QUEUE ....
C
      CALL D6INT7(N,NG,T,Y,BVAL,YWORK,YINIT,DYINIT,QUEUE,LQUEUE,
     +            IPOINT,YLAG,DYLAG,HNEXT,WORK,INDEX,ICHECK,
     +            IVAN,EXTRAP,RPAR,IPAR,TOLD,YOLD,K0,K1,K2,
     +            K3,K4,K5,K6,K7,K8,K9,IMETH,VANISH)
C
 9005 CONTINUE
C
      IF (N.LE.0) THEN
         INDEX = -14
         CALL D6MESG(25,'D6INT3')
      ENDIF
C
      RETURN
      END
C*DECK D6INT4
      SUBROUTINE D6INT4(N,QUEUE,LQUEUE,IPOINT,WORK,YWORK,HNEXT,
     +                  BVAL,YLAG,DYLAG,INDEX)
C
C     This differs from D6INT2 in the following ways. Some
C     unnecessary (if called by the user) flags are deleted
C     from the parameter list. This routine also allows the
C     solution to be extrapolated beyond the interval spanned
C     by the solution history queue. This subroutine is not
C     called by DKLAG6.
C
C***BEGIN PROLOGUE  D6INT4
C***PURPOSE  The purpose of D6INT4 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver. This is a modification
C            of D6INT2 which allows extrapolations beyond the
C            interval spanned by the solution history queue.
C            Note:
C            Refer to the "NONSTANDARD INTERPOLATION" section of
C            the prologue for DKLAG6 for a description of the
C            usage of D6INT2.
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
C     The following parameters must be input to D6INT4:
C
C     N       = number of differential equations
C
C     QUEUE   = QUEUE array of length LQUEUE for DKLAG6
C
C     LQUEUE  = length of QUEUE array
C
C     IPOINT  = integer pointer work array for DKLAG6 (starts in
C               the same location as the IW array for DKLAG6)
C
C     WORK    = work array for DKLAG6 (starts in the same location
C               as DW(1+13*N) where  DW is the work array for
C               DKLAG6) Note:QUEUE, LQUEUE, IPOINT, and WORK
C               may be communicated to the calling subroutine
C               via COMMON.
C
C     YWORK   = scratch work array of length N
C
C     HNEXT   = any positive number if the integration is
C               going left to right and any negative number
C               if the integration is going right to left
C
C     BVAL(I) = delay time at which the Ith components of the
C               solution and derivative are to be interpolated,
C               I = 1,...,N
C
C     The following will be output by D6INT4:
C
C     YVAL(*)  - If INDEX > -1, the delayed solution at time BVAL(I)
C                will be returned in YVAL(I), I = 1,...,N.
C
C     DYVAL(*) - If INDEX > -1, the derivative of the delayed solution
C                at time BVAL(I) will be returned in DYVAL(I),
C                I = 1,...,N.
C
C     INDEX    - =  2 if D6INT4 was successful and the times at
C                     which interpolations were requested fell
C                     within the interval spanned by the solution
C                     history queue.
C                =  1 D6INT4 allowed an extrapolation of the most
C                     recent solution polynomial.
C                =  0 D6INT4 allowed an extrapolation of the oldest
C                     solution polynomial.
C                = -1 No interpolation or extrapolation was
C                     performed since HNEXT = 0.0D0.
C                = -2 No interpolation or extrapolation was
C                     performed since this call was made before
C                     the completion of the first integration
C                     step.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D6POLY, D6PNTR, D6SRC4, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT4
C
      IMPLICIT NONE
C
      INTEGER I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IBEGIN,IDIR,IERRUS,IFIRST,IFOUND,IORDER,
     +        INDEX,INDEXO,INEW,IOLD,IPOINT,IQUEUE,RSTRT,ICOMP,
     +        IVAL,JQUEUE,KQUEUE,LQUEUE,N,NG,NUMPT,IK7,IK8,IK9,
     +        IPAIR
      DOUBLE PRECISION BVAL,DELINT,HNEXT,QUEUE,TINIT,TN,TO,TVAL,WORK,
     +                 YLAG,YWORK,DYLAG
C
      DIMENSION QUEUE(N+1,*),IPOINT(*),BVAL(*),YWORK(*),YLAG(*),WORK(*),
     +          DYLAG(*),ICOMP(1)
C
      EXTERNAL D6SRC4
C
C***FIRST EXECUTABLE STATEMENT  D6INT4
C
      INDEX = 2
      NG = IPOINT(30)
C
      IF (HNEXT.EQ.0.0D0) THEN
          INDEX = -1
          CALL D6MESG(11,'D6INT4')
          GO TO 9005
      ENDIF
C
C     IQUEUE POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
      IQUEUE = IPOINT(21)
      JQUEUE = IPOINT(22)
      KQUEUE = IPOINT(23)
C
C     RETURN IF THE FIRST INTEGRATION STEP HAS NOT BEEN COMPLETED.
C
      IF (IQUEUE .EQ. 0) THEN
         INDEX = -2
         GO TO 9005
      ENDIF
C
C     NUMPT IS THE MAXIMUM NUMBER OF POINTS FOR WHICH
C     HISTORY DATA CAN BE STORED ....
C
      NUMPT = LQUEUE/ (12*(N+1))
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
      DO 20 IVAL = 1,N
C
C         DEFINE THE DELAYED TIME FOR THIS COMPONENT.
C
          TVAL = BVAL(IVAL)
C
C         INTERPOLATE USING THE SOLUTION HISTORY QUEUE.
C
          IOLD = JQUEUE
          INEW = IQUEUE
C
C         CALL THE SEARCH ROUTINE TO LOCATE THE OLD SLUG
C         OF DATA TO BE INTERPOLATED FOR EACH VALUE
C         OF T-BETA.
C
          CALL D6SRC4(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX)
C
C         THE DATA TO BE INTERPOLATED CORRESPONDS TO
C         TO = TOLD, TN = TNEW. INDEXO IS THE POINTER
C         FOR THE CORRESPONDING SLUG OF DATA. (THE DATA
C         FOR TOLD TO BE INTERPOLATED STARTS IN QUEUE
C         LOCATION (1,IBEGIN+1).)
C
          IBEGIN = NUMPT + (INDEXO-1)*11
C
C         DEFINE THE STEPSIZE CORRESPONDING TO THE
C         INTERPOLATION DATA.
C
          DELINT = TN - TO
C
C         INTERPOLATE THE DATA. THE YWORK VECTOR AND QUEUE(*,IBEGIN+2)
C         ARE USED FOR SCRATCH STORAGE.
C
          CALL D6POLY(N,TO,QUEUE(1,IBEGIN+1),DELINT,QUEUE(1,IBEGIN+3),
     +                QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +                QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +                QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +                QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +                YWORK,QUEUE(1,IBEGIN+2),TVAL,IORDER,
     +                ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,5)
C
          YLAG(IVAL) = YWORK(IVAL)
C
          DYLAG(IVAL) = QUEUE(IVAL,IBEGIN+2)
C
   20 CONTINUE
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6INT5
      SUBROUTINE D6INT5(N,QUEUE,LQUEUE,IPOINT,WORK,YWORK,HNEXT,
     +                  BVAL,YLAG,DYLAG,INDEX,ICOMP)
C
C     This differs from D6INT4 only in that it allows selected
C     solution components, as dictated by array ICOMP, to be
C     interpolated rather than the entire solution array.
C     This subroutine is not called by DKLAG6.
C
C***BEGIN PROLOGUE  D6INT5
C***PURPOSE  The purpose of D6INT5 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver. This is a modification
C            of D6INT4 which allows extrapolations beyond the
C            interval spanned by the solution history queue. In
C            addition, only selected components are interpolated,
C            as dictated by array ICOMP, rather than the entire
C            solution array.
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
C     The following parameters must be input to D6INT5:
C
C     N       = number of differential equations
C
C     QUEUE   = QUEUE array of length LQUEUE for DKLAG6
C
C     LQUEUE  = length of QUEUE array
C
C     IPOINT  = integer pointer work array for DKLAG6 (starts in
C               the same location as the IW array for DKLAG6)
C
C     WORK    = work array for DKLAG6 (starts in the same location
C               as DW(1+13*N) where  DW is the work array for
C               DKLAG6) Note:QUEUE, LQUEUE, IPOINT, and WORK
C               may be communicated to the calling subroutine
C               via COMMON.
C
C     YWORK   = scratch work array of length N
C
C     HNEXT   = any positive number if the integration is
C               going left to right and any negative number
C               if the integration is going right to left
C
C     BVAL(I) = delay time at which the Ith components of the
C               solution and derivative are to be interpolated,
C               I = 1,...,N
C
C     ICOMP(I) = 0 if an interpolated value for solution
C                component I is desired.
C
C     The following will be output by D6INT5:
C
C     YVAL(*)  - If INDEX > -1, the delayed solution at time BVAL(I)
C                will be returned in YVAL(I), if ICOMP(I) = 0,
C                I = 1,...,N.
C
C     DYVAL(*) - If INDEX > -1, the derivative of the delayed solution
C                at time BVAL(I) will be returned in DYVAL(I), if
C                ICOMP(I) = 0, I = 1,...,N.
C
C     INDEX    - =  2 if D6INT5 was successful and the times at
C                     which interpolations were requested fell
C                     within the interval spanned by the solution
C                     history queue.
C                =  1 D6INT5 allowed an extrapolation of the most
C                     recent solution polynomial.
C                =  0 D6INT5 allowed an extrapolation of the oldest
C                     solution polynomial.
C                = -1 No interpolation or extrapolation was
C                     performed since HNEXT = 0.0D0.
C                = -2 No interpolation or extrapolation was
C                     performed since this call was made before
C                     the completion of the first integration
C                     step.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D6POLY, D6PNTR, D6SRC4, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT5
C
      IMPLICIT NONE
C
      INTEGER I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IBEGIN,IDIR,IERRUS,IFIRST,IFOUND,
     +        INDEX,INDEXO,INEW,IOLD,IORDER,IPOINT,IQUEUE,RSTRT,
     +        IVAL,JQUEUE,KQUEUE,LQUEUE,N,NG,NUMPT,ICOMP,IK7,IK8,IK9,
     +        IPAIR
      DOUBLE PRECISION BVAL,DELINT,HNEXT,QUEUE,TINIT,TN,TO,TVAL,WORK,
     +                 YLAG,YWORK,DYLAG
C
      DIMENSION QUEUE(N+1,*),IPOINT(*),BVAL(*),YWORK(*),YLAG(*),WORK(*),
     +          DYLAG(*),ICOMP(*)
C
      EXTERNAL D6SRC4
C
C***FIRST EXECUTABLE STATEMENT  D6INT5
C
      INDEX = 2
      NG = IPOINT(30)
C
      IF (HNEXT.EQ.0.0D0) THEN
          INDEX = -1
          CALL D6MESG(11,'D6INT5')
          GO TO 9005
      ENDIF
C
C     IQUEUE POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
      IQUEUE = IPOINT(21)
      JQUEUE = IPOINT(22)
      KQUEUE = IPOINT(23)
C
C     RETURN IF THE FIRST INTEGRATION STEP HAS NOT BEEN COMPLETED.
C
      IF (IQUEUE .EQ. 0) THEN
         INDEX = -2
         GO TO 9005
      ENDIF
C
C     NUMPT IS THE MAXIMUM NUMBER OF POINTS FOR WHICH
C     HISTORY DATA CAN BE STORED ....
C
      NUMPT = LQUEUE/ (12*(N+1))
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
      DO 20 IVAL = 1,N
C
C         DO NOT INTERPOLATE FOR THIS SOLUTION UNLESS
C         ICOMP(IVAL) = 0.
C
          IF (ICOMP(IVAL) .NE. 0) GO TO 20
C
C         DEFINE THE DELAYED TIME FOR THIS COMPONENT.
C
          TVAL = BVAL(IVAL)
C
C         INTERPOLATE USING THE SOLUTION HISTORY QUEUE.
C
          IOLD = JQUEUE
          INEW = IQUEUE
C
C         CALL THE SEARCH ROUTINE TO LOCATE THE OLD SLUG
C         OF DATA TO BE INTERPOLATED FOR EACH VALUE
C         OF T-BETA.
C
          CALL D6SRC4(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX)
C
C         THE DATA TO BE INTERPOLATED CORRESPONDS TO
C         TO = TOLD, TN = TNEW. INDEXO IS THE POINTER
C         FOR THE CORRESPONDING SLUG OF DATA. (THE DATA
C         FOR TOLD TO BE INTERPOLATED STARTS IN QUEUE
C         LOCATION (1,IBEGIN+1).)
C
          IBEGIN = NUMPT + (INDEXO-1)*11
C
C         DEFINE THE STEPSIZE CORRESPONDING TO THE
C         INTERPOLATION DATA.
C
          DELINT = TN - TO
C
C         INTERPOLATE THE DATA. THE YWORK VECTOR AND QUEUE(*,IBEGIN+2)
C         ARE USED FOR SCRATCH STORAGE.
C
          CALL D6POLY(N,TO,QUEUE(1,IBEGIN+1),DELINT,QUEUE(1,IBEGIN+3),
     +                QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +                QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +                QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +                QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +                YWORK,QUEUE(1,IBEGIN+2),TVAL,IORDER,
     +                ICOMP,IPAIR,-3,0,0.0D0,0.0D0,0,6)
C
          YLAG(IVAL) = YWORK(IVAL)
C
          DYLAG(IVAL) = QUEUE(IVAL,IBEGIN+2)
C
   20 CONTINUE
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6INT6
      SUBROUTINE D6INT6(N,NG,T,Y,BVAL,YWORK,YINIT,DYINIT,QUEUE,
     +                  LQUEUE,IPOINT,YLAG,DYLAG,HNEXT,WORK,
     +                  INDEX,ICHECK,IVAN,EXTRAP,RPAR,IPAR)
C
C***BEGIN PROLOGUE  D6INT6
C***SUBSIDIARY
C***PURPOSE  The purpose of D6INT6 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver.
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
C***ROUTINES CALLED  D6POLY, D6PNTR, D6SRC2, D6SRC3, D1MACH, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT6
C
C     INTERPOLATE THE HISTORY QUEUE FOR D6DRV3.
C
      IMPLICIT NONE
C
      INTEGER I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IBEGIN,ICHECK,IDIR,IERRUS,IFIRST,IFOUND,
     +        INDEX,INDEXO,INEW,IOLD,IORDER,IPOINT,IQUEUE,RSTRT,
     +        IVAL,JQUEUE,KQUEUE,LQUEUE,N,NG,NUMPT,IVAN,IPAR,IOK,
     +        IK7,IK8,IK9,IPAIR,ICOMP
      DOUBLE PRECISION BVAL,DELINT,HNEXT,QUEUE,T,TINIT,TN,TO,TVAL,WORK,
     +                 Y,YLAG,YWORK,DYLAG,D1MACH,RPAR
      LOGICAL EXTRAP
C
      DIMENSION QUEUE(N+1,*),IPOINT(*),BVAL(*),YWORK(*),YLAG(*),WORK(*),
     +          Y(*),DYLAG(*),RPAR(*),IPAR(*),ICOMP(1)
C
      EXTERNAL YINIT,DYINIT
C
      EXTERNAL D6SRC2,D6SRC3
C
C***FIRST EXECUTABLE STATEMENT  D6INT6
C
      INDEX = 2
C
C     IQUEUE POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
      IQUEUE = IPOINT(21)
      JQUEUE = IPOINT(22)
      KQUEUE = IPOINT(23)
C
C     NUMPT IS THE MAXIMUM NUMBER OF POINTS FOR WHICH
C     HISTORY DATA CAN BE STORED ....
C
      NUMPT = LQUEUE/ (12*(N+1))
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
      DO 20 IVAL = 1,N
C
C         DEFINE THE DELAYED TIME FOR THIS COMPONENT.
C
          CALL D6BETA(N,T,HNEXT,BVAL,IOK)
C
          TVAL = BVAL(IVAL)
C
C         ERROR IF THE STEPSIZE EXCEEDS THE DELAY.
C
          IF (IOK.EQ.1) THEN
C         IF ((TVAL.GT.T) .AND. (HNEXT.GT.0.0D0) .OR.
C    +        (TVAL.LT.T) .AND. (HNEXT.LT.0.0D0)) THEN
              INDEX = -11
              CALL D6MESG(24,'D6INT6')
              GO TO 9005
          ENDIF
C
C         DO NOT INTERPOLATE IF THE DELAY VANISHES.
C
          IF (TVAL.EQ.T) THEN
              YLAG(IVAL) = Y(IVAL)
C             BIG NUMBER FOR THE DERIVATIVE ...
              DYLAG(IVAL) = D1MACH(2)
              GO TO 20
          ENDIF
C
C         INTERPOLATE USING THE SOLUTION HISTORY QUEUE.
C
          IOLD = JQUEUE
          INEW = IQUEUE
C
C         USE YINIT OR CALL THE SEARCH ROUTINE TO LOCATE THE
C         OLD SLUG OF DATA TO BE INTERPOLATED FOR EACH VALUE
C         OF T-BETA.
C
C         USE YINIT AND DYINIT IF TVAL FALLS IN THE INITIAL
C         DELAY INTERVAL ....
C
          IF ((HNEXT.GT.0.0D0) .AND. (TVAL.GT.TINIT)) GO TO 10
          IF ((HNEXT.LT.0.0D0) .AND. (TVAL.LT.TINIT)) GO TO 10
          CALL YINIT(N,TVAL,YLAG(IVAL),IVAL,RPAR,IPAR)
          IF (N.LE.0) GO TO 40
          CALL DYINIT(N,TVAL,DYLAG(IVAL),IVAL,RPAR,IPAR)
          IF (N.LE.0) GO TO 40
          GO TO 20
C
C         OTHERWISE USE THE HISTORY QUEUE ....
C
   10     CONTINUE
C
          IF (.NOT. EXTRAP)
     +    CALL D6SRC2(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX,ICHECK)
C
          IF (EXTRAP)
     +    CALL D6SRC3(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX,ICHECK,IVAN)
C
C         CHECK FOR ERROR RETURN.
C
          IF (INDEX.EQ.-11) GO TO 9005
          IF (INDEX.EQ.-12) GO TO 9005
C
C         THE DATA TO BE INTERPOLATED CORRESPONDS TO
C         TO = TOLD, TN = TNEW. INDEXO IS THE POINTER
C         FOR THE CORRESPONDING SLUG OF DATA. (THE DATA
C         FOR TOLD TO BE INTERPOLATED STARTS IN QUEUE
C         LOCATION (1,IBEGIN+1).)
C
          IBEGIN = NUMPT + (INDEXO-1)*11
C
C         DEFINE THE STEPSIZE CORRESPONDING TO THE
C         INTERPOLATION DATA.
C
          DELINT = TN - TO
C
C         INTERPOLATE THE DATA. THE YWORK VECTOR AND QUEUE(*,IBEGIN+2)
C         ARE USED FOR SCRATCH STORAGE.
C
          CALL D6POLY(N,TO,QUEUE(1,IBEGIN+1),DELINT,QUEUE(1,IBEGIN+3),
     +                QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +                QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +                QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +                QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +                YWORK,QUEUE(1,IBEGIN+2),TVAL,IORDER,
     +                ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,7)
C
          YLAG(IVAL) = YWORK(IVAL)
C
          DYLAG(IVAL) = QUEUE(IVAL,IBEGIN+2)
C
   20 CONTINUE
C
   40 CONTINUE
C
C     CHECK IF THE USER TERMINATED THE INTEGRATION.
C
      IF (N.LE.0) THEN
         INDEX = -14
         CALL D6MESG(25,'D6INT6')
      ENDIF
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6INT7
      SUBROUTINE D6INT7(N,NG,T,Y,BVAL,YWORK,YINIT,DYINIT,QUEUE,
     +                  LQUEUE,IPOINT,YLAG,DYLAG,HNEXT,WORK,
     +                  INDEX,ICHECK,IVAN,EXTRAP,RPAR,IPAR,
     +                  TOLD,YOLD,K0,K1,K2,K3,K4,K5,K6,K7,
     +                  K8,K9,IMETH,VANISH)
C
C***BEGIN PROLOGUE  D6INT7
C***SUBSIDIARY
C***PURPOSE  The purpose of D6INT7 is to search and interpolate
C            the past solution history queue for the DKLAG6
C            differential equation solver.
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
C***ROUTINES CALLED  D6POLY, D6PNTR, D6SRC2, D6SRC3, D1MACH, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT7
C
C     INTERPOLATE THE HISTORY QUEUE FOR D6DRV3.
C
      IMPLICIT NONE
C
      INTEGER I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IBEGIN,ICHECK,IDIR,IERRUS,IFIRST,IFOUND,
     +        INDEX,INDEXO,INEW,IOLD,IORDER,IPOINT,IQUEUE,RSTRT,
     +        IVAL,JQUEUE,KQUEUE,LQUEUE,N,NG,NUMPT,IVAN,IPAR,IMETH,
     +        IK7,IK8,IK9,IPAIR,ICOMP
      DOUBLE PRECISION BVAL,DELINT,HNEXT,QUEUE,T,TINIT,TN,TO,TVAL,WORK,
     +                 Y,YLAG,YWORK,DYLAG,RPAR,TOLD,YOLD,K0,K1,K2,K3,
     +                 K4,K5,K6,K7,K8,K9
C     DOUBLE PRECISION D1MACH
      LOGICAL EXTRAP,VANISH
C
      DIMENSION QUEUE(N+1,*),IPOINT(*),BVAL(*),YWORK(*),YLAG(*),WORK(*),
     +          Y(*),DYLAG(*),RPAR(*),IPAR(*),YOLD(*),K0(*),K1(*),K2(*),
     +          K3(*),K4(*),K5(*),K6(*),K7(*),K8(*),K9(*),ICOMP(1)
C
      EXTERNAL YINIT,DYINIT
C
      EXTERNAL D6SRC2,D6SRC3
C
C***FIRST EXECUTABLE STATEMENT  D6INT7
C
      INDEX = 2
C
C     IQUEUE POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
      IQUEUE = IPOINT(21)
      JQUEUE = IPOINT(22)
      KQUEUE = IPOINT(23)
C
C     NUMPT IS THE MAXIMUM NUMBER OF POINTS FOR WHICH
C     HISTORY DATA CAN BE STORED ....
C
      NUMPT = LQUEUE/ (12*(N+1))
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
      DO 20 IVAL = 1,N
C
C         DEFINE THE DELAYED TIME FOR THIS COMPONENT.
C
          TVAL = BVAL(IVAL)
C
          IF (TVAL.EQ.T) THEN
C             EXTRAPOLATE FOR DYLAG.
              CALL D6INT9(N,TOLD,YOLD,HNEXT,K0,K1,K2,K3,K4,K5,K6,
     +                    K7,K8,K9,BVAL,IVAL,YLAG,DYLAG,IMETH,
     +                    IPAIR,IORDER)
              VANISH = .TRUE.
C             DO NOT INTERPOLATE FOR YLAG.
              YLAG(IVAL) = Y(IVAL)
              GO TO 20
          ENDIF
C
C         EXTRAPOLATE IF THE STEPSIZE EXCEEDS THE DELAY.
C
          IF ((TVAL.GT.TOLD) .AND. (HNEXT.GT.0.0D0) .OR.
     +        (TVAL.LT.TOLD) .AND. (HNEXT.LT.0.0D0)) THEN
              CALL D6INT9(N,TOLD,YOLD,HNEXT,K0,K1,K2,K3,K4,K5,K6,
     +                    K7,K8,K9,BVAL,IVAL,YLAG,DYLAG,IMETH,
     +                    IPAIR,IORDER)
              VANISH = .TRUE.
              GO TO 20
          ENDIF
C
C         INTERPOLATE USING THE SOLUTION HISTORY QUEUE.
C
          IOLD = JQUEUE
          INEW = IQUEUE
C
C         USE YINIT OR CALL THE SEARCH ROUTINE TO LOCATE THE
C         OLD SLUG OF DATA TO BE INTERPOLATED FOR EACH VALUE
C         OF T-BETA.
C
C         USE YINIT AND DYINIT IF TVAL FALLS IN THE INITIAL
C         DELAY INTERVAL
C
          IF ((HNEXT.GT.0.0D0) .AND. (TVAL.GT.TINIT)) GO TO 10
          IF ((HNEXT.LT.0.0D0) .AND. (TVAL.LT.TINIT)) GO TO 10
          CALL YINIT(N,TVAL,YLAG(IVAL),IVAL,RPAR,IPAR)
          IF (N.LE.0) GO TO 40
          CALL DYINIT(N,TVAL,DYLAG(IVAL),IVAL,RPAR,IPAR)
          IF (N.LE.0) GO TO 40
          GO TO 20
C
C         OTHERWISE USE THE HISTORY QUEUE ....
C
   10     CONTINUE
C
          IF (.NOT. EXTRAP)
     +    CALL D6SRC2(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX,ICHECK)
C
          IF (EXTRAP)
     +    CALL D6SRC3(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TO,TN,
     +                INDEX,ICHECK,IVAN)
C
C         CHECK FOR ERROR RETURN.
C
          IF (INDEX.EQ.-11) GO TO 9005
          IF (INDEX.EQ.-12) GO TO 9005
C
C         THE DATA TO BE INTERPOLATED CORRESPONDS TO
C         TO = TOLD, TN = TNEW. INDEXO IS THE POINTER
C         FOR THE CORRESPONDING SLUG OF DATA. (THE DATA
C         FOR TOLD TO BE INTERPOLATED STARTS IN QUEUE
C         LOCATION (1,IBEGIN+1).)
C
          IBEGIN = NUMPT + (INDEXO-1)*11
C
C         DEFINE THE STEPSIZE CORRESPONDING TO THE
C         INTERPOLATION DATA.
C
          DELINT = TN - TO
C
C         INTERPOLATE THE DATA. THE YWORK VECTOR AND QUEUE(*,IBEGIN+2)
C         ARE USED FOR SCRATCH STORAGE.
C
          CALL D6POLY(N,TO,QUEUE(1,IBEGIN+1),DELINT,QUEUE(1,IBEGIN+3),
     +                QUEUE(1,IBEGIN+8),QUEUE(1,IBEGIN+4),
     +                QUEUE(1,IBEGIN+5),QUEUE(1,IBEGIN+6),
     +                QUEUE(1,IBEGIN+7),QUEUE(1,IBEGIN+9),
     +                QUEUE(1,IBEGIN+10),QUEUE(1,IBEGIN+11),
     +                YWORK,QUEUE(1,IBEGIN+2),TVAL,IORDER,
     +                ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,8)
C
          YLAG(IVAL) = YWORK(IVAL)
C
          DYLAG(IVAL) = QUEUE(IVAL,IBEGIN+2)
C
   20 CONTINUE
C
   40 CONTINUE
C
C     CHECK IF THE USER TERMINATED THE INTEGRATION.
C
      IF (N.LE.0) THEN
         INDEX = -14
         CALL D6MESG(25,'D6INT7')
      ENDIF
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6INT8
      SUBROUTINE D6INT8(N,NG,TOLD,YOLD,TNEW,WORK,TOUT,YOUT,DYOUT,IPOINT)
C
C***BEGIN PROLOGUE  D6INT8
C***SUBSIDIARY
C***PURPOSE  The purpose of D6INT8 is to evaluate the solution
C            polynomial and the derivative polynomial for the
C            DKLAG6 differential equation solver.
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
C***ROUTINES CALLED  D6POLY, D6PNTR
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT8
C
C     INTERPOLATE THE SOLUTION AND DERIVATIVE AT TOUT.
C
C     THE INPUT PARAMETERS N, NG, TOLD, YOLD, DYOLD, TNEW, AND
C     WORK ARE OUTPUT FROM THE LAST CALL TO D6DRV3.  TOUT
C     IS THE VALUE OF THE INDEPENDENT VARIABLE AT WHICH
C     THE SOLUTION AND DERIVATIVES ARE TO BE INTERPOLATED.
C     THE INTERPOLATED SOLUTION IS RETURNED IN YOUT. THE
C     INTERPOLATED DERIVATIVES ARE RETURNED IN DYOUT.
C
      IMPLICIT NONE
C
      INTEGER I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IDIR,IERRUS,IFIRST,IFOUND,IORDER,
     +        IPOINT,IQUEUE,RSTRT,JQUEUE,KQUEUE,LQUEUE,N,NG,
     +        IK7,IK8,IK9,IPAIR,ICOMP
      DOUBLE PRECISION DELINT,TINIT,TOLD,TNEW,TOUT,YOLD,YOUT,DYOUT,WORK
C
      DIMENSION YOLD(*),YOUT(*),DYOUT(*),WORK(*),IPOINT(*),ICOMP(1)
C
C     RESTORE THE POINTERS AND INTEGRATION ORDER.
C
C***FIRST EXECUTABLE STATEMENT  D6INT8
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
C     DEFINE THE STEPSIZE THAT WAS USED.
C
      DELINT = TNEW - TOLD
C
C     DO THE INTERPOLATION.
C
      CALL D6POLY(N,TOLD,YOLD,DELINT,WORK(I1),WORK(I3),WORK(I4),
     +            WORK(I5),WORK(I6),WORK(I7),WORK(IK7),WORK(IK8),
     +            WORK(IK9),YOUT,DYOUT,TOUT,IORDER,ICOMP,IPAIR,
     +            3,0,0.0D0,0.0D0,0,9)
C
      RETURN
      END
C*DECK D6INT9
      SUBROUTINE D6INT9(N,TOLD,YOLD,H,K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,
     +                  BVAL,IVAL,YVAL,DYVAL,IMETH,IPAIR,IORDER)
C
C***AUTHOR  THOMPSON, S., RADFORD UNIVERSITY, RADFORD, VIRGINIA.
C***BEGIN PROLOGUE  D6INT9
C***SUBSIDIARY
C***PURPOSE  The purpose of D6INT9 is to evaluate the solution
C            polynomial and the derivative polynomial for the
C            DKLAG6 differential equation solver using one of
C            the members of a family of imbedded methods of
C            orders 1,...,6.
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
C***ROUTINES CALLED  D6POLY
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6INT9
C
C     APPROXIMATE THE SOLUTION AND DERIVATIVE USING A MEMBER
C     OF THE FAMILY OF IMBEDDED METHODS.
C
      IMPLICIT NONE
C
      INTEGER IVAL,IMETH,ICOMP,IORDER,N,IPAIR
      DOUBLE PRECISION H,DYVAL,K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,
     +                 TOLD,TVAL,C,YOLD,YVAL,Z,BVAL
C
      DIMENSION K0(*),K1(*),K2(*),K3(*),K4(*),K5(*),K6(*),
     +          YOLD(*),YVAL(*),DYVAL(*),BVAL(*),
     +          K7(*),K8(*),K9(*),ICOMP(1)
C
C     NOTE:
C     THE K'S ARE NOT MULTIPLIED BY H.
C
C***FIRST EXECUTABLE STATEMENT  D6INT9
C
      TVAL = BVAL(IVAL)
C
      IF (IMETH .EQ. 9) GO TO 900
C
      CALL D6POLY(N,TOLD,YOLD,H,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            YVAL,DYVAL,TVAL,IORDER,ICOMP,IPAIR,3,0,
     +            0.0D0,0.0D0,IVAL,10)
C
      GO TO 1000
C
  900 CONTINUE
C
C     FIRST ORDER POLYNOMIAL Y(1) (EQ. 38a)
C
      Z = TVAL - TOLD
      C = Z / H
C
C     YVAL(IVAL) = YOLD(IVAL)
C
      YVAL(IVAL) = YOLD(IVAL) + Z*K0(IVAL)
C
C     DYVAL(IVAL) = K0(IVAL)
C
      DYVAL(IVAL) = K0(IVAL)
C
 1000 CONTINUE
C
      RETURN
      END
C*DECK D6HST1
      SUBROUTINE D6HST1(DF,NEQ,A,B,Y,YPRIME,ETOL,MORDER,SMALL,
     +                  BIG,SPY,PV,YP,SF,BVAL,YLAG,DYLAG,BETA,
     +                  YINIT,DYINIT,INDEX,H,IPAIR,RPAR,IPAR)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6HST1
C***SUBSIDIARY
C***PURPOSE  The function of subroutine D6HST1 is to compute a
C            starting stepsize to be used in solving initial
C            value problems in delay differential equations.
C            It is based on a modification of DHSTRT from the
C            SLATEC library. It is called by subroutine D6DRV2.
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
C     Subroutine D6HST1 computes a starting stepsize to be used
C     by the DKLAG6 delay differential equation solver. D6HST1 is
C     based on an estimate of the local Lipschitz constant for the
C     differential equation (lower bound on a norm of the Jacobian),
C     a bound on the differential equation first derivative), and a
C     bound on the partial derivative of the equation with respect
C     to the independent variable (all approximated near the initial
C     point A).
C
C     Subroutine D6HST1 uses a function subprogram D6VEC3 for computing
C     a vector norm. The maximum norm is presently utilized though it
C     can easily be replaced by any other vector norm. It is presumed
C     that any replacement norm routine would be carefully coded to
C     prevent unnecessary underflows or overflows from occurring, and
C     also, would not alter the vector or number of components.
C
C  On input you must provide the following:
C
C      DF -- This is a subroutine of the form
C                      DF(NEQ,X,Y,YLAG,DYLAG,YPRIME)
C             which defines the system of first order differential
C             equations to be solved. For the given values of X, the
C             solution vector Y at time X, the delayed solution vector
C             YLAG corresponding to times X - BVAL(*), and the delayed
C             derivative vector DYLAG corresponding to times X-BVAL(*),
C             the subroutine must evaluate the NEQ components of the
C             system of differential equations
C                     DY/DX = DF(NEQ,X,Y,YLAG,DYLAG)
C             and store the derivatives in the array YPRIME(*), that
C             is, YPRIME(I) = * DY(I)/DX *  for equations I=1,...,NEQ.
C
C             Subroutine DF must not alter X, Y(*), YLAG(*), or
C             DYLAG(*). You must declare the name DF in an external
C             statement in your program that calls D5HST1. You must
C             dimension Y, YLAG, DYLAG, and YPRIME in DF.
C
C      NEQ -- This is the number of (first order) differential equations
C             to be integrated.
C
C      A -- This is the initial point of integration.
C
C      B -- This is a value of the independent variable used to define
C             the direction of integration. A reasonable choice is to
C             set B to the first point at which a solution is desired.
C             You can also use B, if necessary, to restrict the length
C             of the first integration step because the algorithm will
C             not compute a starting step length which is bigger than
C             ABS(B-A), unless B has been chosen too close to A.
C             (it is presumed that D6HST1 has been called with B
C             different from A on the machine being used. Also see the
C             discussion about the parameter SMALL.)
C
C      Y(*) -- This is the vector of initial values of the NEQ solution
C             components at the initial point A.
C
C      YPRIME(*) -- This is the vector of derivatives of the NEQ
C             solution components at the initial point A.
C             (defined by the differential equations in subroutine DF)
C
C      ETOL -- This is the vector of error tolerances corresponding to
C             the NEQ solution components. It is assumed that all
C             elements are positive. Following the first integration
C             step, the tolerances are expected to be used by the
C             integrator in an error test which roughly requires that
C                        ABS(LOCAL ERROR)  .LE.  ETOL
C             for each vector component. For consistency with the error
C             control in DKLAG6, it is recommended that ETOL be
C                ETOL(I) = 0.5D0 * (ABSERR + RELERR * ABS(Y(I)))
C             where ABSERR and RELERR are the absolute and relative
C             error tolerances for DKLAG6.
C
C      MORDER -- This is the order of the formula which will be used by
C             the initial value method for taking the first integration
C             step. MORDER should be 6 for DKLAG6.
C
C      SMALL -- This is a small positive machine dependent constant
C             which is used for protecting against computations with
C             numbers which are too small relative to the precision of
C             floating point arithmetic. SMALL should be set to
C             (approximately) the smallest positive DOUBLE PRECISION
C             number such that  (1.+SMALL) .GT. 1.  on the machine being
C             used. The quantity  SMALL**(3/8)  is used in computing
C             increments of variables for approximating derivatives by
C             differences. Also the algorithm will not compute a
C             starting step length which is smaller than
C             100*SMALL*ABS(A).
C
C      BIG -- This is a large positive machine dependent constant which
C             is used for preventing machine overflows. A reasonable
C             choice is to set big to (approximately) the square root of
C             the largest DOUBLE PRECISION number which can be held in
C             the machine.
C
C      SPY(*),PV(*),YP(*),SF(*),BVAL(*),YLAG(*),DYLAG(*) -- These are
C             DOUBLE PRECISION work arrays of length NEQ which provide
C             the routine with needed storage space.
C
C      BETA, YINIT, DYINIT -- These are the corresponding three EXTERNAL
C             subroutines for DKLAG6. Refer to the documentation for
C             DKLAG6 for a description of these subroutines.
C
C      RPAR(*),IPAR(*) -- These are user arrays passed via DKLAG6.
C
C  On Output  (after the return from D6HST1),
C
C      H -- is an appropriate starting stepsize to be attempted by
C             the differential equation method if INDEX = 0.
C
C           Caution:
C           If the initial function YLAG is not compatible with
C           the delay differential equation, then a derivative
C           discontinuity may exist at X = A.
C
C      INDEX -- error flag. If INDEX = 0, D6HST1 was successful.
C           Otherwise an error was encountered. In the latter
C           case, H does not contains an appropriate starting
C           stepsize.
C
C           Possible error returns:
C
C           INDEX = -1 means a delay time was encountered which
C                      was beyond the corresponding time
C
C           INDEX = -2 means B - A is equal to 0.0D0. (This will
C                      not happen if D6HST1 is called from
C                      D6DRV2.)
C
C           INDEX =-14 means the integration was terminated
C                      in one of the user subroutines called
C                      by D6HST1
C
C           All input parameters in the call list remain unchanged
C           except for the working arrays SPY(*),PV(*),YP(*),
C           SF(*),YLAG(*), and DYLAG(*).
C
C     Note:
C     If a delay time falls beyond the initial point (e.g.,
C     y'(t) = y(t/2)), D6HST1 will use a family of imbedded
C     methods to approximate the delayed solution and delayed
C     derivative. Otherwise it will use subroutines YINIT and
C     DYINIT.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D6VEC3, D6BETA, D6HST2, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6HST1
C
      IMPLICIT NONE
C
      INTEGER J, K, LK, MIN0, MORDER, NEQ, INDEX, IOK,
     +        IPAR, IPAIR
      INTEGER IMETH, IORDER
      DOUBLE PRECISION A, ABSDX, B, BIG, DA, DELF, DELY,
     +      DFDUB, DFDXB, D6VEC3,
     +      DX, DY, ETOL, FBND, H, PV, RELPER, SF, SMALL, SPY,
     +      SRYDPB, TOLEXP, TOLMIN, TOLP, TOLSUM, Y, YDPB, YP, YPRIME,
     +      BVAL, YLAG, DYLAG, TVAL, RPAR
      DIMENSION Y(NEQ),YPRIME(NEQ),ETOL(NEQ),SPY(NEQ),PV(NEQ),YP(NEQ),
     +          SF(NEQ),BVAL(NEQ),YLAG(NEQ),DYLAG(NEQ),
     +          RPAR(*), IPAR(*)
      EXTERNAL DF, BETA, YINIT, DYINIT
C
C     ..................................................................
C
C     BEGIN BLOCK PERMITTING ...EXITS TO 160
C
C***FIRST EXECUTABLE STATEMENT  D6HST1
C
         INDEX = 0
         DX = B - A
         IF (DX .EQ. 0.0D0) THEN
            INDEX = -2 
            GO TO 8990
         ENDIF
         ABSDX = ABS(DX)
         RELPER = SMALL**0.375D0
C
C        ...............................................................
C
C             COMPUTE AN APPROXIMATE BOUND (DFDXB) ON THE PARTIAL
C             DERIVATIVE OF THE EQUATION WITH RESPECT TO THE
C             INDEPENDENT VARIABLE. PROTECT AGAINST AN OVERFLOW.
C             ALSO COMPUTE A BOUND (FBND) ON THE FIRST DERIVATIVE
C             LOCALLY.
C
         DA = SIGN(MAX(MIN(RELPER*ABS(A),ABSDX),
     +                    100.0D0*SMALL*ABS(A)),DX)
         IF (DA .EQ. 0.0D0) DA = RELPER*DX
         CALL BETA(NEQ,A+DA,Y,BVAL,RPAR,IPAR)
         IF (NEQ.LE.0) GO TO 8985
         CALL D6BETA(NEQ,A+DA,DX,BVAL,IOK)
         IF (IOK .NE. 0) THEN
            INDEX = -1
            GO TO 8995
         ENDIF
C
         TVAL = A + DA
C
         IMETH = 9
         CALL D6HST2(NEQ,Y,DA,INDEX,YLAG,DYLAG,BVAL,YINIT,DYINIT,
     +               A,A,RPAR,IPAR,YPRIME,YPRIME,YPRIME,YPRIME,
     +               YPRIME,YPRIME,YPRIME,YPRIME,YPRIME,YPRIME,
     +               IMETH,IPAIR,IORDER)
         IF (NEQ.LE.0) GO TO 8985
C
C        DO 5 J = 1, NEQ
C           TVAL = BVAL(J)
C              IF ((TVAL.LE.A.AND.B.GT.A)  .OR.
C    +             (TVAL.GE.A.AND.B.LT.A)) THEN
C                  CALL YINIT(NEQ,TVAL,YLAG(J),J,RPAR,IPAR)
C                  IF (NEQ.LE.0) GO TO 8985
C                  CALL DYINIT(NEQ,TVAL,DYLAG(J),J,RPAR,IPAR)
C                  IF (NEQ.LE.0) GO TO 8985
C              ELSE
C                  YLAG(J) = Y(J)
C                  DYLAG(J) = YPRIME(J)
C              ENDIF
C   5    CONTINUE
C
         INDEX = 0
         CALL DF(NEQ,TVAL,Y,YLAG,DYLAG,SF,RPAR,IPAR)
         IF (NEQ.LE.0) GO TO 8985
         DO 10 J = 1, NEQ
            YP(J) = SF(J) - YPRIME(J)
   10    CONTINUE
         DELF = D6VEC3(YP,NEQ)
         DFDXB = BIG
         IF (DELF .LT. BIG*ABS(DA)) DFDXB = DELF/ABS(DA)
         FBND = D6VEC3(SF,NEQ)
C
C        ...............................................................
C
C             COMPUTE AN ESTIMATE (DFDUB) OF THE LOCAL LIPSCHITZ
C             CONSTANT FOR THE SYSTEM OF DIFFERENTIAL EQUATIONS. THIS
C             ALSO REPRESENTS AN ESTIMATE OF THE NORM OF THE JACOBIAN
C             LOCALLY.  THREE ITERATIONS (TWO WHEN NEQ=1) ARE USED TO
C             ESTIMATE THE LIPSCHITZ CONSTANT BY NUMERICAL DIFFERENCES.
C             THE FIRST PERTURBATION VECTOR IS BASED ON THE INITIAL
C             DERIVATIVES AND DIRECTION OF INTEGRATION. THE SECOND
C             PERTURBATION VECTOR IS FORMED USING ANOTHER EVALUATION OF
C             THE DIFFERENTIAL EQUATION.  THE THIRD PERTURBATION VECTOR
C             IS FORMED USING PERTURBATIONS BASED ONLY ON THE INITIAL
C             VALUES. COMPONENTS THAT ARE ZERO ARE ALWAYS CHANGED TO
C             NON-ZERO VALUES (EXCEPT ON THE FIRST ITERATION). WHEN
C             INFORMATION IS AVAILABLE, CARE IS TAKEN TO ENSURE THAT
C             COMPONENTS OF THE PERTURBATION VECTOR HAVE SIGNS WHICH ARE
C             CONSISTENT WITH THE SLOPES OF LOCAL SOLUTION CURVES.
C             ALSO CHOOSE THE LARGEST BOUND (FBND) FOR THE FIRST
C             DERIVATIVE.
C
C                               PERTURBATION VECTOR SIZE IS HELD
C                               CONSTANT FOR ALL ITERATIONS. COMPUTE
C                               THIS CHANGE FROM THE
C                                       SIZE OF THE VECTOR OF INITIAL
C                                       VALUES.
         DELY = RELPER*D6VEC3(Y,NEQ)
         IF (DELY .EQ. 0.0D0) DELY = RELPER
         DELY = SIGN(DELY,DX)
         DELF = D6VEC3(YPRIME,NEQ)
         FBND = MAX(FBND,DELF)
         IF (DELF .EQ. 0.0D0) GO TO 30
C           USE INITIAL DERIVATIVES FOR FIRST PERTURBATION
            DO 20 J = 1, NEQ
               SPY(J) = YPRIME(J)
               YP(J) = YPRIME(J)
   20       CONTINUE
         GO TO 50
   30    CONTINUE
C           CANNOT HAVE A NULL PERTURBATION VECTOR
            DO 40 J = 1, NEQ
               SPY(J) = 0.0D0
               YP(J) = 1.0D0
   40       CONTINUE
            DELF = D6VEC3(YP,NEQ)
   50    CONTINUE
C
         DFDUB = 0.0D0
         LK = MIN0(NEQ+1,3)
         DO 140 K = 1, LK
C           DEFINE PERTURBED VECTOR OF INITIAL VALUES
            DO 60 J = 1, NEQ
               PV(J) = Y(J) + DELY*(YP(J)/DELF)
   60       CONTINUE
            IF (K .EQ. 2) GO TO 80
C              EVALUATE DERIVATIVES ASSOCIATED WITH PERTURBED
C              VECTOR  AND  COMPUTE CORRESPONDING DIFFERENCES
               CALL BETA(NEQ,A,PV,BVAL,RPAR,IPAR)
               IF (NEQ.LE.0) GO TO 8985
               CALL D6BETA(NEQ,A,DX,BVAL,IOK)
               IF (IOK .NE. 0) THEN
                  INDEX = -1
                  GO TO 8995
               ENDIF
C
C              DO 65 J = 1, NEQ
C                 TVAL = BVAL(J)
C                 CALL YINIT(NEQ,TVAL,YLAG(J),J,RPAR,IPAR)
C                 IF (NEQ.LE.0) GO TO 8985
C                 CALL DYINIT(NEQ,TVAL,DYLAG(J),J,RPAR,IPAR)
C                 IF (NEQ.LE.0) GO TO 8985
C  65          CONTINUE
C
               DO 65 J = 1, NEQ
                  TVAL = BVAL(J)
                  IF ((TVAL.LE.A.AND.B.GT.A)  .OR.
     +                (TVAL.GE.A.AND.B.LT.A)) THEN
                      CALL YINIT(NEQ,TVAL,YLAG(J),J,RPAR,IPAR)
                      IF (NEQ.LE.0) GO TO 8985
                      CALL DYINIT(NEQ,TVAL,DYLAG(J),J,RPAR,IPAR)
                      IF (NEQ.LE.0) GO TO 8985
                  ELSE
                      YLAG(J) = Y(J)
                      DYLAG(J) = YPRIME(J)
                  ENDIF
   65          CONTINUE
C
               INDEX = 0
               CALL DF(NEQ,A,PV,YLAG,DYLAG,YP,RPAR,IPAR)
               IF (NEQ.LE.0) GO TO 8985
               DO 70 J = 1, NEQ
                  PV(J) = YP(J) - YPRIME(J)
   70          CONTINUE
            GO TO 100
   80       CONTINUE
C              USE A SHIFTED VALUE OF THE INDEPENDENT VARIABLE
C                                    IN COMPUTING ONE ESTIMATE
               CALL BETA(NEQ,A+DA,PV,BVAL,RPAR,IPAR)
               IF (NEQ.LE.0) GO TO 8985
               CALL D6BETA(NEQ,A+DA,DX,BVAL,IOK)
               IF (IOK .NE. 0) THEN
                  INDEX = -1
                  GO TO 8995
               ENDIF
C
               TVAL = A + DA
C
               IMETH = 9
               CALL D6HST2(NEQ,Y,DA,INDEX,YLAG,DYLAG,BVAL,YINIT,DYINIT,
     +                     A,A,RPAR,IPAR,YPRIME,YPRIME,YPRIME,YPRIME,
     +                     YPRIME,YPRIME,YPRIME,YPRIME,YPRIME,YPRIME,
     +                     IMETH,IPAIR,IORDER)
               IF (NEQ.LE.0) GO TO 8985
C
C              DO 85 J = 1, NEQ
C                 TVAL = BVAL(J)
C                 IF ((TVAL.LE.A.AND.B.GT.A)  .OR.
C    +                (TVAL.GE.A.AND.B.LT.A)) THEN
C                     CALL YINIT(NEQ,TVAL,YLAG(J),J,RPAR,IPAR)
C                     IF (NEQ.LE.0) GO TO 8985
C                     CALL DYINIT(NEQ,TVAL,DYLAG(J),J,RPAR,IPAR)
C                     IF (NEQ.LE.0) GO TO 8985
C                 ELSE
C                     YLAG(J) = Y(J)
C                     DYLAG(J) = YPRIME(J)
C                 ENDIF
C  85          CONTINUE
C
               INDEX = 0
               CALL DF(NEQ,TVAL,PV,YLAG,DYLAG,YP,RPAR,IPAR)
               IF (NEQ.LE.0) GO TO 8985
               DO 90 J = 1, NEQ
                  PV(J) = YP(J) - SF(J)
   90          CONTINUE
  100       CONTINUE
C           CHOOSE LARGEST BOUNDS ON THE FIRST DERIVATIVE
C                          AND A LOCAL LIPSCHITZ CONSTANT
            FBND = MAX(FBND,D6VEC3(YP,NEQ))
            DELF = D6VEC3(PV,NEQ)
C        ...EXIT
            IF (DELF .GE. BIG*ABS(DELY)) GO TO 150
            DFDUB = MAX(DFDUB,DELF/ABS(DELY))
C     ......EXIT
            IF (K .EQ. LK) GO TO 160
C           CHOOSE NEXT PERTURBATION VECTOR
            IF (DELF .EQ. 0.0D0) DELF = 1.0D0
            DO 130 J = 1, NEQ
               IF (K .EQ. 2) GO TO 110
                  DY = ABS(PV(J))
                  IF (DY .EQ. 0.0D0) DY = DELF
               GO TO 120
  110          CONTINUE
                  DY = Y(J)
                  IF (DY .EQ. 0.0D0) DY = DELY/RELPER
  120          CONTINUE
               IF (SPY(J) .EQ. 0.0D0) SPY(J) = YP(J)
               IF (SPY(J) .NE. 0.0D0) DY = SIGN(DY,SPY(J))
               YP(J) = DY
  130       CONTINUE
            DELF = D6VEC3(YP,NEQ)
  140    CONTINUE
  150    CONTINUE
C
C        PROTECT AGAINST AN OVERFLOW
         DFDUB = BIG
  160 CONTINUE
C
C     ..................................................................
C
C          COMPUTE A BOUND (YDPB) ON THE NORM OF THE SECOND DERIVATIVE
C
      YDPB = DFDXB + DFDUB*FBND
C
C     ..................................................................
C
C          DEFINE THE TOLERANCE PARAMETER UPON WHICH THE STARTING STEP
C          SIZE IS TO BE BASED.  A VALUE IN THE MIDDLE OF THE ERROR
C          TOLERANCE RANGE IS SELECTED.
C
      TOLMIN = BIG
      TOLSUM = 0.0D0
      DO 170 K = 1, NEQ
         TOLEXP = LOG10(ETOL(K))
         TOLMIN = MIN(TOLMIN,TOLEXP)
         TOLSUM = TOLSUM + TOLEXP
  170 CONTINUE
      TOLP = 10.0D0
     +       **(0.5D0*(TOLSUM/DBLE(FLOAT(NEQ)) + TOLMIN)
     +          /DBLE(FLOAT(MORDER+1)))
C
C     ..................................................................
C
C          COMPUTE A STARTING STEPSIZE BASED ON THE ABOVE FIRST AND
C          SECOND DERIVATIVE INFORMATION
C
C                            RESTRICT THE STEP LENGTH TO BE NOT BIGGER
C                            THAN ABS(B-A).   (UNLESS  B  IS TOO CLOSE
C                            TO  A)
      H = ABSDX
C
      IF (YDPB .NE. 0.0D0 .OR. FBND .NE. 0.0D0) GO TO 180
C
C        BOTH FIRST DERIVATIVE TERM (FBND) AND SECOND
C                     DERIVATIVE TERM (YDPB) ARE ZERO
         IF (TOLP .LT. 1.0D0) H = ABSDX*TOLP
      GO TO 200
  180 CONTINUE
C
      IF (YDPB .NE. 0.0D0) GO TO 190
C
C        ONLY SECOND DERIVATIVE TERM (YDPB) IS ZERO
         IF (TOLP .LT. FBND*ABSDX) H = TOLP/FBND
      GO TO 200
  190 CONTINUE
C
C        SECOND DERIVATIVE TERM (YDPB) IS NON-ZERO
         SRYDPB = SQRT(0.5D0*YDPB)
         IF (TOLP .LT. SRYDPB*ABSDX) H = TOLP/SRYDPB
  200 CONTINUE
C
C     FURTHER RESTRICT THE STEP LENGTH TO BE NOT
C                               BIGGER THAN  1/DFDUB
      IF (H*DFDUB .GT. 1.0D0) H = 1.0D0/DFDUB
C
C     FINALLY, RESTRICT THE STEP LENGTH TO BE NOT
C     SMALLER THAN  100*SMALL*ABS(A).  HOWEVER, IF
C     A=0. AND THE COMPUTED H UNDERFLOWED TO ZERO,
C     THE ALGORITHM RETURNS  SMALL*ABS(B)  FOR THE
C                                     STEP LENGTH.
      H = MAX(H,100.0D0*SMALL*ABS(A))
      IF (H .EQ. 0.0D0) H = SMALL*ABS(B)
C
C     NOW SET DIRECTION OF INTEGRATION
      H = SIGN(H,DX)
C
C     NORMAL RETURNS ...
C
      RETURN
C
C 300 CONTINUE
C
C     IF AN ERROR OCCURRED , D6HST1 WILL SIMPLY SET THE
C     INITIAL STEPSIZE EQUAL TO B-A.
C
C     INDEX = 0
C     H = B - A
C     RETURN
C
C     ERROR RETURNS ...
C
 8985 CONTINUE
C
      INDEX = -14
      CALL D6MESG(25,'D6HST1')
      RETURN
C
 8990 CONTINUE
C
      CALL D6MESG(33,'D6HST1')
      RETURN
C
 8995 CONTINUE
C
      CALL D6MESG(34,'D6HST1')
C
      RETURN
      END
C*DECK D6HST2
      SUBROUTINE D6HST2(N,YOLD,H,INDEX,YLAG,DYLAG,BVAL,YINIT,
     +                  DYINIT,TINIT,TOLD,RPAR,IPAR,K0,K1,K2,
     +                  K3,K4,K5,K6,K7,K8,K9,IMETH,IPAIR,
     +                  IORDER)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6HST2
C***SUBSIDIARY
C***PURPOSE  The purpose of D6HST2 is to interpolate the delayed
C            solution and derivative during the first step of
C            the integration for the D6HST1 initial stepsize
C            selection subroutine for the DKLAG6 differential
C            equation solver. It is called by D6HST1 if a delay
C            time falls beyond the initial time queue on the
C            first integration step.
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
C***ROUTINES CALLED  D1MACH, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6HST2
C
      IMPLICIT NONE
C
      INTEGER I,INDEX,N,IPAR,IMETH,IPAIR,IORDER
      DOUBLE PRECISION BVAL,H,TINIT,YLAG,YOLD,DYLAG,RPAR,
     +       K0,K1,K2,K3,K4,K5,K6,K7,K8,K9,TOLD
C
C     DOUBLE PRECISION D1MACH
C
      DIMENSION YOLD(*),YLAG(*),BVAL(*),DYLAG(*),RPAR(*),
     +          IPAR(*),K0(*),K1(*),K2(*),K3(*),K4(*),
     +          K5(*),K6(*),K7(*),K8(*),K9(*)
C
      EXTERNAL YINIT,DYINIT
C
C***FIRST EXECUTABLE STATEMENT  D6HST2
C
      INDEX = 2
C
      DO 100 I = 1 , N
C
         IF (BVAL(I) .LE. TINIT) THEN
C
            CALL YINIT(N,BVAL(I),YLAG(I),I,RPAR,IPAR)
            IF (N.LE.0) GO TO 200
            CALL DYINIT(N,BVAL(I),DYLAG(I),I,RPAR,IPAR)
            IF (N.LE.0) GO TO 200
C
         ELSE
C
            CALL D6INT9(N,TOLD,YOLD,H,K0,K1,K2,K3,K4,K5,K6,
     +                  K7,K8,K9,BVAL,I,YLAG,DYLAG,IMETH,
     +                  IPAIR,IORDER)
C
         ENDIF
C
  100 CONTINUE
C
  200 CONTINUE
C
C     CHECK IF THE USER TERMINATED THE INTEGRATION.
C
      IF (N.LE.0) THEN
         CALL D6MESG(25,'D6HST2')
      ENDIF
C
      RETURN
      END
C*DECK D6HST3
      SUBROUTINE D6HST3(IERTST,RATIO,H,HNEXT,IERRUS,IORDER,ICVAL)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6HST3
C***SUBSIDIARY
C***PURPOSE  The purpose of D6HST3 is to calculate the stepsize
C            for the DKLAG6 differential equation solver.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6HST3
C
      IMPLICIT NONE
C
C     ESTIMATE THE STEPSIZE TO BE ATTEMPTED ON THE NEXT
C     INTEGRATION STEP.
C
      INTEGER IERTST,IERRUS,IORDER,ICVAL,JORDER
      DOUBLE PRECISION H,HEXP,DORDER,HNEXT,RATIO,RATMAX,
     +                 RATMIN,EFAC,FACUP,FACDN
C
C***FIRST EXECUTABLE STATEMENT  D6HST3
C
      HNEXT = H
      IERTST = 0
C
C     REDUCE THE ORDER BY ONE IF DERIVATIVE ERROR CONTROL
C     IS BEING USED.
C
      JORDER = IORDER
      IF (ICVAL.GE.7) JORDER = IORDER - 1
C
C     NOTE:
C     IF A RETURN IS MADE AT THIS POINT, THE STEPSIZE
C     WILL NOT BE CHANGED FROM IT'S INITIAL VALUE (IF
C     NO ROOTFINDING IS BEING PERFORMED AND NO ILLEGAL
C     DELAYS ARE ENCOUNTERED).
C
C     USE ERROR PER UNIT STEP IF IERRUS = 0. USE ERROR
C     PER STEP OTHERWISE.
C     NOTE: THE K'S ARE NOT MULTIPLIED BY H.
C
      IF (IERRUS.NE.0.AND.ICVAL.LE.6) RATIO = RATIO*ABS(H)
C
C     FACDN IS THE MAXIMUM FACTOR BY WHICH THE STEPSIZE
C     WILL BE DECREASED.
C
C     FACDN = 10.0D0
      FACDN = 2.0D0
C
C     FACUP IS THE MAXIMUM FACTOR BY WHICH THE STEPSIZE
C     WILL BE INCREASED.
C
C     FACUP = 5.0D0
      FACUP = 2.0D0
C
C     ATTEMPT TO OBTAIN A LOCAL ERROR OF EFAC TIMES
C     THE ERROR TOLERANCE VECTOR.
C
      EFAC = 0.5D0
C
C     IF RATIO IS GREATER THAN 1.0D0, THE ERROR IS TOO LARGE.
C
      IF (RATIO.GT.1.0D0) GO TO 10
C
C     THE ESTIMATED ERROR FOR THIS STEP IS SMALL ENOUGH.
C     ESTIMATE THE STEPSIZE FOR THE NEXT STEP. LIMIT THE
C     INCREASE TO A FACTOR OF AT MOST FACUP AND A DECREASE
C     TO A FACTOR OF AT MOST FACDN.
C
      IERTST = 0
      DORDER = DBLE(FLOAT(JORDER))
      RATMAX = EFAC*(FACDN**DORDER)
      RATMIN = EFAC/(FACUP**DORDER)
      RATIO = MAX(RATIO,RATMIN)
      RATIO = MIN(RATIO,RATMAX)
      HEXP = 1.0D0/DORDER
      HNEXT = ((EFAC/RATIO)**HEXP)*ABS(H)
      HNEXT = SIGN(HNEXT,H)
      GO TO 20
C
   10 CONTINUE
C
C     THE ESTIMATED ERROR FOR THIS STEP IS TOO LARGE.
C     ESTIMATE A SMALLER STEPSIZE WITH WHICH TO REDO
C     THE STEP. LIMIT THE DECREASE TO A FACTOR OF AT
C     MOST FACDN.
C
      IERTST = 1
      DORDER = DBLE(FLOAT(JORDER))
      RATMAX = EFAC*(FACDN**DORDER)
      RATIO = MIN(RATIO,RATMAX)
      HEXP = 1.0D0/DORDER
      HNEXT = ((EFAC/RATIO)**HEXP)*ABS(H)
      HNEXT = SIGN(HNEXT,H)
C
   20 CONTINUE
C
      RETURN
      END
C*DECK D6HST4
      SUBROUTINE D6HST4(HNEXT,HMAX)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6HST4
C***SUBSIDIARY
C***PURPOSE  The purpose of D6HST4 is to limit the stepsize to
C            be no larger than the maximum stepsize for the
C            DKLAG6 differential equation solver.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6HST4
C
      IMPLICIT NONE
C
      DOUBLE PRECISION HMAX,HMAXN,HNEXT
C
C     LIMIT THE ESTIMATED STEPSIZE TO BE NO LARGER IN
C     MAGNITUDE THAN HMAX.
C
C***FIRST EXECUTABLE STATEMENT  D6HST4
C
      HMAXN = ABS(HMAX)
      HNEXT = SIGN(MIN(ABS(HNEXT),HMAXN),HNEXT)
C
      RETURN
      END
C*DECK D6QUE1
      SUBROUTINE D6QUE1(N,QUEUE,LQUEUE,IPOINT,TOLD,YOLD,TNEW,K0,
     +                  K2,K3,K4,K5,K6,K7,K8,K9)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6QUE1
C***SUBSIDIARY
C***PURPOSE  The purpose of D6QUE1 is to update the past solution
C            history queue for the DKLAG6 differential equation
C            solver. It is called by D6DRV3 following each
C            successful integration step.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6QUE1
C
C     UPDATE THE CIRCULAR QUEUE HISTORY ARRAY.
C
      IMPLICIT NONE
C
      INTEGER I,IBEGIN,IPOINT,IQUEUE,JQUEUE,KQUEUE,LQUEUE,N,NUMPT
      DOUBLE PRECISION K0,K2,K3,K4,K5,K6,K7,K8,K9,QUEUE,YOLD,
     +                 TNEW,TOLD
C
      DIMENSION QUEUE(N+1,*),IPOINT(*),YOLD(*),K0(*),K2(*),K3(*),
     +          K4(*),K5(*),K6(*),K7(*),K8(*),K9(*)
C
C     IPOINT POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; JQUEUE TO THE OLDEST. KQUEUE IS USED TO
C     INCREMENT JQUEUE.
C
C                         QUEUE PLAY
C
C     n      = N is the number of equations in the system
C     QUEUE  = solution history queue. size = QUEUE(N+1,*)
C     LQUEUE = length of QUEUE
C     m = NUMPT = LQUEUE/(12*(N+1)) is the maximum number
C                                  of points for which the
C                                  solution can be saved
C
C  ------> more recent              -------->
C
C    row/column         |     stored in slugs of 11 vectors
C                       |     (yold for second told begins
C                       |     in column m+12)
C                       |
C       1      ...>   m | m+1   m+2   m+3 m+4 m+5 m+6 m+7 m+8 9,10,11
C-----------------------|--------------------------------------------
C 1    told's  ...>     | yold dyold  k0  k3  k4  k5  k6  k2 k7 k8 k9
C 2    tnew's  ...>     |      (scratch)
C-----------------------| .    .      .   .   .   .   .   .  .  .  .
C .                     | .    .      .   .   .   .   .   .  .  .  .
C .                     | .    .      .   .   .   .   .   .  .  .  .
C .                     |
C    not used           |
C    if n > 2           | (n)
C                       |--------------------------------------------
C n+1                   |              not used
C-----------------------|--------------------------------------------
C
C     When the queue is full, the information for the oldest
C     value of told is over-written (circular queue).
C
C***FIRST EXECUTABLE STATEMENT  D6QUE1
C
      IQUEUE = IPOINT(21)
      JQUEUE = IPOINT(22)
      KQUEUE = IPOINT(23)
C
C     THE QUEUE CAN SAVE INFORMATION FOR AS MANY
C     AS NUMPT POINTS ....
C
      NUMPT = LQUEUE/ (12*(N+1))
C
C     UPDATE THE POINTER TO THE MOST RECENT INFORMATION
C     IN THE CIRCULAR QUEUE.
C
      IQUEUE = IQUEUE + 1
C
      IF (IQUEUE.GT.NUMPT) THEN
C
C         CIRCLE BACK TO THE BEGINNING OF THE QUEUE AND
C         OVERWRITE THE OLDEST INFORMATION STORED THERE.
C
          IQUEUE = 1
C
C         THE QUEUE IS NOW FILLED. (UNTIL WE REACH HERE
C         THE FIRST TIME, KQUEUE HAS A VALUE OF 0.)
C
          KQUEUE = 1
C
      ENDIF
C
C     UPDATE THE INFORMATION TO THE OLDEST HISTORY INFORMATION.
C
      JQUEUE = JQUEUE + KQUEUE
C
      IF (JQUEUE.GT.NUMPT) JQUEUE = 1
C
C     UPDATE THE POINTER INFORMATION WITH THE NEW QUEUE
C     POINTER INFORMATION.
C
      IPOINT(21) = IQUEUE
      IPOINT(22) = JQUEUE
      IPOINT(23) = KQUEUE
C
C     UPDATE THE INDEPENDENT VARIABLE INFORMATION. (SAVING
C     BOTH AVOIDS PROBLEM OF RETRIEVING BOTH WHEN THERE IS
C     ONLY ONE POINT IN THE QUEUE.)
C
      QUEUE(1,IQUEUE) = TOLD
      QUEUE(2,IQUEUE) = TNEW
C
C     STORE THE INTEGRATION INFORMATION FOR (TOLD,TNEW)
C     IN THE QUEUE.
C
C     NOTE:
C     THERE IS NO NEED TO SAVE DYOLD SINCE K0 = DYOLD. THIS
C     SPACE WILL BE USED FOR SCRATCH STORAGE IN SUBROUTINE
C     D6INT6. K1 IS ALSO NOT SAVED SINCE IT IS NOT USED IN
C     THE INTERPOLATION SUBROUTINE.
C
      IBEGIN = NUMPT + (IQUEUE-1)*11
C
      DO 20 I = 1,N
C
          QUEUE(I,IBEGIN+1)  = YOLD(I)
C         QUEUE(I,IBEGIN+2)  = DYOLD(I)
          QUEUE(I,IBEGIN+3)  = K0(I)
          QUEUE(I,IBEGIN+4)  = K3(I)
          QUEUE(I,IBEGIN+5)  = K4(I)
          QUEUE(I,IBEGIN+6)  = K5(I)
          QUEUE(I,IBEGIN+7)  = K6(I)
          QUEUE(I,IBEGIN+8)  = K2(I)
          QUEUE(I,IBEGIN+9)  = K7(I)
          QUEUE(I,IBEGIN+10) = K8(I)
          QUEUE(I,IBEGIN+11) = K9(I)
C
   20 CONTINUE
C
      RETURN
      END
C*DECK D6QUE2
      SUBROUTINE D6QUE2(N,TFINAL,IPOINT,QUEUE)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6QUE2
C***SUBSIDIARY
C***PURPOSE  The purpose of D6QUE2 is to adjust the last mesh
C            point entry in the solution history queue for
C            the DKLAG6 ordinary differential equation solver,
C            in the event D6DRV2 performs an extrapolation near
C            the final integration time.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6QUE2
C
      IMPLICIT NONE
C
      INTEGER N, IPOINT(*), IQUEUE
      DOUBLE PRECISION TFINAL,QUEUE(N+1,*)
C
C***FIRST EXECUTABLE STATEMENT  D6QUE2
C
      IQUEUE = IPOINT(21)
      QUEUE(2,IQUEUE) = TFINAL
C
      RETURN
      END
C*DECK D6VEC1
      SUBROUTINE D6VEC1(N,VEC,CON)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6VEC1
C***SUBSIDIARY
C***PURPOSE  The purpose of D6VEC1 is to load a constant
C            into a vector for the DKLAG6 differential
C            equation solver.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6VEC1
C
      IMPLICIT NONE
C
      INTEGER I,N
      DOUBLE PRECISION CON, VEC
C
      DIMENSION VEC(*)
C
C***FIRST EXECUTABLE STATEMENT  D6VEC1
C
      DO 10 I = 1,N
         VEC(I) = CON
   10 CONTINUE
C
      RETURN
      END
C*DECK D6VEC2
      SUBROUTINE D6VEC2(N,X,ALPHA)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6VEC2
C***SUBSIDIARY
C***PURPOSE  The purpose of D6VEC2 is to multiply a constant
C            times a vector for the DKLAG6 differential
C            equation solver.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6VEC2
C
C     MULTIPLY A VECTOR BY A CONSTANT.
C
      IMPLICIT NONE
C
      INTEGER I,N
      DOUBLE PRECISION ALPHA,X
C
      DIMENSION X(*)
C
C***FIRST EXECUTABLE STATEMENT  D6VEC2
C
      DO 10 I = 1,N
          X(I) = ALPHA*X(I)
   10 CONTINUE
C
      RETURN
      END
C*DECK D6VEC3
      DOUBLE PRECISION FUNCTION D6VEC3(V,NCOMP)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE D6VEC3 
C***SUBSIDIARY
C***PURPOSE  The function of D6VEC3 is to compute the maximum
C            norm for the DKLAG6 differential equation solver.
C            D6VEC3 is the same as DVNORM from the SLATEC
C            library. It is called by the D6HST1 initial
C            stepsize selection subroutine.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6VEC3
C
      IMPLICIT NONE
C
      INTEGER K, NCOMP
      DOUBLE PRECISION V
      DIMENSION V(NCOMP)
C
C***FIRST EXECUTABLE STATEMENT  D6VEC3
C
      D6VEC3 = 0.0D0
      DO 10 K = 1, NCOMP
         D6VEC3 = MAX(D6VEC3,ABS(V(K)))
   10 CONTINUE
C
      RETURN
      END
C*DECK D6SRC1
      SUBROUTINE D6SRC1(N,QUEUE,HNEXT,IORDER,IOLD,INEW,NUMPT,INDEXO,
     +                  TVAL)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6SRC1
C***SUBSIDIARY
C***PURPOSE  The purpose of D6SRC1 is to perform binary searches
C            of the solution history queue for the DKLAG6
C            differential equation solver.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6SRC1
C
      IMPLICIT NONE
C
C     BINARY SEARCH ROUTINE CALLED BY D6SRC2 AND D6SRC3 TO
C     LOCATE THE QUEUE INTEGRATION DATA NEEDED TO INTERPOLATE
C     YLAG FOR THIS VALUE OF TVAL.
C
      INTEGER I,INDEXO,INEW,IOLD,IORDER,IQ,J,JP,JPQ,JQ,N,NUMPT
      DOUBLE PRECISION HNEXT,QUEUE,TVAL
C
      DIMENSION QUEUE(N+1,*)
C
C***FIRST EXECUTABLE STATEMENT  D6SRC1
C
C     CASE 1 -- THE QUEUE IS ORDERED LINEARLY.
C
      IF (IORDER.EQ.0) THEN
C
C         IF HNEXT > 0.0D0, THE T-DATA IS INCREASING ....
C
          IF (HNEXT.GT.0.0D0) THEN
C
C             IQ,JQ ARE BOTH T-INDEXES AND Q-INDEXES.
C
              IF (TVAL .GE. QUEUE(1,INEW)) THEN
                 INDEXO = INEW
                 GO TO 9005
              ENDIF
              IQ = IOLD
              JQ = INEW
   10         CONTINUE
              IF (JQ-IQ.EQ.1) GO TO 30
              JPQ = (JQ+IQ)/2
              IF (TVAL.LT.QUEUE(1,JPQ)) GO TO 20
              IQ = JPQ
              GO TO 10
   20         CONTINUE
              JQ = JPQ
              GO TO 10
   30         CONTINUE
              INDEXO = IQ
              GO TO 9005
          ENDIF
C
C         IF HNEXT < 0.0D0, THE T-DATA IS DECREASING ....
C
          IF (HNEXT.LT.0.0D0) THEN
C
C             IQ,JQ ARE BOTH T-INDEXES AND Q-INDEXES.
C
              IF (TVAL .LE. QUEUE(1,INEW)) THEN
                 INDEXO = INEW
                 GO TO 9005
              ENDIF
              IQ = IOLD
              JQ = INEW
   40         CONTINUE
              IF (JQ-IQ.EQ.1) GO TO 60
              JPQ = (JQ+IQ)/2
              IF (TVAL.GT.QUEUE(1,JPQ)) GO TO 50
              IQ = JPQ
              GO TO 40
   50         CONTINUE
              JQ = JPQ
              GO TO 40
   60         CONTINUE
              INDEXO = IQ
              GO TO 9005
          ENDIF
C
      ENDIF
C
C     CASE 2 -- THE QUEUE IS ORDERED CIRCULARLY.
C
      IF (IORDER.EQ.1) THEN
C
C         IF HNEXT > 0.0D0, THE T-DATA IS INCREASING ....
C
          IF (HNEXT.GT.0.0D0) THEN
C
C             I,J ARE T-INDEXES AND IQ,JQ ARE Q-INDEXES.
C
              IF (TVAL .GE. QUEUE(1,INEW)) THEN
                 INDEXO = INEW
                 GO TO 9005
              ENDIF
              I = 1
              J = NUMPT
              IQ = IOLD + I - 1
              JQ = IOLD + J - 1
              IF (IQ.GT.NUMPT) IQ = IQ - NUMPT
              IF (JQ.GT.NUMPT) JQ = JQ - NUMPT
   70         CONTINUE
              IF (J-I.EQ.1) GO TO 90
              JP = (J+I)/2
              JPQ = IOLD + JP - 1
              IF (JPQ.GT.NUMPT) JPQ = JPQ - NUMPT
              IF (TVAL.LT.QUEUE(1,JPQ)) GO TO 80
              I = JP
              IQ = JPQ
              GO TO 70
   80         CONTINUE
              J = JP
              JQ = JPQ
              GO TO 70
   90         CONTINUE
              INDEXO = IQ
              GO TO 9005
          ENDIF
C
C         IF HNEXT < 0.0D0, THE T-DATA IS DECREASING ....
C
          IF (HNEXT.LT.0.0D0) THEN
C
C             I,J ARE T-INDEXES AND IQ,JQ ARE Q-INDEXES.
C
              IF (TVAL .LE. QUEUE(1,INEW)) THEN
                 INDEXO = INEW
                 GO TO 9005
              ENDIF
              I = 1
              J = NUMPT
              IQ = IOLD + I - 1
              JQ = IOLD + J - 1
              IF (IQ.GT.NUMPT) IQ = IQ - NUMPT
              IF (JQ.GT.NUMPT) JQ = JQ - NUMPT
  100         CONTINUE
              IF (J-I.EQ.1) GO TO 120
              JP = (J+I)/2
              JPQ = IOLD + JP - 1
              IF (JPQ.GT.NUMPT) JPQ = JPQ - NUMPT
              IF (TVAL.GT.QUEUE(1,JPQ)) GO TO 110
              I = JP
              IQ = JPQ
              GO TO 100
  110         CONTINUE
              J = JP
              JQ = JPQ
              GO TO 100
  120         CONTINUE
              INDEXO = IQ
          ENDIF
C
      ENDIF
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6SRC2
      SUBROUTINE D6SRC2(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TOLD,
     +                  TNEW,INDEX,ICHECK)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6SRC2
C***SUBSIDIARY
C***PURPOSE  The purpose of D6SRC2 is to assist in the queue
C            search and interpolation for the DKLAG6 differential
C            equation solver (for non-vanishing delay problems).
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
C***ROUTINES CALLED  D1MACH, D6SRC1, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6SRC2
C
C     SEARCH THE HISTORY QUEUE FOR D6DRV3 TO LOCATE THE
C     SLUG OF DATA TO BE INTERPOLATED FOR THIS VALUE
C     OF TVAL.
C
C     CAUTION:
C     THIS ROUTINE IS NOT INTENDED TO BE CALLED WITH
C     HNEXT=0.0D0 OR INEW=0. (THE LATTER CASE WOULD
C     CORRESPOND TO THE FIRST STEP OF THE INTEGRATION
C     AND IS HANDLED SEPARATELY IN SUBROUTINE D6INT1.
C     THE FORMER SHOULD NEVER OCCUR AND SHOULD BE
C     TREATED AS AN ERROR.)
C
      IMPLICIT NONE
C
      INTEGER ICHECK,INDEX,INDEXO,INEW,IOLD,JORDER,N,NUMPT
      DOUBLE PRECISION D1MACH,HNEXT,QUEUE,TNEW,TOLD,TVAL
C
      DIMENSION QUEUE(N+1,*)
C
C***FIRST EXECUTABLE STATEMENT  D6SRC2
C
      INDEX = 2
C
C     INEW POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; IOLD TO THE OLDEST.
C
C     CHECK FOR EXTRAPOLATION OUTSIDE THE QUEUE. ALSO
C     HANDLE THE CASE IN WHICH TVAL = TNEW.
C
      IF (HNEXT.GT.0.0D0) THEN
C
C         IF (TVAL.LT.QUEUE(1,IOLD)) THEN
C
          IF ((TVAL-QUEUE(1,IOLD)) .LT.
     +    (-1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(1,IOLD)))))
     +    THEN
C
C             **********
C             ** NOTE **
C             **********
C
C             THE FOLLOWING STATEMENTS WOULD ALLOW
C             EXTRAPOLATION OF THE OLDEST POLYNOMIAL.
C
C             CALL D6MESG(17,'D6SRC2')
C             INDEXO = IOLD
C             TOLD = QUEUE(1,IOLD)
C             TNEW = QUEUE(2,IOLD)
C             GO TO 9005
C
              CALL D6MESG(26,'D6SRC2')
              INDEX = -12
              GO TO 9005
C
          END IF
C
          IF (TVAL.GE.QUEUE(2,INEW)) THEN
C
C            IF (TVAL.GT.QUEUE(2,INEW)) THEN
C
             IF ((TVAL-QUEUE(2,INEW)) .GT.
     +       (1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(2,INEW)))))
     +       THEN
C
C               **********
C               ** NOTE **
C               **********
C               IF THE FOLLOWING THREE STATEMENTS ARE REMOVED, THE
C               MOST RECENT POLYNOMIAL WILL BE EXTRAPOLATED.
C
                IF (ICHECK .NE.2)
     +          CALL D6MESG(27,'D6SRC2')
                INDEX = -11
                GO TO 9005
C
             ENDIF
C
              INDEXO = INEW
              TOLD = QUEUE(1,INEW)
              TNEW = QUEUE(2,INEW)
              GO TO 9005
C
          END IF
C
      END IF
C
      IF (HNEXT.LT.0.0D0) THEN
C
C         IF (TVAL.GT.QUEUE(1,IOLD)) THEN
C
          IF ((TVAL-QUEUE(1,IOLD)) .GT.
     +    (1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(1,IOLD)))))
     +    THEN
C
C             **********
C             ** NOTE **
C             **********
C
C             THE FOLLOWING STATEMENTS WOULD ALLOW
C             EXTRAPOLATION OF THE OLDEST POLYNOMIAL.
C
C             CALL D6MESG(17,'D6SRC2')
C             INDEXO = IOLD
C             TOLD = QUEUE(1,IOLD)
C             TNEW = QUEUE(2,IOLD)
C             GO TO 9005
C
              CALL D6MESG(26,'D6SRC2')
              INDEX = -12
              GO TO 9005
C
          END IF
C
          IF (TVAL.LE.QUEUE(2,INEW)) THEN
C
C             IF (TVAL.LT.QUEUE(2,INEW)) THEN
C
              IF ((TVAL-QUEUE(2,INEW)) .LT.
     +        (-1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(2,INEW)))))
     +        THEN
C
C                 **********
C                 ** NOTE **
C                 **********
C
C                 IF THE FOLLOWING THREE STATEMENTS ARE REMOVED, THE
C                 MOST RECENT POLYNOMIAL WILL BE EXTRAPOLATED.
C
                  IF (ICHECK .NE.2)
     +            CALL D6MESG(27,'D6SRC2')
                  INDEX = -11
                  GO TO 9005
C
              END IF
C
              INDEXO = INEW
              TOLD = QUEUE(1,INEW)
              TNEW = QUEUE(2,INEW)
              GO TO 9005
C
          END IF
C
      END IF
C
C     HANDLE THE CASES IN WHICH THERE IS ONLY ONE POINT
C     IN THE QUEUE.
C
      IF ((NUMPT.EQ.1) .OR. ((IOLD.EQ.1).AND. (INEW.EQ.1))) THEN
C
          INDEXO = 1
          TOLD = QUEUE(1,1)
          TNEW = QUEUE(2,1)
          GO TO 9005
C
      END IF
C
C     WE GET HERE ONLY IF THERE IS MORE THAN ONE POINT IN THE
C     QUEUE AND QUEUE(1,IOLD) .LE. TVAL .LT. QUEUE(1,INEW) .
C
C     HANDLE THE CASE IN WHICH (1=) IOLD < INEW (SO THAT THE
C     DATA IS ALREADY ORDERED). NOTE THAT THE QUEUE MAY NOT
C     YET BE FULL.
C
      IF (IOLD.LT.INEW) THEN
C
          JORDER = 0
          CALL D6SRC1(N,QUEUE,HNEXT,JORDER,IOLD,INEW,NUMPT,INDEXO,TVAL)
          TOLD = QUEUE(1,INDEXO)
          TNEW = QUEUE(2,INDEXO)
          GO TO 9005
C
      END IF
C
C     HANDLE THE CASE IN WHICH INEW < IOLD (=INEW+1) SO THAT
C     THE DATA IS CIRCULARLY ORDERED. NOTE THAT THE QUEUE
C     MUST BE FULL IN THIS CASE.
C
      IF (IOLD.GT.INEW) THEN
C
          JORDER = 1
          CALL D6SRC1(N,QUEUE,HNEXT,JORDER,IOLD,INEW,NUMPT,INDEXO,TVAL)
          TOLD = QUEUE(1,INDEXO)
          TNEW = QUEUE(2,INDEXO)
C
      END IF
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6SRC3
      SUBROUTINE D6SRC3(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TOLD,
     +                  TNEW,INDEX,ICHECK,IVAN)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6SRC3
C***SUBSIDIARY
C***PURPOSE  The purpose of D6SRC3 is to assist in the queue search
C            and interpolation for the DKLAG6 differential equation
C            solver (for vanishing delay problems).
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
C***ROUTINES CALLED  D1MACH, D6SRC1, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6SRC3
C
C     SEARCH THE HISTORY QUEUE FOR D6DRV3 TO LOCATE THE
C     SLUG OF DATA TO BE INTERPOLATED FOR THIS VALUE
C     OF TVAL.
C
C     CAUTION:
C     THIS ROUTINE IS NOT INTENDED TO BE CALLED WITH
C     HNEXT=0.0D0 OR INEW=0. (THE LATTER CASE WOULD
C     CORRESPOND TO THE FIRST STEP OF THE INTEGRATION
C     AND IS HANDLED SEPARATELY IN SUBROUTINE D6INT1.
C     THE FORMER SHOULD NEVER OCCUR AND SHOULD BE
C     TREATED AS AN ERROR.)
C
      IMPLICIT NONE
C
      INTEGER ICHECK,INDEX,INDEXO,INEW,IOLD,JORDER,N,
     +        NUMPT,IVAN
      DOUBLE PRECISION D1MACH,HNEXT,QUEUE,TNEW,TOLD,TVAL
C
      DIMENSION QUEUE(N+1,*)
C
C***FIRST EXECUTABLE STATEMENT  D6SRC3
C
      INDEX = 2
C
C     INEW POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; IOLD TO THE OLDEST.
C
C     CHECK FOR EXTRAPOLATION OUTSIDE THE QUEUE. ALSO
C     HANDLE THE CASE IN WHICH TVAL = TNEW.
C
      IF (HNEXT.GT.0.0D0) THEN
C
C         IF (TVAL.LT.QUEUE(1,IOLD)) THEN
C
          IF ((TVAL-QUEUE(1,IOLD)) .LT.
     +    (-1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(1,IOLD)))))
     +    THEN
C
C             **********
C             ** NOTE **
C             **********
C
C             THE FOLLOWING STATEMENTS WOULD ALLOW
C             EXTRAPOLATION OF THE OLDEST POLYNOMIAL.
C
C             CALL D6MESG(17,'D6SRC3')
C             INDEXO = IOLD
C             TOLD = QUEUE(1,IOLD)
C             TNEW = QUEUE(2,IOLD)
C             GO TO 9005
C
              CALL D6MESG(26,'D6SRC3')
              INDEX = -12
              GO TO 9005
C
          ENDIF
C
          IF (TVAL.GE.QUEUE(2,INEW)) THEN
C
C            IF (TVAL.GT.QUEUE(2,INEW)) THEN
C
             IF ((TVAL-QUEUE(2,INEW)) .GT.
     +       (1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(2,INEW)))))
     +       THEN
C
C               **********
C               ** NOTE **
C               **********
C
C               EXTRAPOLATION OF THE MOST RECENT POLYNOMIAL IS NOT
C               ALLOWED IF WE REACHED HERE AS A RESULT OF A CALL
C               TO D6INT1 BY THE USER OR BY D6DRV2, OR THE SECOND
C               CALL TO D6STP2 FROM D6DRV3. IF WE REACHED HERE AS
C               A RESULT OF THE FIRST CALL TO D6STP2 IN D6DRV3,
C               EXTRAPOLATION OF THE MOST RECENT POLYNOMIAL IS
C               ALLOWED FOLLOWING A SUCCESSFUL STEP FOR WHICH A
C               DELAY FALLS BEYOND THE HISTORY QUEUE ON THE NEXT
C               STEP.
C
                IF (ICHECK.NE.2) THEN
                   CALL D6MESG(27,'D6SRC3')
                   INDEX = -11
                   GO TO 9005
                ENDIF
C               ICHECK = 2 ...
                INDEXO = INEW
                TOLD = QUEUE(1,INEW)
                TNEW = QUEUE(2,INEW)
                IF (IVAN.EQ.0) INDEX = -11
                GO TO 9005
C
             ENDIF
C
          ENDIF
C
      ENDIF
C
      IF (HNEXT.LT.0.0D0) THEN
C
C         IF (TVAL.GT.QUEUE(1,IOLD)) THEN
C
          IF ((TVAL-QUEUE(1,IOLD)) .GT.
     +    (1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(1,IOLD)))))
     +    THEN
C
C             **********
C             ** NOTE **
C             **********
C
C             THE FOLLOWING STATEMENTS WOULD ALLOW
C             EXTRAPOLATION OF THE OLDEST POLYNOMIAL.
C
C             CALL D6MESG(17,'D6SRC3')
C             INDEXO = IOLD
C             TOLD = QUEUE(1,IOLD)
C             TNEW = QUEUE(2,IOLD)
C             GO TO 9005
C
              CALL D6MESG(26,'D6SRC3')
              INDEX = -12
              GO TO 9005
C
          ENDIF
C
          IF (TVAL.LE.QUEUE(2,INEW)) THEN
C
C             IF (TVAL.LT.QUEUE(2,INEW)) THEN
C
              IF ((TVAL-QUEUE(2,INEW)) .LT.
     +        (-1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(2,INEW)))))
     +        THEN
C
C               **********
C               ** NOTE **
C               **********
C
C               EXTRAPOLATION OF THE MOST RECENT POLYNOMIAL IS NOT
C               ALLOWED IF WE REACHED HERE AS A RESULT OF A CALL
C               TO D6INT1 BY THE USER OR BY D6DRV2, OR THE SECOND
C               CALL TO D6STP2 FROM D6DRV3. IF WE REACHED HERE AS
C               A RESULT OF THE FIRST CALL TO D6STP2 IN D6DRV3,
C               EXTRAPOLATION OF THE MOST RECENT POLYNOMIAL IS
C               ALLOWED FOLLOWING A SUCCESSFUL STEP FOR WHICH A
C               DELAY FALLS BEYOND THE HISTORY QUEUE ON THE NEXT
C               STEP.
C
              IF (ICHECK.NE.2) THEN
                 CALL D6MESG(27,'D6SRC3')
                 INDEX = -11
                 GO TO 9005
              ENDIF
C             ICHECK = 2 ...
              INDEXO = INEW
              TOLD = QUEUE(1,INEW)
              TNEW = QUEUE(2,INEW)
              IF (IVAN.EQ.0) INDEX = -11
              GO TO 9005
C
              ENDIF
C
          ENDIF
C
      ENDIF
C
C     HANDLE THE CASES IN WHICH THERE IS ONLY ONE POINT
C     IN THE QUEUE.
C
      IF ((NUMPT.EQ.1) .OR. ((IOLD.EQ.1).AND. (INEW.EQ.1))) THEN
C
          INDEXO = 1
          TOLD = QUEUE(1,1)
          TNEW = QUEUE(2,1)
          GO TO 9005
C
      ENDIF
C
C     WE GET HERE ONLY IF THERE IS MORE THAN ONE POINT IN THE
C     QUEUE AND QUEUE(1,IOLD) .LE. TVAL .LT. QUEUE(1,INEW) .
C
C     HANDLE THE CASE IN WHICH (1=) IOLD < INEW (SO THAT THE
C     DATA IS ALREADY ORDERED). NOTE THAT THE QUEUE MAY NOT
C     YET BE FULL.
C
      IF (IOLD.LT.INEW) THEN
C
          JORDER = 0
          CALL D6SRC1(N,QUEUE,HNEXT,JORDER,IOLD,INEW,NUMPT,INDEXO,TVAL)
          TOLD = QUEUE(1,INDEXO)
          TNEW = QUEUE(2,INDEXO)
          GO TO 9005
C
      ENDIF
C
C     HANDLE THE CASE IN WHICH INEW < IOLD (=INEW+1) SO THAT
C     THE DATA IS CIRCULARLY ORDERED. NOTE THAT THE QUEUE
C     MUST BE FULL IN THIS CASE.
C
      IF (IOLD.GT.INEW) THEN
C
          JORDER = 1
          CALL D6SRC1(N,QUEUE,HNEXT,JORDER,IOLD,INEW,NUMPT,INDEXO,TVAL)
          TOLD = QUEUE(1,INDEXO)
          TNEW = QUEUE(2,INDEXO)
C
      ENDIF
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6SRC4
      SUBROUTINE D6SRC4(N,TVAL,QUEUE,NUMPT,IOLD,INEW,HNEXT,INDEXO,TOLD,
     +                  TNEW,INDEX)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C     This is a modification of the search routine D6SRC2 which allows
C     extrapolation of the solution beyond the interval spanned by the
C     solution history queue. It is called only by subroutine D6INT4
C     (which is not called by DKLAG6).
C
C***BEGIN PROLOGUE  D6SRC4
C***SUBSIDIARY
C***PURPOSE  The purpose of D6SRC4 is to assist in the queue
C            search and interpolation for the DKLAG6 differential
C            equation solver (for non-vanishing delay problems).
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
C***ROUTINES CALLED  D1MACH, D6SRC1, D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6SRC4
C
C     SEARCH THE HISTORY QUEUE FOR D6DRV3 TO LOCATE THE
C     SLUG OF DATA TO BE INTERPOLATED FOR THIS VALUE
C     OF TVAL.
C
C     CAUTION:
C     THIS ROUTINE IS NOT INTENDED TO BE CALLED WITH
C     HNEXT=0.0D0 OR INEW=0. (THE LATTER CASE WOULD
C     CORRESPOND TO THE FIRST STEP OF THE INTEGRATION
C     AND IS HANDLED SEPARATELY IN SUBROUTINE D6INT1.
C     THE FORMER SHOULD NEVER OCCUR AND SHOULD BE
C     TREATED AS AN ERROR.)
C
      IMPLICIT NONE
C
      INTEGER INDEX,INDEXO,INEW,IOLD,JORDER,N,NUMPT
      DOUBLE PRECISION D1MACH,HNEXT,QUEUE,TNEW,TOLD,TVAL
C
      DIMENSION QUEUE(N+1,*)
C
C***FIRST EXECUTABLE STATEMENT  D6SRC4
C
      INDEX = 2
C
C     INEW POINTS TO THE MOST RECENT ADDITION TO THE
C     QUEUE; IOLD TO THE OLDEST.
C
C     CHECK FOR EXTRAPOLATION OUTSIDE THE QUEUE. ALSO
C     HANDLE THE CASE IN WHICH TVAL = TNEW.
C
      IF (HNEXT.GT.0.0D0) THEN
C
C         IF (TVAL.LT.QUEUE(1,IOLD)) THEN
C
          IF ((TVAL-QUEUE(1,IOLD)) .LT.
     +    (-1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(1,IOLD)))))
     +    THEN
C
C             **********
C             ** NOTE **
C             **********
C
C             THE FOLLOWING STATEMENTS WOULD ALLOW
C             EXTRAPOLATION OF THE OLDEST POLYNOMIAL.
C
C             CALL D6MESG(17,'D6SRC4')
              INDEXO = IOLD
              TOLD = QUEUE(1,IOLD)
              TNEW = QUEUE(2,IOLD)
              INDEX = 0
              GO TO 9005
C
C             THESE STATEMENTS WOULD CAUSE AN ERROR RETURN.
C
C             CALL D6MESG(26,'D6SRC4')
C             INDEX = 0
C             GO TO 9005
C
          ENDIF
C
          IF (TVAL.GE.QUEUE(2,INEW)) THEN
C
C            IF (TVAL.GT.QUEUE(2,INEW)) THEN
C
C            IF ((TVAL-QUEUE(2,INEW)) .GT.
C    +       (1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(2,INEW)))))
C    +       THEN
C
C               **********
C               ** NOTE **
C               **********
C               IF THE FOLLOWING THREE STATEMENTS ARE REMOVED, THE
C               MOST RECENT POLYNOMIAL WILL BE EXTRAPOLATED.
C
C               IF (ICHECK .NE.2)
C    +          CALL D6MESG(27,'D6SRC4')
C               INDEX = 1
C               GO TO 9005
C
C             ENDIF
C
C             THESE STATEMENTS ALLOW THE EXTRAPOLATION.
C
              INDEXO = INEW
              TOLD = QUEUE(1,INEW)
              TNEW = QUEUE(2,INEW)
              INDEX = 1
              GO TO 9005
C
          ENDIF
C
      ENDIF
C
      IF (HNEXT.LT.0.0D0) THEN
C
C         IF (TVAL.GT.QUEUE(1,IOLD)) THEN
C
          IF ((TVAL-QUEUE(1,IOLD)) .GT.
     +    (1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(1,IOLD)))))
     +    THEN
C
C             **********
C             ** NOTE **
C             **********
C
C             THE FOLLOWING STATEMENTS WOULD ALLOW
C             EXTRAPOLATION OF THE OLDEST POLYNOMIAL.
C
C             CALL D6MESG(17,'D6SRC4')
              INDEXO = IOLD
              TOLD = QUEUE(1,IOLD)
              TNEW = QUEUE(2,IOLD)
              INDEX = 0
              GO TO 9005
C
C             THE FOLLOWING STATEMENTS WOULD GENERATE AN
C             ERROR RETURN.
C
C             CALL D6MESG(26,'D6SRC4')
C             INDEX = 0
C             GO TO 9005
C
          ENDIF
C
          IF (TVAL.LE.QUEUE(2,INEW)) THEN
C
C             IF (TVAL.LT.QUEUE(2,INEW)) THEN
C
C             IF ((TVAL-QUEUE(2,INEW)) .LT.
C    +        (-1.0D3*D1MACH(4)*MAX(ABS(TVAL),ABS(QUEUE(2,INEW)))))
C    +        THEN
C
C                 **********
C                 ** NOTE **
C                 **********
C                 IF THE FOLLOWING THREE STATEMENTS ARE REMOVED, THE
C                 MOST RECENT POLYNOMIAL WILL BE EXTRAPOLATED.
C
C                 IF (ICHECK .NE.2)
C    +            CALL D6MESG(27,'D6SRC4')
C                 INDEX = 1
C                 GO TO 9005
C
C             ENDIF
C
              INDEXO = INEW
              TOLD = QUEUE(1,INEW)
              TNEW = QUEUE(2,INEW)
              INDEX = 1
              GO TO 9005
C
          ENDIF
C
      ENDIF
C
C     HANDLE THE CASES IN WHICH THERE IS ONLY ONE POINT
C     IN THE QUEUE.
C
      IF ((NUMPT.EQ.1) .OR. ((IOLD.EQ.1).AND. (INEW.EQ.1))) THEN
C
          INDEXO = 1
          TOLD = QUEUE(1,1)
          TNEW = QUEUE(2,1)
          GO TO 9005
C
      ENDIF
C
C     WE GET HERE ONLY IF THERE IS MORE THAN ONE POINT IN THE
C     QUEUE AND QUEUE(1,IOLD) .LE. TVAL .LT. QUEUE(1,INEW) .
C
C     HANDLE THE CASE IN WHICH (1=) IOLD < INEW (SO THAT THE
C     DATA IS ALREADY ORDERED). NOTE THAT THE QUEUE MAY NOT
C
C
      IF (IOLD.LT.INEW) THEN
C
          JORDER = 0
          CALL D6SRC1(N,QUEUE,HNEXT,JORDER,IOLD,INEW,NUMPT,INDEXO,TVAL)
          TOLD = QUEUE(1,INDEXO)
          TNEW = QUEUE(2,INDEXO)
          GO TO 9005
C
      ENDIF
C
C     HANDLE THE CASE IN WHICH INEW < IOLD (=INEW+1) SO THAT
C     THE DATA IS CIRCULARLY ORDERED. NOTE THAT THE QUEUE
C     MUST BE FULL IN THIS CASE.
C
      IF (IOLD.GT.INEW) THEN
C
          JORDER = 1
          CALL D6SRC1(N,QUEUE,HNEXT,JORDER,IOLD,INEW,NUMPT,INDEXO,TVAL)
          TOLD = QUEUE(1,INDEXO)
          TNEW = QUEUE(2,INDEXO)
C
      ENDIF
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6GRT1
      SUBROUTINE D6GRT1(N,TOLD,YOLD,YNEW,DYNEW,TROOT,YROOT,DYROOT,TNEW,
     +                  NG,NGUSER,GRESID,GUSER,JROOT,HINTP,INDEX,IER,
     +                  NGEVAL,GRTOL,K0,K2,K3,K4,K5,K6,K7,K8,K9,GTOLD,
     +                  GTNEW,GROOT,GA,GB,TG,INDEXG,BETA,BVAL,TINTP,
     +                  IORDER,U10,IPAIR,RPAR,IPAR)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6GRT1
C***SUBSIDIARY
C***PURPOSE  The purpose of D6GRT1 is to direct the rootfinding for
C            the DKLAG6 differential equation solver. It is called
C            by subroutine D6DRV3 to perform extrapolatory
C            rootfinding.
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
C***ROUTINES CALLED  DCOPY, D6POLY, D6GRT2, D6MESG, D6RTOL
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6GRT1
C
C     INTERFACE CALLED BY D6DRV3 TO DIRECT THE D6GRT2 ROOTFINDER.
C
      IMPLICIT NONE
C
      INTEGER I,IDUM1,IDUM2,IER,INDEX,INDEXG,IORDER,JFLAG,JROOT,N,NG,
     +        NGEMAX,NGEVAL,NGUSER,IPAR,NSIG,ICOMP,IPAIR,IRCALL
      DOUBLE PRECISION BNEW,BOLD,BVAL,DYNEW,DYROOT,GA,GB,GROOT,GRTOL,
     +                 GTNEW,GTOL,GTOLD,HDUM1,HDUM2,HINTP,K0,K2,K3,
     +                 K4,K5,K6,K7,K8,K9,TG,TINTP,TNEW,TOLD,TROOT,
     +                 U10,YNEW,YOLD,YROOT,RPAR
C
      DIMENSION YOLD(*),YNEW(*),DYNEW(*),YROOT(*),DYROOT(*),JROOT(*),
     +          K0(*),K2(*),K3(*),K4(*),K5(*),K6(*),K7(*),K8(*),K9(*),
     +          BVAL(*),GTOLD(*),GTNEW(*),GROOT(*),GA(*),GB(*),TG(*),
     +          INDEXG(*),RPAR(*),IPAR(*),ICOMP(1)
C
      EXTERNAL GRESID,GUSER,BETA
C
C***FIRST EXECUTABLE STATEMENT  D6GRT1
C
      INDEX = 2
      IER = 0
      NSIG = 0
      NGEMAX = 500
      IRCALL = 0
      CALL D6RTOL(NSIG,NGEMAX,U10)
C
C     INITIALIZE THE ROOTFINDER.
C
      BOLD = TOLD
      BNEW = TNEW
      DO 10 I = 1,NG
          JROOT(I) = 0
   10 CONTINUE
      JFLAG = 0
      GTOL = GRTOL*MAX(ABS(TOLD),ABS(TNEW))
      GTOL = MAX(GTOL,U10)
C
C     LOAD THE RESIDUALS AT THE ENDPOINTS.
C
      CALL DCOPY(NG,GTOLD,1,GA,1)
      CALL DCOPY(NG,GTNEW,1,GB,1)
C
C     CALL THE ROOTFINDER.
C
   20 CONTINUE
C
      CALL D6GRT2(NG,GTOL,JFLAG,BOLD,BNEW,GA,GB,GROOT,TROOT,JROOT,HDUM1,
     +            HDUM2,IDUM1,IDUM2,IRCALL)
C
C     FOLLOW THE INSTRUCTIONS FROM THE ROOTFINDER.
C
      GO TO (30,40,60,80),JFLAG
C
   30 CONTINUE
C
C     EVALUATE THE RESIDUALS FOR THE ROOTFINDER.
C
C     FIRST APPROXIMATE THE SOLUTION AND THE DERIVATIVE ....
C
      CALL D6POLY(N,TINTP,YOLD,HINTP,K0,K2,K3,K4,K5,K6,
     +            K7,K8,K9,YROOT,DYROOT,TROOT,IORDER,
     +            ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,11)
C
C     NOW CALCULATE THE RESIDUALS ....
C
      CALL GRESID(N,TROOT,YROOT,DYROOT,NG,NGUSER,GUSER,GROOT,
     +            BETA,BVAL,INDEXG,TG,RPAR,IPAR)
      IF (N.LE.0) GO TO 75
C
C     AND UPDATE THE RESIDUAL COUNTER ....
C
      NGEVAL = NGEVAL + 1
C
C     CHECK FOR TOO MANY RESIDUAL EVALUATIONS FOR THIS
C     CALL TO D6GRT1.
C
      IF (NGEVAL.GT.NGEMAX) THEN
          INDEX = -10
          GO TO 90
      ENDIF
C
C     CALL THE ROOTFINDER AGAIN.
C
      GO TO 20
C
   40 CONTINUE
C
C     A ROOT WAS FOUND.
C
C     APPROXIMATE THE SOLUTION AND THE DERIVATIVE
C     AT THE ROOT.
C
      CALL D6POLY(N,TINTP,YOLD,HINTP,K0,K2,K3,K4,K5,K6,
     +            K7,K8,K9,YROOT,DYROOT,TROOT,IORDER,
     +            ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,12)
C
C     SET THE FLAG TO INDICATE A ROOT FOUND.
C
      INDEX = 3
C
C     LOAD AN EXACT ZERO FOR THE RESIDUAL.
C
      DO 50 I = 1,NG
          IF (JROOT(I).EQ.1) GROOT(I) = 0.0D0
   50 CONTINUE
C
C     RETURN TO D6DRV3.
C
      GO TO 90
C
   60 CONTINUE
C
C     THE RIGHT ENDPOINT IS A ROOT.
C
C     LOAD THE SOLUTION AND DERIVATIVE
C     INTO YROOT AND DYROOT.
C
      CALL DCOPY(N,YNEW,1,YROOT,1)
      CALL DCOPY(N,DYNEW,1,DYROOT,1)
C
C     SET THE FLAG TO INDICATE A ROOT WAS FOUND.
C
      INDEX = 4
C
C     LOAD AN EXACT ZERO FOR THE RESIDUAL.
C
      DO 70 I = 1,NG
          IF (JROOT(I).EQ.1) GROOT(I) = 0.0D0
   70 CONTINUE
C
C     RETURN TO D6DRV3.
C
      GO TO 90
C
C     THE USER TERMINATED THE INTEGRATION.
C
   75 CONTINUE
      INDEX = -14
      CALL D6MESG(25,'D6GRT1')
      RETURN
C
   80 CONTINUE
C
C     NO SIGN CHANGES WERE DETECTED.
C
C     SET THE FLAG TO INDICATE NO SIGN CHANGES WERE DETECTED.
C
      INDEX = 2
C
C     RETURN TO D6DRV3.
C
   90 CONTINUE
C
      RETURN
      END
C*DECK D6GRT2
      SUBROUTINE D6GRT2(NG,HMIN,JFLAG,X0,X1,G0,G1,GX,X,JROOT,ALPHA,X2,
     +                  IMAX,LAST,IRCALL)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6GRT2
C***SUBSIDIARY
C***PURPOSE  The purpose of D6GRT2 is to perform rootfinding for the
C            DKLAG6 differential equation solver. D6GRT2 is the
C            rootfinder used in the LSODAR ordinary differential
C            equation solver. It is called by subroutine D6GRT1.
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
c
c Note:
c This subroutine should never be called directly
c by the user.
C
C THIS ROUTINE FINDS THE LEFTMOST ROOT OF A SET OF ARBITRARY
C FUNCTIONS GI(X) (I = 1,...,NG) IN AN INTERVAL (X0,X1). USUALLY ONLY
C ROOTS OF ODD MULTIPLICITY (I.E. CHANGES OF SIGN OF THE GI) ARE FOUND.
C HERE THE SIGN OF X1 - X0 IS ARBITRARY, BUT IS CONSTANT FOR A GIVEN
C PROBLEM, AND LEFTMOST MEANS NEAREST TO X0.
C THE VALUES OF THE VECTOR-VALUED FUNCTION G(X) = (G(I), I=1,...,NG)
C ARE COMMUNICATED THROUGH THE CALL SEQUENCE OF D6GRT2.
C THE METHOD USED IS THE ILLINOIS ALGORITHM.
C
C NOTE:
C THIS ROUTINE IS THE PETZOLD/SHAMPINE ROOTFINDER USED IN THE LSODAR
C PACKAGE FOR THE AUTOMATIC SOLUTION OF INITIAL VALUE PROBLEMS IN
C ORDINARY DIFFERENTIAL EQUATIONS.
C
C CAUTION:
C THE ORIGINAL SUBROUTINE WAS MODIFIED TO ELIMINATE THE USE OF COMMON
C BLOCKS.
C
C A FURTHER MODIFICATION IS THE USE OF THE ADDED PARAMETER IRCALL.
C D6GRT2 IS CALLED WITH THE VALUES 0, 1, 2 CORRESPONDING RESPECTIVELY
C TO EXTRAPOLATORY ROOTFINDING, INTERPOLATORY ROOTFINDING ON
C AN INTERVAL (TOLD,TNEW), AND INTERPOLATORY ROOTFINDING ON
C AN INTERVAL (TROOT,TNEW) WHERE TROOT IS THE ROOT LOCATED WHEN
C THE IRCALL = 1 CALL IS MADE. WHEN CALLED WITH IRCALL = 2, D6GRT2
C DOES NOT ALTER THE JROOT(*) ARRAY.
C
C DESCRIPTION OF PARAMETERS.
C
C NG     = NUMBER OF FUNCTIONS GI, OR THE NUMBER OF COMPONENTS OF
C          THE VECTOR VALUED FUNCTION G(X).  INPUT ONLY.
C
C HMIN   = RESOLUTION PARAMETER IN X.  INPUT ONLY.  WHEN A ROOT IS
C          FOUND, IT IS LOCATED ONLY TO WITHIN AN ERROR OF HMIN IN X.
C          TYPICALLY, HMIN SHOULD BE SET TO SOMETHING ON THE ORDER OF
C                10.0D0 * DROUND * MAX(ABS(X0),ABS(X1)),
C          WHERE DROUND IS THE UNIT ROUNDOFF OF THE MACHINE.
C
C JFLAG  = INTEGER FLAG FOR INPUT AND OUTPUT COMMUNICATION.
C
C          ON INPUT, SET JFLAG = 0 ON THE FIRST CALL FOR THE PROBLEM,
C          AND LEAVE IT UNCHANGED UNTIL THE PROBLEM IS COMPLETED.
C          (THE PROBLEM IS COMPLETED WHEN JFLAG .GE. 2 ON RETURN.)
C
C          ON OUTPUT, JFLAG HAS THE FOLLOWING VALUES AND MEANINGS -
C          JFLAG = 1 MEANS D6GRT2 NEEDS A VALUE OF G(X).  SET GX = G(X)
C                    AND CALL D6GRT2 AGAIN.
C          JFLAG = 2 MEANS A ROOT HAS BEEN FOUND.  THE ROOT IS
C                    AT X, AND GX CONTAINS G(X).  (ACTUALLY, X IS THE
C                    RIGHTMOST APPROXIMATION TO THE ROOT ON AN INTERVAL
C                    (X0,X1) OF SIZE HMIN OR LESS.)
C          JFLAG = 3 MEANS X = X1 IS A ROOT, WITH ONE OR MORE OF THE GI
C                    BEING ZERO AT X1 AND NO SIGN CHANGES IN (X0,X1).
C                    GX CONTAINS G(X) ON OUTPUT.
C          JFLAG = 4 MEANS NO ROOTS (OF ODD MULTIPLICITY) WERE
C                    FOUND IN (X0,X1) (NO SIGN CHANGES).
C
C X0,X1  = ENDPOINTS OF THE INTERVAL WHERE ROOTS ARE SOUGHT.
C          X1 AND X0 ARE INPUT WHEN JFLAG = 0 (FIRST CALL), AND
C          MUST BE LEFT UNCHANGED BETWEEN CALLS UNTIL THE PROBLEM IS
C          COMPLETED.  X0 AND X1 MUST BE DISTINCT, BUT X1 - X0 MAY BE
C          OF EITHER SIGN.  HOWEVER, THE NOTION OF LEFT AND RIGHT
C          WILL BE USED TO MEAN NEARER TO X0 OR X1, RESPECTIVELY.
C          WHEN JFLAG .GE. 2 ON RETURN, X0 AND X1 ARE OUTPUT, AND
C          ARE THE ENDPOINTS OF THE RELEVANT INTERVAL.
C
C G0,G1  = ARRAYS OF LENGTH NG CONTAINING THE VECTORS G(X0) AND G(X1),
C          RESPECTIVELY.  WHEN JFLAG = 0, G0 AND G1 ARE INPUT AND
C          NONE OF THE G0(I) SHOULD BE BE ZERO.
C          WHEN JFLAG .GE. 2 ON RETURN, G0 AND G1 ARE OUTPUT.
C
C GX     = ARRAY OF LENGTH NG CONTAINING G(X).  GX IS INPUT
C          WHEN JFLAG = 1, AND OUTPUT WHEN JFLAG .GE. 2.
C
C X      = INDEPENDENT VARIABLE VALUE.  OUTPUT ONLY.
C          WHEN JFLAG = 1 ON OUTPUT, X IS THE POINT AT WHICH G(X)
C          IS TO BE EVALUATED AND LOADED INTO GX.
C          WHEN JFLAG = 2 OR 3, X IS THE ROOT.
C          WHEN JFLAG = 4, X IS THE RIGHT ENDPOINT OF THE INTERVAL, X1.
C
C JROOT  = INTEGER ARRAY OF LENGTH NG.  OUTPUT ONLY.
C          WHEN JFLAG = 2 OR 3, JROOT INDICATES WHICH COMPONENTS
C          OF G(X) HAVE A ROOT AT X.  JROOT(I) IS 1 IF THE I-TH
C          COMPONENT HAS A ROOT, AND JROOT(I) = 0 OTHERWISE.
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  DCOPY
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6GRT2
C
      IMPLICIT NONE
C
      INTEGER I,IMAX,IMXOLD,JFLAG,JROOT,LAST,NG,NXLAST,IRCALL
      DOUBLE PRECISION ALPHA,G0,G1,GX,HMIN,T2,TMAX,X,X0,X1,X2
      LOGICAL BZROOT,SGNCHG,BXROOT
C
      DIMENSION G0(*),G1(*),GX(*),JROOT(*)
C
C***FIRST EXECUTABLE STATEMENT  D6GRT2
C
      IF (JFLAG.EQ.1) GO TO 90
C
C JFLAG .NE. 1.  CHECK FOR CHANGE IN SIGN OF G OR ZERO AT X1.
C
      IMAX = 0
      TMAX = 0.0D0
      BZROOT = .FALSE.
      DO 20 I = 1,NG
          IF (ABS(G1(I)).GT.0.0D0) GO TO 10
          BZROOT = .TRUE.
          GO TO 20
C
C AT THIS POINT, G0(I) HAS BEEN CHECKED AND CANNOT BE ZERO.
C
   10     IF (SIGN(1.0D0,G0(I)).EQ.SIGN(1.0D0,G1(I))) GO TO 20
          T2 = ABS(G1(I)/ (G1(I)-G0(I)))
          IF (T2.LE.TMAX) GO TO 20
          TMAX = T2
          IMAX = I
   20 CONTINUE
      IF (IMAX.GT.0) GO TO 30
      SGNCHG = .FALSE.
      GO TO 40
   30 SGNCHG = .TRUE.
   40 IF (.NOT.SGNCHG) GO TO 200
C
C THERE IS A SIGN CHANGE.  FIND THE FIRST ROOT IN THE INTERVAL.
C
      BXROOT = .FALSE.
      LAST = 1
      NXLAST = 0
C
C REPEAT UNTIL THE FIRST ROOT IN THE INTERVAL IS FOUND.  LOOP POINT.
C
   50 IF (BXROOT) GO TO 170
      IF (NXLAST.EQ.LAST) GO TO 60
      ALPHA = 1.0D0
      GO TO 80
   60 IF (LAST.EQ.0) GO TO 70
      ALPHA = 0.5D0*ALPHA
      GO TO 80
   70 ALPHA = 2.0D0*ALPHA
   80 X2 = X1 - (X1-X0)*G1(IMAX)/ (G1(IMAX)-ALPHA*G0(IMAX))
      IF ((ABS(X2-X0).LT.HMIN) .AND.
     +    (ABS(X1-X0).GT.10.0D0*HMIN)) X2 = X0 + 0.1D0* (X1-X0)
      JFLAG = 1
      X = X2
C
C RETURN TO THE CALLING ROUTINE TO GET A VALUE OF GX = G(X).
C
      RETURN
C
   90 IMXOLD = IMAX
C
C CHECK TO SEE IN WHICH INTERVAL G CHANGES SIGN.
C
      IMAX = 0
      TMAX = 0.0D0
      BZROOT = .FALSE.
      DO 110 I = 1,NG
          IF (ABS(GX(I)).GT.0.0D0) GO TO 100
          BZROOT = .TRUE.
          GO TO 110
C
C NEITHER G0(I) NOR GX(I) CAN BE ZERO AT THIS POINT.
C
  100     IF (SIGN(1.0D0,G0(I)).EQ.SIGN(1.0D0,GX(I))) GO TO 110
          T2 = ABS(GX(I)/ (GX(I)-G0(I)))
          IF (T2.LE.TMAX) GO TO 110
          TMAX = T2
          IMAX = I
  110 CONTINUE
      IF (IMAX.GT.0) GO TO 120
      SGNCHG = .FALSE.
      IMAX = IMXOLD
      GO TO 130
  120 SGNCHG = .TRUE.
  130 NXLAST = LAST
      IF (.NOT.SGNCHG) GO TO 140
C
C SIGN CHANGE BETWEEN X0 AND X2, SO REPLACE X1 WITH X2.
C
      X1 = X2
      CALL DCOPY(NG,GX,1,G1,1)
      LAST = 1
      BXROOT = .FALSE.
      GO TO 160
  140 IF (.NOT.BZROOT) GO TO 150
C
C ZERO VALUE AT X2 AND NO SIGN CHANGE IN (X0,X2), SO X2 IS A ROOT.
C
      X1 = X2
      CALL DCOPY(NG,GX,1,G1,1)
      BXROOT = .TRUE.
      GO TO 160
C
C NO SIGN CHANGE BETWEEN X0 AND X2.  REPLACE X0 WITH X2.
C
  150 CONTINUE
      CALL DCOPY(NG,GX,1,G0,1)
      X0 = X2
      LAST = 0
      BXROOT = .FALSE.
  160 IF (ABS(X1-X0).LE.HMIN) BXROOT = .TRUE.
      GO TO 50
C
C RETURN WITH X1 AS THE ROOT.  SET JROOT.  SET X = X1 AND GX = G1.
C
  170 JFLAG = 2
      X = X1
      CALL DCOPY(NG,G1,1,GX,1)
      DO 190 I = 1,NG
          IF (IRCALL .NE. 2) JROOT(I) = 0
          IF (ABS(G1(I)).GT.0.0D0) GO TO 180
          IF (IRCALL .NE. 2) JROOT(I) = 1
          GO TO 190
  180     IF ((SIGN(1.0D0,G0(I)).NE.SIGN(1.0D0,G1(I)))
     +       .AND. (IRCALL.NE.2)) JROOT(I) = 1
  190 CONTINUE
      RETURN
C
C NO SIGN CHANGE IN THE INTERVAL.  CHECK FOR ZERO AT RIGHT ENDPOINT.
C
  200 IF (.NOT.BZROOT) GO TO 220
C
C ZERO VALUE AT X1 AND NO SIGN CHANGE IN (X0,X1).  RETURN JFLAG = 3.
C
      X = X1
      CALL DCOPY(NG,G1,1,GX,1)
      DO 210 I = 1,NG
          IF (IRCALL .NE. 2) JROOT(I) = 0
          IF ((ABS(G1(I)).LE.0.0D0) .AND. (IRCALL.NE.2))
     +    JROOT(I) = 1
  210 CONTINUE
      JFLAG = 3
      RETURN
C
C NO SIGN CHANGES IN THIS INTERVAL.  SET X = X1, RETURN JFLAG = 4.
C
  220 CALL DCOPY(NG,G1,1,GX,1)
      X = X1
      JFLAG = 4
C
      RETURN
      END
C*DECK D6GRT3
      SUBROUTINE D6GRT3(N,T,Y,DY,NG,NGUSER,GUSER,G,BETA,
     +                  BVAL,INDEXG,TG,RPAR,IPAR)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6GRT3
C***SUBSIDIARY
C***PURPOSE  The purpose of D6GRT3 is to evaluate the residuals
C            for the rootfinder for the DKLAG6 differential
C            equation solver. It is called by subroutine D6GRT1
C            in conjunction with the D6GRT2 rootfinder.
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
C     N         = NUMBER OF ODES
C     T         = INDEPENDENT VARIABLE (TIME)
C     Y(*)      = SOLUTION AT TIME T
C     DY(*)     = APPROXIMATION TO DERIVATIVE AT TIME T
C                 (NOT EXACT DERIVATIVE FROM DERIVS)
C     NG        = NUMBER OF EVENT FUNCTIONS
C     NGUSER    = NUMBER OF USER DEFINED ROOT FUNCTIONS
C     GUSER     = USER PROVIDED EXTERNAL SUBROUTINE TO
C                 EVALUATE THE RESIDUALS FOR THE USER ROOT
C                 FUNCTIONS
C     BETA      = NAME OF SUBROUTINE IN WHICH TO EVALUATE DELAYS
C     BVAL(*)   = ARRAY TO HOLD DELAYS
C     INDEXG(*) = COMPONENT NUMBER INDICATOR FOR THE ROOT
C                 FUNCTIONS
C     TG(*)     = ARRAY TO HOLD THE PREVIOUSLY CALCULATED ROOTS
C
C     RPAR(*),  = USER ARRAYS PASSED VIA DKLAG6
C     IPAR(*)
C
C***SEE ALSO  DKLAG6
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D6MESG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6GRT3
C
C     THIS SUBROUTINE MUST DEFINE THE EVENT FUNCTION
C     RESIDUALS G(I), I=1 ,..., NG.
C
      IMPLICIT NONE
C
      INTEGER I,N,NG,NGUSER,NGUSP1,INDEXG,IPAR
      DOUBLE PRECISION BVAL,DY,G,TG,Y,RPAR,T
C
      DIMENSION Y(*),DY(*),G(*),INDEXG(*),TG(*),BVAL(*),
     +          RPAR(*),IPAR(*)
C
      EXTERNAL BETA,GUSER
C
C     EVALUATE THE RESIDUALS FOR THE USER ROOT FUNCTIONS.
C
C***FIRST EXECUTABLE STATEMENT  D6GRT3
C
      IF (NGUSER.GT.0) THEN
         CALL GUSER(N,T,Y,DY,NGUSER,G,RPAR,IPAR)
         IF (N.LE.0) GO TO 9005
      ENDIF
C
C     EVALUATE THE RESIDUALS FOR THE DISCONTINUITY ROOT
C     FUNCTIONS.
C
      IF (NG.GT.NGUSER) THEN
C
C        EVALUATE THE SYSTEM DELAYS.
C
         CALL BETA(N,T,Y,BVAL,RPAR,IPAR)
         IF (N.LE.0) GO TO 9005
C
C        EVALUATE THE ROOT FUNCTION RESIDUALS.
C
         NGUSP1 = NGUSER+1
         DO 20 I = NGUSP1,NG
             G(I) = BVAL(INDEXG(I)) - TG(I)
   20    CONTINUE
C
      ENDIF
C
 9005 CONTINUE
C
      IF (N.LE.0) THEN
         CALL D6MESG(25,'D6GRT3')
      ENDIF
C
      RETURN
      END
C*DECK D6GRT4
      SUBROUTINE D6GRT4(WORK,NG,K,JROOT,INDEXG,LEVEL,TROOT,
     +                  IPOINT,LEVMAX,ANCESL)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6GRT4
C***SUBSIDIARY
C***PURPOSE  The purpose of D6GRT4 is to shuffle storage for
C            the rootfinder for the DKLAG6 differential
C            equation solver when new residual functions
C            are added (when zeroes of the discontinuity
C            delay functions are found).
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
C***ROUTINES CALLED  D6PNTR
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6GRT4
C
C     SHUFFLE STORAGE FOR THE ROOTFINDER IN D6DRV3.
C
C     CAUTION:
C     D6GRT4 ASSUMES THAT BEFORE IT IS CALLED (LIKE IN
C     D6DRV2) K IS SET TO A POSITIVE NUMBER AND THAT
C     STORAGE WAS CHECKED TO MAKE SURE THERE IS ENOUGH
C     SPACE TO MAKE THE NECESSARY CHANGES IN THE WORK,
C     JROOT, AND INDEXG VECTORS.
C
      IMPLICIT NONE
C
      INTEGER I,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,I9N,I10N,I11N,I12N,I13N,I14N,I15N,IDIR,IERRUS,
     +        IFIRST,IFOUND,IORDER,IQUEUE,RSTRT,JN,JO,JQUEUE,KQUEUE,
     +        LQUEUE,N,NGNEW,K,NG,INDEXG,IPOINT,JROOT,IK7,IK8,IK9,IPAIR,
     +        LEVEL,LEVMAX,ANCESL
      DOUBLE PRECISION WORK,TINIT,TROOT
C
      DIMENSION WORK(*),JROOT(*),INDEXG(*),LEVEL(*),IPOINT(*)
C
C     RESTORE THE POINTERS AND INTEGRATION FLAGS.
C
C***FIRST EXECUTABLE STATEMENT  D6GRT4
C
      CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,
     +            I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,IFOUND,
     +            IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +            TINIT,WORK,IK7,IK8,IK9,IPAIR,3)
C
C     THE NEW NUMBER OF ROOT FUNCTIONS WILL BE ...
C
      NGNEW = NG + K
C
C     THE NEW LOCATIONS OF THE G VECTORS AND THE
C     TG VECTOR IN WORK WILL BE ...
C
      I9N = I9
      I10N = I9N + NGNEW
      I11N = I10N + NGNEW
      I12N = I11N + NGNEW
      I13N = I12N + NGNEW
      I14N = I13N + NGNEW
      I15N = I14N + NGNEW
C
C     MOVE THE OLD TG VECTOR TO ITS NEW LOCATION.
C
      JN = I15N + NG
      JO = I15 + NG
      DO 10 I = 1,NG
          WORK(JN-I) = WORK(JO-I)
   10 CONTINUE
C
C     MOVE THE OLD G VECTORS TO THEIR NEW LOCATIONS.
C
      JN = I14N + NG
      JO = I14 + NG
      DO 20 I = 1,NG
          WORK(JN-I) = WORK(JO-I)
   20 CONTINUE
C
      JN = I13N + NG
      JO = I13 + NG
      DO 30 I = 1,NG
          WORK(JN-I) = WORK(JO-I)
   30 CONTINUE
C
      JN = I12N + NG
      JO = I12 + NG
      DO 40 I = 1,NG
          WORK(JN-I) = WORK(JO-I)
   40 CONTINUE
C
      JN = I11N + NG
      JO = I11 + NG
      DO 50 I = 1,NG
          WORK(JN-I) = WORK(JO-I)
   50 CONTINUE
C
      JN = I10N + NG
      JO = I10 + NG
      DO 60 I = 1,NG
          WORK(JN-I) = WORK(JO-I)
   60 CONTINUE
C
      JN = I9N + NG
      JO = I9 + NG
      DO 70 I = 1,NG
          WORK(JN-I) = WORK(JO-I)
   70 CONTINUE
C
C     LOAD TROOT INTO THE NEW TG SLOTS.
C
      JN = I15N + NGNEW
      DO 80 I = 1,K
          WORK(JN-I) = TROOT
   80 CONTINUE
C
C     LOAD EXACT ZEROES INTO THE NEW RESIDUAL SLOTS.
C
      JN = I14N + NGNEW
      DO 90 I = 1,K
          WORK(JN-I) = 0.0D0
   90 CONTINUE
C
      JN = I13N + NGNEW
      DO 100 I = 1,K
          WORK(JN-I) = 0.0D0
  100 CONTINUE
C
      JN = I12N + NGNEW
      DO 110 I = 1,K
          WORK(JN-I) = 0.0D0
  110 CONTINUE
C
      JN = I11N + NGNEW
      DO 120 I = 1,K
          WORK(JN-I) = 0.0D0
  120 CONTINUE
C
      JN = I10N + NGNEW
      DO 130 I = 1,K
          WORK(JN-I) = 0.0D0
  130 CONTINUE
C
      JN = I9N + NGNEW
      DO 140 I = 1,K
          WORK(JN-I) = 0.0D0
  140 CONTINUE
C
C     FLAG THE ODE COMPONENT NUMBERS CORRESPONDING TO
C     THE NEW ROOT FUNCTIONS (THOSE WITH A ROOT AT
C     TROOT).
C
C     IG = 0
C
C     DO 150 I = 1,NG
C         IF (JROOT(I).EQ.1) THEN
C         IF (JROOT(I).EQ.1 .AND. INDEXG(I).NE.0) THEN
C             IG = IG + 1
C             INDEXG(NG+IG) = INDEXG(I)
C         ENDIF
C 150 CONTINUE
C
C     NOTE: K = N
C
      DO 150 I = 1,K
C BUG:   INDEXG(NG+I) = INDEXG(I)
         INDEXG(NG+I) = I
         JROOT(NG+I) = 1
         LEVEL(NG+I) = ANCESL + 1
  150 CONTINUE
C
      RETURN
      END
C*DECK D6PNTR
      SUBROUTINE D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                  I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                  IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,JQUEUE,
     +                  KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,ITASK)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6PNTR
C***SUBSIDIARY
C***PURPOSE  The purpose of D6PNTR is to define and restore work
C            array pointers and integration flags for the DKLAG6
C            differential equation solver.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6PNTR
C
C     SAVE AND RESTORE THE POINTERS AND INTEGRATION FLAGS.
C
      IMPLICIT NONE
C
      INTEGER I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I9M1,
     +        I10M1,I11M1,IDIR,IERRUS,IFIRST,IFOUND,IORDER,KINC,
     +        IPOINT,IQUEUE,RSTRT,ITASK,JQUEUE,KQUEUE,LQUEUE,N,NG,
     +        IK7,IK8,IK9,IPAIR
      DOUBLE PRECISION TINIT,WORK
C
      DIMENSION IPOINT(*)
      DIMENSION WORK(*)
C
C***FIRST EXECUTABLE STATEMENT  D6PNTR
C
      GO TO (10,20,30),ITASK
C
   10 CONTINUE
C
C     DEFINE THE POINTERS AND THE INTEGRATION FLAGS.
C
C     USE THE SIXTH ORDER METHOD FOR THE BASIC
C     INTEGRATION.
C
      IORDER = 6
C***
C     IORDER = 5
C***
C
C     INDICATE WHICH METHOD PAIR IS TO BE USED.
C
      IPAIR = 1
C
C     USE ERROR PER UNIT STEP CONTROL IF IERRUS = 0.
C     OTHERWISE USE ERROR PER STEP CONTROL.
C
      IERRUS = 1
C
C     DEFINE THE POINTERS INTO THE WORK ARRAY.
C
C     I1  -  K0
C     I2  -  K1
C     I3  -  K2
C     I4  -  K3
C     I5  -  K4
C     I6  -  K5
C     I7  -  K6
C     IK7 -  K7
C     IK8 -  K8
C     IK9 -  K9
C     I8  -  R
C     I9-1-  TINIT
C     I9  -  G(TOLD)
C     I10 -  G(TNEW)
C     I11 -  G(TROOT)
C     I12 -  GA
C     I13 -  GB
C     I14 -  G2(TROOT)
C     I15 -  TG
C
C     KINC = N
C     IF (EXTRAP) KINC = 2*N
C
C     NOTE: THE FOLLOWING CAN BE CHANGED TO KINC = N IF
C     EXTRAP = .FALSE. THE EXTRA N OF STORAGE IS USED
C     IN D6STP2 FOR THE KPRED* VECTORS.
C
      KINC = 2*N
C
      I1  = 1
      I2  = I1 + KINC 
      I3  = I2 + KINC 
      I4  = I3 + KINC 
      I5  = I4 + KINC 
      I6  = I5 + KINC
      I7  = I6 + KINC
      IK7 = I7 + KINC
      IK8 = IK7 + KINC
      IK9 = IK8 + KINC
      I8  = IK9 + KINC
      I9  = I8 + 1 + N
      I10 = I9 + NG
      I11 = I10 + NG
      I12 = I11 + NG
      I13 = I12 + NG
      I14 = I13 + NG
      I15 = I14 + NG
      I9M1 = I9 - 1
      I10M1 = I10 - 1
      I11M1 = I11 - 1
      IF (IQUEUE.EQ.0) THEN
          JQUEUE = 1
          KQUEUE = 0
          WORK(I9-1) = TINIT
          IFIRST = 0
          IFOUND = 0
      ENDIF
C
      GO TO 40
C
   20 CONTINUE
C
C     SAVE THE POINTERS AND THE INTEGRATION FLAGS.
C
      IPOINT(1) = I1
      IPOINT(2) = I2
      IPOINT(3) = I3
      IPOINT(4) = I4
      IPOINT(5) = I5
      IPOINT(6) = I6
      IPOINT(7) = I7
      IPOINT(8) = I8
      IPOINT(9) = I9
      IPOINT(10) = I10
      IPOINT(11) = I11
      IPOINT(12) = I12
      IPOINT(13) = I13
      IPOINT(14) = I9M1
      IPOINT(15) = I10M1
      IPOINT(16) = I11M1
      IPOINT(17) = IORDER
      IPOINT(18) = IERRUS
      IPOINT(20) = LQUEUE
      IPOINT(21) = IQUEUE
      IPOINT(22) = JQUEUE
      IPOINT(23) = KQUEUE
      IPOINT(24) = I14
      IPOINT(25) = I15
      IPOINT(26) = IFIRST
      IPOINT(27) = IFOUND
      IPOINT(28) = IDIR
      IPOINT(29) = RSTRT
      IPOINT(30) = NG
      IPOINT(31) = IK7
      IPOINT(32) = IK8
      IPOINT(33) = IK9
      IPOINT(34) = IPAIR 
      GO TO 40
C
   30 CONTINUE
C
C     RESTORE THE POINTERS AND THE INTEGRATION FLAGS.
C
      I1 = IPOINT(1)
      I2 = IPOINT(2)
      I3 = IPOINT(3)
      I4 = IPOINT(4)
      I5 = IPOINT(5)
      I6 = IPOINT(6)
      I7 = IPOINT(7)
      I8 = IPOINT(8)
      I9 = IPOINT(9)
      I10 = IPOINT(10)
      I11 = IPOINT(11)
      I12 = IPOINT(12)
      I13 = IPOINT(13)
      I9M1 = IPOINT(14)
      I10M1 = IPOINT(15)
      I11M1 = IPOINT(16)
      IORDER = IPOINT(17)
      IERRUS = IPOINT(18)
      LQUEUE = IPOINT(20)
      IQUEUE = IPOINT(21)
      JQUEUE = IPOINT(22)
      KQUEUE = IPOINT(23)
      I14 = IPOINT(24)
      I15 = IPOINT(25)
      IFIRST = IPOINT(26)
      IFOUND = IPOINT(27)
      IDIR = IPOINT(28)
      RSTRT = IPOINT(29)
      TINIT = WORK(I9-1)
      NG = IPOINT(30)
      IK7 = IPOINT(31)
      IK8 = IPOINT(32)
      IK9 = IPOINT(33)
      IPAIR= IPOINT(34)
C
   40 CONTINUE
C
      RETURN
      END
C*DECK D6MESG
      SUBROUTINE D6MESG(IMESG,SUBROU)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6MESG
C***SUBSIDIARY
C***PURPOSE  The purpose of D6MESG is to generate error messages
C            for the DKLAG6 differential equation solver. All
C            invocations of the SLATEC/XERROR message handling
C            package are made via this subroutine.
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
C***ROUTINES CALLED  XERMSG
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6MESG
C
C     GENERATE DIAGNOSTIC ERROR MESSAGES FOR THE DKLAG6 SOLVER.
C
      IMPLICIT NONE
C
      INTEGER IMESG
      CHARACTER*(*) SUBROU
C
C***FIRST EXECUTABLE STATEMENT  D6MESG
C
      IF (IMESG.LT.1) GO TO 1000
      IF (IMESG.GT.50) GO TO 1000
C
      CALL XERMSG('SLATEC',SUBROU,
     +'AN ERROR WAS ENCOUNTERED IN THE DKLAG6 SOLVER.
     +$$REFER TO THE FOLLOWING MESSAGE ... .',
     +IMESG,0)
C
      GO TO ( 10, 20, 30, 40, 50, 60, 70, 80, 90,100,110,120,
     +       130,140,150,160,170,180,190,200,210,220,230,240,
     +       250,260,270,280,290,300,310,320,330,340,350,360,
     +       370,380,390,400,410,420,430,440,450,460,470,480,
     +       490,500,510,520,530), IMESG
C
   10 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'INDEX IS LESS THAN 1 OR GREATER THAN 5. ',
     +IMESG,0)
      GO TO 1000
C
   20 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'NG IS NEGATIVE. ',
     +IMESG,0)
      GO TO 1000
C
   30 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'N IS LESS THAN ONE. ',
     +IMESG,0)
      GO TO 1000
C
   40 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'RELERR IS NEGATIVE. ',
     +IMESG,0)
      GO TO 1000
C
   50 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ABSERR IS NEGATIVE. ',
     +IMESG,0)
      GO TO 1000
C
   60 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'BOTH RELERR AND ABSERR ARE ZERO. ',
     +IMESG,0)
      GO TO 1000
C
   70 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'NSIG MUST BE AT LEAST ZERO. ',
     +IMESG,0)
      GO TO 1000
C
   80 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'TFINAL AND TOLD MUST BE DIFFERENT. ',
     +IMESG,0)
      GO TO 1000
C
   90 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE LENGTH OF THE WORK(W) ARRAY IS TOO SMALL. ',
     +IMESG,0)
      GO TO 1000
C
  100 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'A COMPONENT OF THE SOLUTION VANISHED MAKING
     +$$IT IMPOSSIBLE TO CONTINUE WITH A PURE
     +$$RELATIVE ERROR TEST. ',
     +IMESG,0)
      GO TO 1000
C
  110 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE REQUIRED STEPSIZE IS TOO SMALL. THIS
     +$$USUALLY INDICATES A SUDDEN CHANGE IN THE
     +$$EQUATIONS OR A VANISHING DELAY PROBLEM. ',
     +IMESG,0)
      GO TO 1000
C
  120 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE ROOTFINDER WAS NOT SUCCESSFUL. ',
     +IMESG,0)
      GO TO 1000
C
  130 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'A TERMINAL ERROR OCCURRED IN D6DRV3.
     +$$REFER TO THE OTHER PRINTED MESSAGES. ',
     +IMESG,0)
      GO TO 1000
C
  140 CONTINUE
C     INACTIVE ...
      CALL XERMSG('SLATEC',SUBROU,
     +'AN ERROR OCCURRED IN SUBROUTINE DSTURM. ',
     +IMESG,0)
      GO TO 1000
C
  150 CONTINUE
C     INACTIVE ...
      CALL XERMSG('SLATEC',SUBROU,
     +'AN ERROR OCCURRED IN SUBROUTINE ZRPOLY. ',
     +IMESG,0)
      GO TO 1000
C
  160 CONTINUE
C     INACTIVE ...
      CALL XERMSG('SLATEC',SUBROU,
     +'AN ERROR OCCURRED IN SUBROUTINE D6SORT. ',
     +IMESG,0)
      GO TO 1000
C
  170 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'AN EXTRAPOLATION TO THE LEFT BEYOND THE
     +$$SOLUTION HISTORY QUEUE WAS PERFORMED. ',
     +IMESG,0)
      GO TO 1000
C
  180 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'AN EXTRAPOLATION TO THE RIGHT BEYOND THE
     +$$SOLUTION HISTORY QUEUE WAS PERFORMED. ',
     +IMESG,0)
      GO TO 1000
C
  190 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'AN ABNORMAL ERROR WAS ENCOUNTERED IN D6DRV2.
     +$$REFER TO THE OTHER PRINTED MESSAGES. ',
     +IMESG,0)
      GO TO 1000
C
  200 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE LENGTH OF THE QUEUE ARRAY IS TOO SMALL. ',
     +IMESG,0)
      GO TO 1000
C
  210 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE LENGTH OF THE IWORK (IW) ARRAY IS TOO SMALL. ',
     +IMESG,0)
      GO TO 1000
C
  220 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE OUTPUT PRINT INCREMENT MUST BE NONZERO. ',
     +IMESG,0)
      GO TO 1000
C
  230 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'IROOT MUST BE SET TO 0 OR 1 IN INITAL. ',
     +IMESG,0)
      GO TO 1000
C
  240 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'A SYSTEM DELAY IS BEYOND THE CURRENT TIME. ',
     +IMESG,0)
      GO TO 1000
C
  250 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE INTEGRATION WAS TERMINATED BY THE USER. ',
     + IMESG,0)
      GO TO 1000
C
  260 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE INTEGRATION WAS TERMINATED TO AVOID
     +$$EXTRAPOLATING PAST THE QUEUE. ',
     +IMESG,0)
      GO TO 1000
C
  270 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE INTEGRATION WAS TERMINATED TO AVOID
     +$$EXTRAPOLATING BEYOND THE QUEUE. ',
     +IMESG,0)
      GO TO 1000
C
  280 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE SOLVER CANNOT START. IN D6INT1, THE DELAY
     +$$TIME DOES NOT FALL IN THE INITAL INTERVAL. ',
     +IMESG,0)
      GO TO 1000
C
  290 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE STEPSIZE WAS HALVED THIRTY TIMES IN D6STP1
     +$$DUE TO ILLEGAL DELAYS. THE INTEGRATION WILL
     +$$BE TERMINATED. ',
     +IMESG,0)
      GO TO 1000
C
  300 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'SUBROUTINE CHANGE CANNOT ALTER N. ',
     +IMESG,0)
      GO TO 1000
C
  310 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'SUBROUTINE CHANGE CANNOT ALTER NG. ',
     +IMESG,0)
      GO TO 1000
C
  320 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'A TERMINAL ERROR OCCURRED IN D6HST2. ',
     +IMESG,0)
      GO TO 1000
C
  330 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'A - B IS EQUAL TO ZERO IN D6HST1. ',
     +IMESG,0)
      GO TO 1000
C
  340 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'A SYSTEM DELAY TIME IS BEYOND THE
     +$$CURRENT TIME IN SUBROUTINE D6HST1. ',
     +IMESG,0)
      GO TO 1000
C
  350 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THERE IS NOT ENOUGH INFORMATION IN THE
     +$$SOLUTION HISTORY QUEUE TO INTERPOLATE FOR
     +$$THE VALUE OF T REQUESTED WHEN D6INT2 WAS
     +$$LAST CALLED. ',
     +IMESG,0)
      GO TO 1000
C
  360 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'YOU MUST DEFINE ITOL TO BE EITHER 1 OR N
     +$$IN SUBROUTINE INITIAL. ',
     +IMESG,0)
      GO TO 1000
C
  370 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'YOU HAVE CHANGED THE VALUE OF ITOL TO A VALUE
     +$$OTHER THAN 1 OR N IN YOUR CHANGE SUBROUTINE. ',
     +IMESG,0)
      GO TO 1000
C
  380 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'YOU HAVE SPECIFIED A NONZERO RELATIVE ERROR WHICH
     +$$IS SMALLER THAN THE FLOOR VALUE. IT WILL BE
     +$$REPLACED WITH THE FLOOR VALUE DEFINED IN
     +$$SUBROUTINE D6RERS. EXECUTION WILL CONTINUE. ',
     +IMESG,0)
      GO TO 1000
C
  390 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'YOU CHANGED THE STEPSIZE TO 0.0D0 WHEN YOUR SUBROUTINE
     +$$CHANGE WAS CALLED. THE INTEGRATION WILL BE TERMINATED. ',
     +IMESG,0)
      GO TO 1000
C
  400 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'THE STEPSIZE WAS HALVED THIRTY TIMES IN D6STP2
     +$$DUE TO CONVERGENCE FAILURES. THE INTEGRATION
     +$$WILL BE TERMINATED. ',
     +IMESG,0)
      GO TO 1000
C
  410 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'AFTER A ZERO OF AN EVENT FUNCTION WAS LOCATED AND
     +$$THE INTEGRATION STEP WAS REPEATED UP TO THE ZERO,
     +$$THE CORRECTOR ITERATION IN D6STP2 DID NOT CONVERGE. ',
     +IMESG,0)
      GO TO 1000
C
  420 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'AN ADVANCED DELAY WILL BE ALLOWED IN D6BETA. ',
     +IMESG,0)
      GO TO 1000
C
  430 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL LENGTH FOR HISTORY QUEUE IN D6DRV4. ',
     +IMESG,0)
      GO TO 1000
C
  440 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'AN ERROR WHEN D6DRV4 CALLED D6FIND. ',
     +IMESG,0)
      GO TO 1000
C
  450 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL MTFINS RETURNED FROM D6FIND. ',
     +IMESG,0)
      GO TO 1000
C
  460 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL MAXLEV IN D6FIND. ',
     +IMESG,0)
      GO TO 1000
C
  470 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL VALUE OF NUMLAGS IN D6FIND. ',
     +IMESG,0)
      GO TO 1000
C
  480 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL DELAY IN D6FIND. ',
     +IMESG,0)
      GO TO 1000
C
  490 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL VALUE OF MAXLEN IN D6FIND. ',
     +IMESG,0)
      GO TO 1000
C
  500 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'TINIT AND TFINAL MUST BE DIFFERENT IN D6DRV4. ',
     +IMESG,0)
      GO TO 1000
C
  510 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL NUMBER OF SORT POINTS IN ENCOUNTERED IN DKSORT. ',
     +IMESG,0)
      GO TO 1000
C
  520 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL VALUE OF KFLAG ENCOUNTERED IN DKSORT. ',
     +IMESG,0)
      GO TO 1000
C
  530 CONTINUE
      CALL XERMSG('SLATEC',SUBROU,
     +'ILLEGAL VALUE OF NEXTRA ENCOUNTERED IN DKSORT. ',
     +IMESG,0)
      GO TO 1000
C
 1000 CONTINUE
C
      RETURN
      END
C*DECK D6ERR1
      SUBROUTINE D6ERR1(N,ABSERR,RELERR,INDEX,RATIO,YNEW,K0,K2,K3,
     +                  K4,K5,K6,K7,K8,K9,R,ICVAL,ITOL,IPAIR)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6ERR1
C***SUBSIDIARY
C***PURPOSE  The purpose of D6ERR1 is to calculate error estimates
C            for the DKLAG6 differential equation solver. D6ERR1 is
C            called by the interval oriented driver D6DRV3.
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
C***ROUTINES CALLED  DAXPY, DCOPY, D6VEC2, D6POLY, D6VEC1
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6ERR1
C
      IMPLICIT NONE
C
C     CALCULATE THE MAXIMUM ERROR RATIO.
C
      INTEGER N,I,INDEX,ICVAL,ITOL,ICOMP,JCOMP,ITASK,IORDER,
     +        IPAIR,IFROM,IVAL,METH5
      DOUBLE PRECISION K0,K2,K3,K4,K5,K6,K7,K8,K9,R,YNEW,
     +                 ABSERR,RATIO,RELERR,TR,XDENOM,
     +                 XNUMER,C,FAC,CMAX,ALPHA,W9,TOLFAC
C
      DIMENSION YNEW(*),R(*),K0(*),K2(*),K3(*),K4(*),K5(*),K6(*),
     +          K7(*),K8(*),K9(*),ABSERR(*),RELERR(*),ICOMP(1)
C
C     NOTE:
C     THE K'S ARE NOT MULTIPLIED BY H.
C
C***FIRST EXECUTABLE STATEMENT  D6ERR1
C
      INDEX = 2
C
      GO TO (10,20,30,40,50,60,70,80,90,100), ICVAL
C
   10 CONTINUE
C
C     USE THE DIFFERENCE BETWEEN THE FIFTH AND FOURTH ORDER
C     METHODS TO ESTIMATE THE ERROR. THE ESTIMATE IS
C     (Y(6,8) - Y(5,10))(C) WITH C = 1.
C
      CALL DCOPY(N,K0,1,R,1)
      ALPHA = -184.0D0 / 3375.0D0
      CALL D6VEC2(N,R,ALPHA)
      ALPHA = 564.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K2,1,R,1)
      ALPHA = -60.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K3,1,R,1)
      ALPHA = -1720.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K4,1,R,1)
      ALPHA = -735.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K5,1,R,1)
      ALPHA = -1596.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K6,1,R,1)
      ALPHA = -644.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K7,1,R,1)
      ALPHA = 1000.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K8,1,R,1)
      ALPHA = 3375.0D0 / 3375.0D0
      CALL DAXPY(N,ALPHA,K9,1,R,1)
C     MULTIPLY THE CONTENTS OF R BY 
C     (Y(6,8) - Y(5,10)) (1) = -Y(5,10) (1)
      IF (IPAIR .EQ. 1) METH5 = 17
      IF (IPAIR .EQ. 2) METH5 = 25
      IF (IPAIR .EQ. 3) METH5 = 27
      IF (METH5 .EQ. 17) THEN
         W9 = 0.25D0
      ELSE
         IF (METH5 .EQ. 18) THEN
            W9 = -1.0D0
         ELSE
            IF (METH5 .EQ. 25) THEN
               W9 = 2.0D0
            ELSE
               IF (METH5 .EQ. 26) THEN
                  W9 = -2.0D0
               ELSE
C                 CAN MAKE ANOTHER SUBSTITUTION AT THIS POINT ...
                  W9 = 1.0D0
               ENDIF
            ENDIF
         ENDIF
      ENDIF
      CALL D6VEC2(N,R,W9)
      GO TO 200
C
   20 CONTINUE
C
C     USE THE DIFFERENCE BETWEEN THE SIXTH FIFTH AND FIFTH
C     ORDER METHODS TO ESTIMATE THE ERROR. THE ESTIMATE IS
C     (Y((6,10)) - Y((5,10))(C) WITH C = 1. IN PRINCIPLE,
C     THE FOLLOWING GIVES THE SAME RESULTS AS THE ABOVE
C     CALCULATIONS FOLLOWING STATEMENT NUMBER 10.
C
C     LOAD THE FIFTH ORDER INTERPOLANT IN R ...
C
      IORDER = 5
      ITASK = 1
      IFROM = 1
      C = 1.0D0
      FAC = 1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
C
C     SUBTRACT FROM IT THE SIXTH ORDER INTERPOLANT ...
C
      IORDER = 6
      ITASK = 4
      IFROM = 1
      C = 1.0D0
      FAC = -1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
      GO TO 200
C
   30 CONTINUE
C
C     USE THE DIFFERENCE BETWEEN THE SIXTH AND FIFTH ORDER
C     METHODS TO ESTIMATE THE ERROR. THE ESTIMATE IS
C     (Y((6,10)) - Y((5,10))(C) WITH C = 2.
C
C     LOAD THE FIFTH ORDER INTERPOLANT IN R ...
C
      IORDER = 5
      ITASK = 1
      IFROM = 1
      C = 2.0D0
      FAC = 1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
C
C     SUBTRACT FROM IT THE SIXTH ORDER INTERPOLANT ...
C
      IORDER = 6
      ITASK = 4
      IFROM = 1
      C = 2.0D0
      FAC = -1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
      GO TO 200
C
   40 CONTINUE
C
C     USE THE DIFFERENCE BETWEEN THE SIXTH AND FIFTH ORDER
C     METHODS TO ESTIMATE THE ERROR. THE ESTIMATE IS
C     (Y((6,10)) - Y((5,10))(C) WITH C GIVING THE MAXIMUM
C     DIFFERENCE ON [0,1]. (FOR THE METHODS USED, THIS
C     GIVES THE SAME ESTIMATE AS OPTION 2.)
C
C     LOAD THE FIFTH ORDER INTERPOLANT IN R ...
C
      CMAX = 1.0D0
      IORDER = 5
      ITASK = 1
      IFROM = 1
      C = CMAX
      FAC = 1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
C
C     SUBTRACT FROM IT THE SIXTH ORDER INTERPOLANT ...
C
      IORDER = 6
      ITASK = 4
      IFROM = 1
      C = CMAX
      FAC = -1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
      GO TO 200
C
   50 CONTINUE
C
C     USE THE INTEGRAL OVER [0,1] OF THE DIFFERENCE
C     |(Y((6,10)) - Y((5,10))'(C)| TO ESTIMATE THE
C     ERROR. FOR THE METHODS USED, THIS GIVES THE
C     SAME ESTIMATE AS OPTION 2.
C
      GO TO 20
C
   60 CONTINUE
C
C     USE THE INTEGRAL OVER [0,2] OF THE DIFFERENCE
C     |(Y((6,10)) - Y((5,10))'(C)| TO ESTIMATE THE
C     ERROR. FOR THE METHODS USED, THIS GIVES THE
C     SAME ESTIMATE AS OPTION 3.
C
      GO TO 30
C
   70 CONTINUE
C
C     USE THE DIFFERENCE BETWEEN THE DERIVATIVES OF THE
C     SIXTH FIFTH AND FIFTH ORDER METHODS TO ESTIMATE
C     THE ERROR. THE ESTIMATE IS (Y((6,10)) - Y((5,10))'(C)
C     WITH C = 1. 
C
C     LOAD THE DERIVATIVE OF THE FIFTH ORDER INTERPOLANT
C     IN R ...
C
      IORDER = 5
      ITASK = 3
      IFROM = 1
      C = 1.0D0
      FAC = 1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
C
C     SUBTRACT FROM IT THE DERIVATIVE OF THE SIXTH ORDER
C     INTERPOLANT ...
C
      IORDER = 6
      ITASK = 6
      IFROM = 1
      C = 1.0D0
      FAC = -1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
      GO TO 200
C
   80 CONTINUE
C
C     USE THE MAXIMUM DIFFERENCE BETWEEN THE SIXTH AND
C     FIFTH ORDER DERIVATIVES TO ESTIMATE THE ERROR.
C     THE ESTIMATE IS (Y((6,10)) - Y((5,10))'(C) WITH C
C     GIVING THE MAXIMUM DIFFERENCE.
C
C     LOAD THE DERIVATIVE OF THE FIFTH ORDER INTERPOLANT
C     IN R ...
C
      CMAX = 1.0D0
      IORDER = 5
      ITASK = 3
      IFROM = 1
      C = CMAX
      FAC = 1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
C
C     SUBTRACT FROM IT THE DERIVATIVE OF THE SIXTH ORDER
C     INTERPOLANT ...
C
      IORDER = 6
      ITASK = 6
      IFROM = 1
      C = CMAX
      FAC = -1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
      GO TO 200
C
   90 CONTINUE
C
C     USE THE DIFFERENCE BETWEEN THE SIXTH AND FIFTH ORDER
C     DERIVATIVES TO ESTIMATE THE ERROR. THE ESTIMATE IS
C     (Y((6,10)) - Y((5,10))'(C) WITH C = 2.
C
C     LOAD THE DERIVATIVE OF THE FIFTH ORDER INTERPOLANT
C     IN R ...
C
      IORDER = 5
      ITASK = 3
      IFROM = 1
      C = 2.0D0
      FAC = 1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
C
C     SUBTRACT FROM IT THE DERIVATIVE OF THE SIXTH ORDER
C     INTERPOLANT ...
C
      IORDER = 6
      ITASK = 6
      IFROM = 1
      C = 2.0D0
      FAC = -1.0D0
      IVAL = 0
      CALL D6POLY(N,0.0D0,R,0.0D0,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            R,R,0.0D0,IORDER,ICOMP,IPAIR,ITASK,IFROM,
     +            C,FAC,IVAL,13)
      GO TO 200
C
  100 CONTINUE
C
C     USE THE INTEGRAL OVER [0,1] OF THE DIFFERENCE
C     |P(C)| = |(Y((6,10)) - Y((5,10))'(C)| TO
C     ESTIMATE THE ERROR. FOR THE METHODS USED, THIS
C     GIVES THE SAME ESTIMATE AS OPTION 2.
C
      GO TO 20
C
  200 CONTINUE
C
C     CALCULATE THE ERROR RATIO (MAXIMUM ESTIMATED
C     ERROR PER STEP DIVIDED BY THE WEIGHT).
C
      RATIO = 0.0D0
      DO 210 I = 1,N
          JCOMP = I
          IF (I.GT.ITOL) JCOMP = 1
          XNUMER = ABS(R(I))
          XDENOM = RELERR(JCOMP)*ABS(YNEW(I)) + ABSERR(JCOMP)
          TR = XNUMER/XDENOM
          RATIO = MAX(RATIO,TR)
  210 CONTINUE
C
C     THE FOLLOWING VALUE OF TOLFAC WILL DECREASE THE NEXT
C     ESTIMATED STEPSIZE BY A FACTOR OF (1/15)**(1/6) AND
C     THEREFORE MAY INCREASE THE NUMBER OF STEPS BY ABOUT 60%.
C
      TOLFAC = 15.0D0
C
      RATIO = TOLFAC * RATIO
C
      RETURN
      END
C*DECK D6POLY
      SUBROUTINE D6POLY (N, TOLD, YOLD, H, K0, K2, K3, K4, K5, K6,
     +                   K7, K8, K9, YVAL, DYVAL, TVAL, IORDER,
     +                   ICOMP, IPAIR, ITASK, IFROM, CVAL, FAC,
     +                   IONLY, ISPOT)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***AUTHOR  THOMPSON, S., RADFORD UNIVERSITY, RADFORD, VIRGINIA.
C***BEGIN PROLOGUE  D6POLY
C***SUBSIDIARY
C***PURPOSE  The purpose of D6POLY is to evaluate the solution
C            polynomial and/or the derivative polynomial for
C            the DKLAG6 differential equation solver. D6POLY
C            is the core (lowest level) interpolation subroutine
C            for DKLAG6. This subroutine is where the action is.
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
C***ROUTINES CALLED  D6WSET
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6POLY
C
C     CORE INTERPOLATION SUBROUTINE FOR THE SOLUTION
C     AND/OR FOR THE DERIVATIVE
C
C     PARAMETERS:
C
C     IF IFROM = 0 ...
C
C     N        = NUMBER OF EQUATIONS
C
C     TOLD     = OLD INTEGRATION TIME
C
C     YOLD(*)  = Y(T0)
C
C     H        = STEPSIZE
C
C     K0, K2,  = RKS DERIVATIVE STAGE APPROXIMATIONS
C     K3, K4,
C     K5, K6,
C     K7, K8,
C     K9
C
C     YVAL(*)  = ARRAY OF LENGTH N. IF REQUESTED, THE
C                INTERPOLATED SOLUTION WILL BE RETURNED
C                IN THIS ARRAY.
C                NOT USED IF ITASK = 2,5,-2,-5.
C
C     DYVAL(*) = ARRAY OF LENGTH N. IF REQUESTED, THE
C                INTERPOLATED DERIVATIVE WILL BE RETURNED
C                IN THIS ARRAY.
C                NOT USED IF ITASK = 1,4,-1,-4
C
C     TVAL     = TIME AT WHICH THE SOLUTION AND/OR
C                DERIVATIVE IS TO BE APPROXIMATED
C
C     IORDER   = ORDER OF METHOD TO BE USED (5 OR 6)
C
C     ICOMP(*) = ARRAY OF LENGTH N. IF ICOMP(I) IS NOT 0,
C                INTERPOLATIONS WILL NOT BE FORMED FOR
C                COMPONENT I. NOT USED UNLESS ITASK IS
C                NEGATIVE.
C
C     IONLY    = IF ITASK > 0 AND IONLY .NE. 0, THE
C                CALCULATIONS WILL BE PERFORMED ONLY
C                FOR COMPONENT IONLY.
C
C     IPAIR    = WHICH PAIR OF METHODS TO USE (1 OR 2)
C
C     ITASK    = TASK FLAG, WITH THE FOLLOWING MEANINGS:
C
C              =  1 RETURN THE INTERPOLATED SOLUTION IN
C                   YVAL
C
C              =  2 RETURN THE INTERPOLATED DERIVATIVE
C                   IN DYVAL.
C
C              =  3 RETURN THE INTERPOLATED SOLUTION IN
C                   YVAL AND RETURN THE INTERPOLATED
C                   DERIVATIVE IN YVAL.
C
C              =  4 ADD THE INPUT VALUES OF YVAL TO THE
C                   INTERPOLATED SOLUTION AND RETURN
C                   THE RESULTS IN YVAL.
C
C              =  5 ADD THE INPUT VALUES OF DYVAL
C                   TO THE INTERPOLATED DERIVATIVE
C                   AND RETURN THE RESULTS IN DYVAL.
C
C              =  6 ADD THE INPUT VALUES OF YVAL TO THE
C                   INTERPOLATED SOLUTION AND ADD THE
C                   INPUT VALUES OF DYVAL TO THE
C                   INTERPOLATED DERIVATIVE; AND RETURN
C                   THE RESULTS IN YVAL AND DYVAL,
C                   RESPECTIVELY.
C
C              = -1 SAME AS 1 FOR SELECTED COMPONENTS,
C                   DETERMINED BY ICOMP(*).
C
C              = -2 SAME AS 2 FOR SELECTED COMPONENTS,
C                   DETERMINED BY ICOMP(*).
C
C              = -3 SAME AS 3 FOR SELECTED COMPONENTS,
C                   DETERMINED BY ICOMP(*).
C
C              = -4 SAME AS 4 FOR SELECTED COMPONENTS,
C                   DETERMINED BY ICOMP(*).
C
C              = -5 SAME AS 5 FOR SELECTED COMPONENTS,
C                   DETERMINED BY ICOMP(*).
C
C              = -6 SAME AS 6 FOR SELECTED COMPONENTS,
C                   DETERMINED BY ICOMP(*).
C
C     CVAL AND FAC ARE NOT USED IF IFROM = 0.
C
C     ISPOT = FLAG TO INDICATE THE SPOT FROM WHICH D6POLY
C             IS CALLED.
C
C     IF IFROM = 1 ...
C
C     THIS CALL WAS MADE BY THE ERROR ESTIMATION SUBROUTINE
C     D6ERR1. IN THIS CASE, THE VALUE OF C TO BE
C     USED IS CVAL. THE CONTENTS OF YVAL(*) AND/OR DYVAL(*)
C     WILL BE MULTIPLIED BY FAC BEFORE ADDING THE NEW RESULTS.
C     IN THIS CASE YOLD(*), H, TVAL, AND TOLD ARE NOT USED.
C
      IMPLICIT NONE
C
      INTEGER N, ICOMP, IORDER, ITASK, JTASK, METHOD, J, IFROM,
     +        IONLY, ISPOT, IABS, IPAIR
      DOUBLE PRECISION C, W, WD, TOLD, YOLD, H, K0, K2, K3, K4,
     +                 K5, K6, K7, K8, K9, TVAL, YVAL, DYVAL,
     +                 SUM, CVAL, FAC
      LOGICAL POLYS, POLYD
      DIMENSION W(0:9), WD(0:9), YOLD(*), K0(*), K2(*), K3(*),
     +          K4(*), K5(*), K6(*), K7(*), K8(*), K9(*),
     +          ICOMP(*), YVAL(*), DYVAL(*)
C
C***FIRST EXECUTABLE STATEMENT  D6POLY
C
C     DETERMINE THE VALUE OF C AT WHICH THE SOLUTION AND/OR
C     DERIVATIVE ARE TO BE INTERPOLATED ...
C
      IF (IFROM .EQ. 0) THEN
         C = (TVAL - TOLD) / H
      ELSE
         C = CVAL
      ENDIF
C
C     ITASK MAY BE 1, 2, 3, 4, 5, 6, -1, -2, -3, 4, 5, 6,
C     DEPENDING ON WHICH SUBROUTINE MADE THIS CALL;
C     DETERMINE WHAT IS TO BE DONE ...
C
      JTASK = IABS(ITASK)
C
C     DETERMINE IF THE SOLUTION IS TO BE INTERPOLATED ...
C
      IF (JTASK .EQ. 2 .OR. JTASK .EQ. 5) THEN
         POLYS = .FALSE.
      ELSE
         POLYS = .TRUE.
      ENDIF
C
C     DETERMINE IF THE DERIVATIVE IS TO BE INTERPOLATED ...
C
      IF (JTASK .EQ. 1 .OR. JTASK .EQ. 4) THEN
         POLYD = .FALSE.
      ELSE
         POLYD = .TRUE.
      ENDIF
C
C     DETERMINE THE METHOD TO BE USED ...
C
      IF (IPAIR .EQ. 1) THEN
         IF (IORDER .EQ. 6) METHOD = 4
         IF (IORDER .EQ. 5) METHOD = 17
      ENDIF
      IF (IPAIR .EQ. 2) THEN
         IF (IORDER .EQ. 6) METHOD = 2
         IF (IORDER .EQ. 5) METHOD = 25
      ENDIF
      IF (IPAIR .EQ. 3) THEN
         IF (IORDER .EQ. 6) METHOD = 4
         IF (IORDER .EQ. 5) METHOD = 27
      ENDIF
C
C     CALCULATE THE INTERPOLATION COEFFICIENTS AND/OR
C     THEIR DERIVATIVES ...
C
      CALL D6WSET (C, W, WD, POLYS, POLYD, IORDER, METHOD)
C
C     IF THIS CALL TO D6POLY WAS NOT MADE FOR PURPOSES OF
C     ERROR ESTIMATION BY D6ERR1, DO THE NECESSARY
C     INTERPOLATIONS ...
C
      IF (IFROM .EQ. 0) THEN
C
C        EVALUATE THE INTERPOLATION POLYNOMIAL FOR ALL METHODS
C        USING EQUATION 3A ...
C
         IF (POLYS) THEN
C
            DO 20 J = 1 , N
C
               IF (ITASK .LT. 0) THEN
                  IF (ICOMP(J) .NE. 0) GO TO 20
               ENDIF
C
               IF (ITASK .GT. 0) THEN
                  IF (IONLY .GT. 0 .AND. IONLY .NE. J) THEN
                     GO TO 20
                  ENDIF
               ENDIF
C
               SUM = W(0) * K0(J) + W(2) * K2(J) + W(3) * K3(J)
     +             + W(4) * K4(J) + W(5) * K5(J) + W(6) * K6(J)
     +             + W(7) * K7(J) + W(8) * K8(J) + W(9) * K9(J)
C
               IF (JTASK .EQ. 1 .OR. JTASK .EQ. 3)
     +            YVAL(J) = YOLD(J) + H * SUM
C
               IF (JTASK .EQ. 4 .OR. JTASK .EQ. 6)
     +            YVAL(J) = (YOLD(J) + H * SUM) + YVAL(J)
C
   20       CONTINUE
C
         ENDIF
C
C        EVALUATE THE DERIVATIVE OF THE INTERPOLATION POLYNOMIAL
C        FOR ALL METHODS USING THE DERIVATIVE OF EQUATION 3A ...
C
         IF (POLYD) THEN
C
            DO 40 J = 1 , N
C
               IF (ITASK .LT. 0) THEN
                  IF (ICOMP(J) .NE. 0) GO TO 40
               ENDIF
C
               IF (ITASK .GT. 0) THEN
                  IF (IONLY .GT. 0 .AND. IONLY .NE. J) THEN
                     GO TO 40
                  ENDIF
               ENDIF
C
               SUM = WD(0) * K0(J) + WD(2) * K2(J) + WD(3) * K3(J)
     +             + WD(4) * K4(J) + WD(5) * K5(J) + WD(6) * K6(J)
     +             + WD(7) * K7(J) + WD(8) * K8(J) + WD(9) * K9(J)
C
               IF (JTASK .EQ. 2 .OR. JTASK .EQ. 3)
     +            DYVAL(J) = SUM
C
               IF (JTASK .EQ. 5 .OR. JTASK .EQ. 6)
     +            DYVAL(J) = SUM + DYVAL(J)
C
   40       CONTINUE
C
         ENDIF
C
      ENDIF
C
C     IF THIS CALL TO D6POLY WAS MADE FOR PURPOSES OF
C     ERROR ESTIMATION BY D6ERR1 FORM THE NECESSARY
C     ESTIMATES ...
C
      IF (IFROM .NE. 0) THEN
C
C        EVALUATE THE INTERPOLATION POLYNOMIAL FOR ALL METHODS
C        USING EQUATION 3A BUT LEAVE OFF YOLD(*) AND DO NOT
C        MULTIPLY BY H ...
C
         IF (POLYS) THEN
C
            DO 60 J = 1 , N
C
               IF (ITASK .LT. 0) THEN
                  IF (ICOMP(J) .NE. 0) GO TO 60
               ENDIF
C
               IF (ITASK .GT. 0) THEN
                  IF (IONLY .GT. 0 .AND. IONLY .NE. J) THEN
                     GO TO 60
                  ENDIF
               ENDIF
C
               SUM = W(0) * K0(J) + W(2) * K2(J) + W(3) * K3(J)
     +             + W(4) * K4(J) + W(5) * K5(J) + W(6) * K6(J)
     +             + W(7) * K7(J) + W(8) * K8(J) + W(9) * K9(J)
C
               IF (JTASK .EQ. 1 .OR. JTASK .EQ. 3)
     +            YVAL(J) = SUM
C
               IF (JTASK .EQ. 4 .OR. JTASK .EQ. 6)
     +            YVAL(J) = SUM + FAC * YVAL(J)
C
   60       CONTINUE
C
         ENDIF
C
C        EVALUATE THE DERIVATIVE OF THE INTERPOLATION POLYNOMIAL
C        FOR ALL METHODS USING THE DERIVATIVE OF EQUATION 3A ...
C
         IF (POLYD) THEN
C
            DO 80 J = 1 , N
C
               IF (ITASK .LT. 0) THEN
                  IF (ICOMP(J) .NE. 0) GO TO 80
               ENDIF
C
               IF (ITASK .GT. 0) THEN
                  IF (IONLY .GT. 0 .AND. IONLY .NE. J) THEN
                     GO TO 80
                  ENDIF
               ENDIF
C
               SUM = WD(0) * K0(J) + WD(2) * K2(J) + WD(3) * K3(J)
     +             + WD(4) * K4(J) + WD(5) * K5(J) + WD(6) * K6(J)
     +             + WD(7) * K7(J) + WD(8) * K8(J) + WD(9) * K9(J)
C
               IF (JTASK .EQ. 2 .OR. JTASK .EQ. 3)
     +            DYVAL(J) = SUM
C
               IF (JTASK .EQ. 5 .OR. JTASK .EQ. 6)
     +            DYVAL(J) = SUM + FAC * DYVAL(J)
C
   80       CONTINUE
C
         ENDIF
C
      ENDIF
C
      RETURN
      END
C*DECK D6WSET
      SUBROUTINE D6WSET (C, W, WD, POLYS, POLYD, IORDER, METHOD)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***AUTHOR  THOMPSON, S., RADFORD UNIVERSITY, RADFORD, VIRGINIA.
C***BEGIN PROLOGUE  D6WSET
C***SUBSIDIARY
C***PURPOSE  The purpose of D6WSET is to calculate the interpolation
C            coefficients and/or their derivatives for the DKLAG6
C            differential equation solver. It is called by the core
C            (lowest level) interpolation subroutine D6POLY.
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
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6WSET
C
C     PARAMETERS:
C
C     C      = VALUE FOR WHICH THE COEFFICIENTS WILL BE FORMED
C
C     W(*)   = INTERPOLATION COEFFICIENTS. DIMENSION W(0:9)
C
C     WD(*)  = DERIVATIVE COEFFICIENTS. DIMENSION WD(0:9)
C
C     POLYS  = LOGICAL VARIABLE. IF TRUE, THE INTERPOLATION
C              COEFFICIENTS WILL BE FORMED
C
C     POLYD  = LOGICAL VARIABLE. IF TRUE, THE DERIVATIVE
C              COEFFICIENTS WILL BE FORMED
C
C     IORDER = ORDER OF METHOD
C
C     METHOD = WHICH METHOD IS TO BE USED
C
      IMPLICIT NONE
C
      INTEGER IORDER, METHOD
      DOUBLE PRECISION C, W, WD
      LOGICAL POLYS, POLYD
      DIMENSION W(0:9), WD(0:9)
C
C***FIRST EXECUTABLE STATEMENT  D6WSET
C
C     SOLUTION ...
C
      IF (POLYS) THEN
C
C        W_9 ...
C
C        FOR THE PRIMARY INTEGRATION METHODS ...
C
         IF (IORDER .EQ. 6) THEN
            IF (METHOD .EQ. 4) THEN
               W(9) = 0.0D0
            ELSE
               IF (METHOD .EQ. 5) THEN
                  W(9) = 0.75D0 * (
     +                    C *
     +                    C * (  22.0D0
     +                  + C * (-137.0D0
     +                  + C * ( 298.0D0
     +                  + C * (-273.0D0
     +                  + C * (  90.0D0 ))))) )
               ELSE
                  IF (METHOD .EQ. 6) THEN
                     W(9) = (27.0D0/4.0D0) * (
     +                     C *
     +                     C * (-2.0D0
     +                   + C * ( 7.0D0
     +                   + C * (-8.0D0
     +                   + C * ( 3.0D0 )))) )
                  ELSE
                     IF (METHOD .EQ. 7) THEN
                        W(9) = (9.0D0/4.0D0) * (
     +                     C *
     +                     C * ( -4.0D0
     +                   + C * ( 11.0D0
     +                   + C * (-10.0D0
     +                   + C * (  3.0D0 )))) )
                     ELSE
C                       CAN MAKE SUBSTITUTION FOR ANOTHER
C                       METHOD AT THIS POINT ...
                        W(9) = 0.0D0
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDIF
C
C        FOR THE SUBSIDIARY INTEGRATION METHODS ...
C
         IF (IORDER .EQ. 5) THEN
            IF (METHOD .EQ. 17) THEN
               W(9) = 0.25D0 * C**5
            ELSE
               IF (METHOD .EQ. 18) THEN
                  W(9) = -(C**5)
               ELSE
                  IF (METHOD .EQ. 25) THEN
                     W(9) = 2.0D0 * (C**5)
                  ELSE
                     IF (METHOD .EQ. 26) THEN
                        W(9) = -2.0D0 * (C**5)
                     ELSE
C                       CAN MAKE ANOTHER SUBSTITUTION AT THIS POINT ...
                        W(9) = C**5
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDIF
C
C        DEFINE THE REMAINING INTERPOLATION COEFFICIENTS ...
C
C        W_0 ...
C
         W(0) =
     +           ( C * (  1.0D0
     +           + C * ( -2427.0D0/500.0D0
     +           + C * (  3586.0D0/375.0D0
     +           + C * ( -1659.0D0/200.0D0
     +           + C * (    66.0D0/ 25.0D0 ))))) )
     +           - (184.0D0/3375.0D0) * W(9)
C
C        W_1 ...
C
         W(1) = 0.0D0
C
C        W_2 ...
C
         W(2) = (  C *
     +             C * (   708.0D0/125.0D0
     +           + C * ( -1977.0D0/125.0D0
     +           + C * (   789.0D0/ 50.0D0
     +           + C * (   -27.0D0/  5.0D0 )))) )
     +           + (188.0D0/1125.0D0) * W(9)
C
C        W_3 ...
C
         W(3) = (  C *
     +             C * (   87.0D0/50.0D0
     +           + C * ( -243.0D0/50.0D0
     +           + C * (  201.0D0/40.0D0
     +           + C * (   -9.0D0/ 5.0D0 )))) )
     +           - (4.0D0/225.0D0) * W(9)
C
C        W_4 ...
C
         W(4) = (  C * 
     +             C * (  -88.0D0/25.0D0
     +           + C * ( 1226.0D0/75.0D0
     +           + C * (  -21.0D0       
     +           + C * (   42.0D0/ 5.0D0 )))) )
     +           - (344.0D0/675.0D0) * W(9)
C
C        W_5 ...
C
         W(5) = (  C *
     +             C * ( -21.0D0/100.0D0
     +           + C * (  21.0D0/ 25.0D0
     +           + C * ( -21.0D0/ 40.0D0 ))) )
     +           - (49.0D0/225.0D0) * W(9)
C
C        W_6 ...
C
         W(6) = (  C *
     +             C * (   228.0D0/125.0D0
     +           + C * ( -1197.0D0/125.0D0
     +           + C * (   741.0D0/ 50.0D0
     +           + C * (  -171.0D0/ 25.0D0 )))) )
     +           - (532.0D0/1125.0D0) * W(9)
C
C        W_7 ...
C
         W(7) = (  C *
     +             C * ( -161.0D0/250.0D0
     +           + C * ( 1127.0D0/750.0D0
     +           + C * ( -161.0D0/200.0D0 ))) )
     +           - (644.0D0/3375.0D0) * W(9)
C
C        W_8 ...
C
         W(8) = (  C * C *
     +             C * (  2.0D0
     +           + C * ( -5.0D0
     +           + C * (  3.0D0 ))) )
     +           + (8.0D0/27.0D0) * W(9)
C
      ENDIF
C
C     DERIVATIVE ...
C
      IF (POLYD) THEN
C
C        WD_9 ...
C
C        FOR THE PRIMARY INTEGRATION METHODS ...
C
         IF (IORDER .EQ. 6) THEN
            IF (METHOD .EQ. 4) THEN
               WD(9) = 0.0D0
            ELSE
               IF (METHOD .EQ. 5) THEN
                  WD(9) = 0.75D0 * (
     +                    C * (   44.0D0
     +                  + C * ( -411.0D0
     +                  + C * ( 1192.0D0
     +                  + C * (-1365.0D0
     +                  + C * (  540.0D0 ))))) )
               ELSE
                  IF (METHOD .EQ. 6) THEN
                     WD(9) = (27.0D0/4.0D0) * (
     +                       C * ( -4.0D0
     +                     + C * ( 21.0D0
     +                     + C * (-32.0D0
     +                     + C * (15.0D0 )))) )
                   ELSE
                      IF (METHOD .EQ. 7) THEN
                         WD(9) = (9.0D0/4.0D0) * (
     +                       C * ( -8.0D0
     +                     + C * ( 33.0D0
     +                     + C * (-40.0D0
     +                     + C * ( 15.0D0 )))) )
                      ELSE
C                        CAN MAKE SUBSTITUTION FOR ANOTHER
C                        METHOD AT THIS POINT ...
                         WD(9) = 0.0D0
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDIF
C
C        FOR THE SUBSIDIARY INTEGRATION METHODS ...
C
         IF (IORDER .EQ. 5) THEN
            IF (METHOD .EQ. 17) THEN
               WD(9) = 1.25D0 * (C**4)
            ELSE
               IF (METHOD .EQ. 18) THEN
                  WD(9) = -5.0D0 * (C**4)
               ELSE
                  IF (METHOD .EQ. 25) THEN
                     WD(9) = 10.0D0 * (C**4)
                  ELSE
                     IF (METHOD .EQ. 26) THEN
                        WD(9) = -10.0D0 * (C**4)
                     ELSE
C                       CAN MAKE ANOTHER SUBSTITUTION AT THIS POINT ...
                        WD(9) = 5.0D0 * (C**4)
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDIF
C
C        DEFINE THE REMAINING DERIVATIVE COEFFICIENTS ...
C
C        WD_0 ...
C
         WD(0) = 
     +           ( 1.0D0
     +           + C * ( -2427.0D0/250.0D0
     +           + C * (  3586.0D0/125.0D0
     +           + C * ( -1659.0D0/ 50.0D0
     +           + C * (    66.0D0/  5.0D0 )))) )
     +           - (184.0D0/3375.0D0) * WD(9)
C
C        WD_1 ...
C
         WD(1) = 0.0D0
C
C        WD_2 ...
C
         WD(2) =
     +           ( C * (  1416.0D0/125.0D0
     +           + C * ( -5931.0D0/125.0D0
     +           + C * (  1578.0D0/ 25.0D0
     +           + C * (   -27.0D0         )))) )
     +           + (188.0D0/1125.0D0) * WD(9)
C
C        WD_3 ...
C
         WD(3) = ( C * (   87.0D0/25.0D0
     +           + C * ( -729.0D0/50.0D0
     +           + C * (  201.0D0/10.0D0
     +           + C * (   -9.0D0        )))) )
     +           - (4.0D0/225.0D0) * WD(9)
C
C        WD_4 ...
C
         WD(4) = ( C * ( -176.0D0/25.0D0
     +           + C * ( 1226.0D0/25.0D0
     +           + C * (  -84.0D0       
     +           + C * (   42.0D0 )))) )
     +           - (344.0D0/675.0D0) * WD(9)
C
C        WD_5 ...
C
         WD(5) = ( C * ( -21.0D0/50.0D0
     +           + C * (  63.0D0/25.0D0
     +           + C * ( -21.0D0/10.0D0 ))) )
     +           - (49.0D0/225.0D0) * WD(9)
C
C        WD_6 ...
C
         WD(6) = ( C * (   456.0D0/125.0D0
     +           + C * ( -3591.0D0/125.0D0
     +           + C * (  1482.0D0/ 25.0D0
     +           + C * (  -171.0D0/  5.0D0 )))) )
     +           - (532.0D0/1125.0D0) * WD(9)
C
C        WD_7 ...
C
         WD(7) = ( C * ( -161.0D0/125.0D0
     +           + C * ( 1127.0D0/250.0D0
     +           + C * ( -161.0D0/ 50.0D0 ))) )
     +           - (644.0D0/3375.0D0) * WD(9)
C
C        WD_8 ...
C
         WD(8) = ( C *
     +             C * (   6.0D0
     +           + C * ( -20.0D0
     +           + C * (  15.0D0 ))) )
     +           + (8.0D0/27.0D0) * WD(9)
C
      ENDIF
C
      RETURN
      END
C
C _____________________________________________________________
C
C THIS IS THE BEGINNING OF THE NEW SUBROUTINES. NOTE: IT WAS
C NECESSARY TO MODIFY SOME OF THE OTHER INTERNAL SUBROUTINES
C TO ACCOMODATE THE EXTRA PARAMETERS TROOT2 AND IRCALL AND
C TO HANDLE NEW ERROR MESSAGES.
C _____________________________________________________________
C
C*DECK DKLG6C
      SUBROUTINE DKLG6C(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,
     +                  YINIT,DYINIT,N,W,LW,IW,LIW,QUEUE,LQUEUE,
     +                  IFLAG,LIN,LOUT,RPAR,IPAR)
C
C***BEGIN PROLOGUE  DKLG6C
C***PURPOSE  Use this driver rather than DKLAG6 if all
C            delays are known to be constant.
C
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
C     Refer to the documentation prologue for DKLAG6 for
C     complete descriptions of the different parameters.
C     Usage of DKLG6C is the same as that for DKLAG6.
C
C     ********UNTESTED EXPERIMENTAL VERSION OF DKLAG6********
C
C     If all delays are constant (and tfinal>tinit, for now),
C     this version does not perform automatic rootfinding for
C     the tree of potential discontinuity times. Instead
C     it builds the tree using subroutine d6find and forces
C     the integrator to step exactly to each point in the
C     tree. It uses a modified version d6drv4 of d6drv2
C     that has not been tested thoroughly. The purpose of
C     doing this is to avoid needless error test failures
C     in the preliminary steps across a discontinuity and
C     to make the author feel much better about the
C     validity of the local error test near such points.
C     This also avoids the need to reduce the step size to
C     step exactly to extrapolatory roots normally found
C     in the rootfinding phase. No changes have been made
C     in any subroutine that is subsidiary to d6drv4. If
C     you turn up any bugs please contact me and I will
C     fix them.
C
      IMPLICIT NONE
C
      INTEGER LIW,LW,LQUEUE,IFLAG,LIN,LOUT,IW,J4SAVE,IPAR,
     +        INEED,J1,J2,J3,J4,J5,J6,J7,J8,J9,J10,J11,J12,
     +        J13,J14,K1,K2,K3,K4,LWORK,LIWORK,N,
     +        IDUM,NTFINS,LQ
      DOUBLE PRECISION W,QUEUE,D6OUTP,RPAR,RDUM
C
      DIMENSION IW(LIW),IPAR(*)
      DIMENSION W(LW),QUEUE(LQUEUE),RPAR(*)
C
C     DECLARE THE OUTPUT INCREMENT FUNCTION.
C
      EXTERNAL D6OUTP
C
C     DECLARE THE USER SUPPLIED PROBLEM DEPENDENT SUBROUTINES.
C
      EXTERNAL INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,DYINIT
C
C     LWORK IS THE LENGTH OF THE WORK ARRAY (FOR D6DRV4; NEED
C     13*N MORE IF GO THROUGH D6KLAG).
C     LWORK MUST BE AT LEAST 21*N + 1.
C
C     LQUEUE IS THE LENGTH OF THE QUEUE HISTORY ARRAY.
C     LQUEUE SHOULD BE APPROXIMATELY
C         (12*(N+1)) * (TFINAL-TINIT)/HAVG ,
C     WHERE TINIT AND TFINAL ARE THE BEGINNING AND FINAL
C     INTEGRATION TIMES AND HAVG IS THE AVERAGE STEPSIZE
C     THAT WILL BE USED BY THE INTEGRATOR.
C
C***FIRST EXECUTABLE STATEMENT  DKLG6C
C
C     RESET THE OUTPUT UNIT NUMBER TO THE ONE SPECIFIED
C     BY THE USER.
C
      IDUM = J4SAVE(3,LOUT,.TRUE.)
C
c     temporary:
      WRITE (LOUT,100)
  100 FORMAT(' Warning: Do not use this version unless all delays',/,
     +       ' are constant. Please be aware that this is an',     /,
     +       ' experimental version that has not been tested',     /,
     +       ' thoroughly. If you turn up any bugs, please',       /,
     +       ' contact me and I will fix them')
C
      CALL D6STAT(N, 1, 0.0D0, RPAR, IPAR, IDUM, RDUM)
C
      J1 = 1
      J2 = J1 + N
      J3 = J2 + N
      J4 = J3 + N
      J5 = J4 + N
      J6 = J5 + N
      J7 = J6 + N
      J8 = J7 + N
      J9 = J8 + N
      J10 = J9 + N
      J11 = J10 + N
      J12 = J11 + N
      J13 = J12 + N
      J14 = J13 + N
      LWORK = LW - (J14-1)
C
C     TELL D6DRV4 THAT THE HISTORY QUEUE AND THE TFINS ARRAY
C     ARE THE SAME SIZE AT THIS POINT AND PASS THE QUEUE
C     ARRAY FOR EACH.
      NTFINS = LQUEUE
      LQ = LQUEUE
C
      K1 = 1
      K2 = K1 + 34 
      LIWORK = LIW - 34 
C     THE REMAINING INTEGER STORAGE WILL BE SPLIT EVENLY
C     BETWEEN THE JROOT AND INDEXG ARRAYS.
      LIWORK = LIWORK/3
      K3 = K2 + LIWORK
      K4 = K3 + LIWORK
C
C     PRELIMINARY CHECK FOR ILLEGAL INPUT ERRORS.
C
      IF (N.LT.1) THEN
          CALL D6MESG(3,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      INEED = 12*(N+1)
      IF (INEED.GT.LQ) THEN
          CALL D6MESG(20,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LIWORK.LT.0 .OR. LIW .LT. 35) THEN
          CALL D6MESG(21,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LWORK.LT.21*N+1) THEN
          CALL D6MESG(9,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      CALL D6DRV4(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,
     +            DYINIT,N,W(J1),W(J2),W(J3),W(J4),W(J5),W(J6),
     +            W(J7),W(J8),W(J9),W(J10),W(J11),W(J12),W(J13),
     +            W(J14),QUEUE,IW(K1),IW(K2),IW(K3),IW(K4),LWORK,
     +            LIWORK,LQ,IFLAG,D6OUTP,LIN,LOUT,RPAR,IPAR,
     +            QUEUE,NTFINS)
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6DRV4
      SUBROUTINE D6DRV4(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,
     +                  YINIT,DYINIT,N,YOLD,DYOLD,YNEW,DYNEW,
     +                  YROOT,DYROOT,YOUT,DYOUT,YLAG,DYLAG,BVAL,
     +                  ABSERR,RELERR,WORK,QUEUE,IPOINT,JROOT,
     +                  INDEXG,LEVEL,LWORK,LIWORK,LQUSER,IFLAG,
     +                  D6OUTP,LIN,LOUT,RPAR,IPAR,TFINS,NTFINS)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6DRV4
C
C***PURPOSE  This second level driver is called by DKLG6C.
C
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
C     ********UNTESTED EXPERIMENTAL VERSION OF D6DRV2********
C
C     Refer to the documentation prologue for D6DRV2 for
C     complete descriptions of the different parameters.
C     Usage of D6DRV4 is the same as that for D6DRV2.
c
      IMPLICIT NONE
C
      INTEGER IFLAG,IHFLAG,LIWORK,LWORK,LIN,LOUT,LQUEUE,MORDER,N,
     +        INDEXG,IPOINT,JROOT,IPAR,IDCOMP,JCOMP,IDISC,IWANT,J,
     +        IDIR,IER,IERRUS,IFIRST,IFOUND,INX1,INDEX,INEED,IOK,
     +        IORDER,IQUEUE,ITYPE,I,I1,I2,I3,I4,I5,I6,I7,I8,I9,
     +        I10,I11,I12,I13,I14,I15,I9M1,I10M1,I11M1,JNDEX,JQUEUE,
     +        KQUEUE,KROOT,NFEVAL,NG,NGEVAL,NSIG,NGUSER,NGTEMP,
     +        NTEMP,ITOL,ICOMP,NGEMAX,NGOLD,NDISC,IG,IDUM,IK7,IK8,
     +        IK9,IPAIR,OFFRT,RSTRT,AUTORF,NGUSP1,LEVEL,LEVMAX,
     +        ANCESL,NTFINS,MTFINS,ITFIN,MAXLEV,ISODE,LQUSER
      DOUBLE PRECISION BVAL,DYLAG,DYNEW,DYOLD,DYOUT,DYROOT,QUEUE,
     +                 YLAG,YNEW,YOLD,YOUT,YROOT,WORK,RPAR,TDISC,
     +                 ABSERR,D1MACH,HDUMMY,HINIT,HINITM,HLEFT,HMAX,
     +                 HNEXT,RATIO,RELERR,TFINAL,TINC,TINIT,TNEW,
     +                 TOLD,TOUT,TROOT,TVAL,U,U13,U26,BIG,SMALL,
     +                 D6OUTP,TEMP,RER,U10,RDUM,U65,TFINS,TFNSAV,
     +                 TINSAV,TROOT2
      LOGICAL EXTRAP,NUTRAL
C
      DIMENSION YOLD(*),DYOLD(*),YNEW(*),DYNEW(*),YROOT(*),DYROOT(*),
     +          WORK(*),JROOT(*),YOUT(*),DYOUT(*),IPOINT(*),YLAG(*),
     +          BVAL(*),INDEXG(*),QUEUE(*),DYLAG(*),ABSERR(*),
     +          RELERR(*),RPAR(*),IPAR(*),LEVEL(*),TFINS(*)
C
C     DECLARE THE USER SUPPLIED PROBLEM DEPENDENT SUBROUTINES.
C
      EXTERNAL INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,DYINIT
C
C     DECLARE THE OUTPUT INCREMENT FUNCTION.
C
      EXTERNAL D6OUTP
C
C     DECLARE THE SUBROUTINE IN WHICH THE ROOT FUNCTION
C     RESIDUALS WILL BE EVALUATED.
C
      EXTERNAL D6GRT3
C
C     DEFINE THE ROUNDOFF FACTOR FOR THE TERMINATION
C     CRITERIA. THE INTEGRATION WILL BE TERMINATED
C     WHEN IT IS WITHIN U26 OF THE FINAL INTEGRATION
C     TIME. ALSO DEFINE THE FLOOR VALUE RER FOR THE
C     RELATIVE ERROR TOLERANCES.
C
C***FIRST EXECUTABLE STATEMENT  D6DRV4
C
      U = D1MACH(4)
      U10 = 10.0D0*U
      U13 = 13.0D0*U
      U26 = 26.0D0*U
      U65 = 65.0D0*U
      CALL D6RERS(RER)
C
C     INITIALIZE THE PROBLEM.
C
      EXTRAP = .FALSE.
      NUTRAL = .FALSE.
      CALL INITAL(LOUT,LIN,N,NGUSER,TOLD,TFINAL,TINC,YOLD,AUTORF,
     +            HINIT,HMAX,ITOL,ABSERR,RELERR,EXTRAP,NUTRAL,
     +            RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
      TFNSAV = TFINAL
      TINSAV = TOLD
c
c     Calculate the times at which derivative discontinuities
c     may propogate.
c
      mtfins = ntfins
      maxlev = 7
      call beta(n,tinsav,yold,bval,rpar,ipar)
c     Check if this is an ode (i.e., whether all the delays
c     are 0).
      isode = 1
      do 10 i = 1 , n
         if (bval(i) .ne. tinsav) isode = 0
   10 continue
      if (isode .eq. 1) then
         if (mtfins .lt. 2) then
            CALL D6MESG(43,'D6DRV4')
            IFLAG = 1
            go to 9005
         endif 
         extrap = .false.
         nutral = .false.
         mtfins = 2
         tfins(1) = tinsav
         tfins(2) = tfnsav
      else
c
c        Calculate the delays.
c
         do 15 i = 1 , n
            bval(i) = tinsav - bval(i)
   15    continue
c
c        Build the discontinuity tree. Note that the entire
c        history queue is used as the tfins array since
c        ntfins = lquser. Note also that d6find will call
c        d6fins and allow the inclusion of up to five additional
c        user defined points within the integration range.
c
         call d6find (tinsav, tfnsav, maxlev, bval, n, tfins,
     +                ntfins, mtfins, iflag, lout)
c
         if (iflag .ne. 0) then
            CALL D6MESG(44,'D6DRV4')
            IFLAG = 1
            if ((mtfins .lt. 2) .or. (mtfins .gt. ntfins)) then
               CALL D6MESG(45,'D6DRV4')
               IFLAG = 1
               go to 9005
            endif 
            go to 9005
         endif
c
c        Move the tfins points to the end of the solution queue.
c        Remember that tfins and queue are the same physical
c        array. Locations 1, ... , lquser-mtfins of the queue(*)
c        will be used for the history queue and locations
c        lquser-mtfins+1, ... , lquser will be used for the
c        tfins(*) points.
c
c        lqueue is the length of the history queue (and isn't
c        the same as the lqueue passed to dklg6c by the user).
c
         lqueue = lquser - mtfins
         call dcopy(mtfins,tfins(1),-1,tfins(lqueue+1),-1)
c
         told = tfins(lqueue+1)
         tfinal = tfins(lqueue+2)
      endif
C
C     TURN OFF DISCONTINUITY ROOTFINDING
      AUTORF = 1
C
C     AVOID PROBLEMS WITH THE APPROXIMATION OF DYLAG(I) FOR
C     NEUTRAL PROBLEMS WHEN BVAL(I) = T.
C
      IF (NUTRAL) EXTRAP = .TRUE.
C
      TINC = D6OUTP(TOLD,TINC)
C
      TINIT = TOLD
C
C     INITIALIZE THE HISTORY QUEUE LOCATION POINTER.
C
      IQUEUE = 0
C
C     CHECK FOR SOME COMMON ERRORS. ALSO RESET THE RELATIVE
C     ERROR TOLERANCES TO THE FLOOR VALUE IF NECESSARY.
C
      IF (ITOL.NE.1 .AND. ITOL.NE.N) THEN
         CALL D6MESG(36,'D6DRV4')
         GO TO 8995
      ENDIF
      IF (N.LT.1) THEN
          CALL D6MESG(3,'D6DRV4')
          GO TO 8995
      ENDIF
      IF (TFINAL.EQ.TOLD) THEN
          CALL D6MESG(8,'D6DRV4')
          GO TO 8995
      ENDIF
      DO 20 I = 1 , N
         IF (I.GT.ITOL) GO TO  20
         IF (RELERR(I).NE.0.0D0 .AND. RELERR(I).LT.RER) THEN
            RELERR(I) = RER
            CALL D6MESG(38,'D6DRV4')
         ENDIF
         IF (ABSERR(I).LT.0.0D0) THEN
             CALL D6MESG(5,'D6DRV4')
             GO TO 8995
         ENDIF
         IF (RELERR(I).LT.0.0D0) THEN
             CALL D6MESG(4,'D6DRV4')
             GO TO 8995
         ENDIF
         IF ((ABSERR(I).EQ.0.0D0).AND.(RELERR(I).EQ.0.0D0)) THEN
             CALL D6MESG(6,'D6DRV4')
             GO TO 8995
         ENDIF
   20 CONTINUE
      IF (TINC.EQ.0.0D0) THEN
          CALL D6MESG(22,'D6DRV4')
          GO TO 8995
      ENDIF
      IF ((AUTORF.LT.0) .OR. (AUTORF.GT.1)) THEN
          CALL D6MESG(23,'D6DRV4')
          GO TO 8995
      ENDIF
C
C     START WITH EITHER N+NGUSER OR NGUSER ROOT FUNCTIONS UNLESS
C     THE USER WISHES TO INPUT ADDITIONAL DISCONTINUITY INFORMATION.
C
      IF (NGUSER.LT.0) NGUSER = 0
      NG = 0
      IF (AUTORF.EQ.0) NG = N
      NG = NG + NGUSER
      NGOLD = NG
      NDISC = 0
C     THIS SECTION IS INACTIVE SINCE AUTORF = 1.
      IF (AUTORF.EQ. 0) THEN
C        DETERMINE THE NUMBER OF ADDITIONAL DISCONTINUITY ROOT
C        FUNCTIONS THE USER WISHES TO SPECIFY.
         DO 40 I = 1 , N
            IWANT = 1
            IDCOMP = I
            JCOMP = 0
            IDISC = 0
            TDISC = TOLD
            CALL D6DISC(IWANT,IDCOMP,IDISC,TDISC,JCOMP,RPAR,IPAR)
            IF (IDISC .GT. 0) THEN
               NDISC = NDISC + IDISC
            ENDIF
   40   CONTINUE
      ENDIF
      NG = NG + NDISC
      IPOINT(30) = NG
C
      IF ((NG.GT.0) .AND. (LIWORK.LT.NG)) THEN
          CALL D6MESG(21,'D6DRV4')
          GO TO 8995
      ENDIF
C
C     FORCE AN INTEGRATION RESTART AT ALL ROOTS (INCLUDING ROOTS
C     OF THE USER PROVIDED EVENT FUNCTIONS).
C
      RSTRT = 0
      IPOINT(29) = RSTRT
C
C     INITIALIZE THE INDEXG, JROOT, G-WORK ARRAYS, IF
C     ROOTFINDING IS TO BE DONE.
C
      IF (NG.GT.0) THEN
          CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,1)
          CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,2)
          INX1 = I9 + 6*NG - 1
          DO 60 IG = 1,NGOLD
              INDEXG(IG) = 0
              IF (IG.GT.NGUSER) INDEXG(IG) = IG - NGUSER
C             INITIALIZE TG(*) = TOLD
C             TG STARTS IN GWORK(6NG+1) WHERE GWORK(1)
C             IS THE SAME AS WORK(I9)
              WORK(INX1+IG) = TOLD
              JROOT(IG) = 0
   60     CONTINUE
C         ALLOW THE USER TO INPUT DISCONTINUITY INFORMATION FOR
C         THE INITIAL FUNCTION IF DISCONTINUITY ROOTFINDING IS
C         BEING USED. (THIS IS TURNED OFF TEMPORARILY IN THIS
C         VERSION.)
          IF (NDISC.GT.0) THEN
             IG = NGOLD
             DO 100 I = 1 , N
                IWANT = 1
                IDCOMP = I
                JCOMP = 0
                IDISC = 0
                TDISC = TOLD
                CALL D6DISC(IWANT,IDCOMP,IDISC,TDISC,JCOMP,RPAR,IPAR)
                IF (IDISC .GT. 0) THEN
                   DO 80 J = 1 , IDISC
                      IG = IG + 1
                      IWANT = 2
                      IDCOMP = I
                      JCOMP = J
                      TDISC = TOLD
                      CALL D6DISC(IWANT,IDCOMP,IDISC,TDISC,JCOMP,
     +                            RPAR,IPAR)
                      INDEXG(IG) = IDCOMP
C                     TG STARTS IN GWORK(6NG+1) WHERE GWORK(1)
C                     IS THE SAME AS WORK(I9)
                      WORK(INX1+IG) = TDISC
                      JROOT(IG) = 0
   80              CONTINUE
                ENDIF
  100        CONTINUE
          ENDIF
      ENDIF
C
C     INITIALIZE THE LEVEL TRACKING ARRAY.
C
      CALL D6LEVL (LEVMAX,NUTRAL)
      LEVMAX = MAX(LEVMAX,0)
      IF (LEVMAX .GT. 0 .AND. NG .GT. 0) THEN
         IF (NGUSER .GT. 0) THEN
            DO 105 I = 1 , NGUSER
               LEVEL(I) = -1
  105       CONTINUE
         ENDIF
         IF (NG .GT. NGUSER) THEN
            NGUSP1 = NGUSER + 1
            DO 110 I = NGUSP1, NG
               LEVEL(I) = 0
  110       CONTINUE
         ENDIF
      ENDIF
C
      DO 8960 ITFIN = 2 , MTFINS
C
C     RESET RELEVANT INTEGRATION PARAMETERS IF THIS
C     IS NOT THE FIRST TRIP THROUGH THE 8960 LOOP.
C
C     PERFORM REMAINING INITIALIZATIONS IF THIS IS
C     THE FIRST PASS THROUGH THE TFINS LOOP
C
C*********************************************************
C
      IF (ITFIN .EQ. 2) THEN
C
C     CALCULATE THE INITIAL DELAYS.
C
      CALL BETA(N,TOLD,YOLD,BVAL,RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
C
C     CHECK FOR ILLEGAL DELAYS.
C
      HDUMMY = TFINAL - TOLD
      CALL D6BETA(N,TOLD,HDUMMY,BVAL,IOK)
C
      IF (IOK.EQ.1) THEN
          INDEX = -11
          GO TO 8995
      ENDIF
C
C     GIVE TINC THE CORRECT SIGN.
C
      TINC = SIGN(ABS(TINC),TFINAL-TOLD)
C
C     CALCULATE THE INITIAL DERIVATIVES.
C
      DO 120 I = 1,N
          TVAL = BVAL(I)
          CALL YINIT(N,TVAL,YLAG(I),I,RPAR,IPAR)
          IF (N.LE.0) GO TO 8985
          CALL DYINIT(N,TVAL,DYLAG(I),I,RPAR,IPAR)
          IF (N.LE.0) GO TO 8985
  120 CONTINUE
      CALL DERIVS(N,TOLD,YOLD,YLAG,DYLAG,DYOLD,RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
C
C     FINISH THE INITIALIZATION.
C
      TINC = D6OUTP(TOLD,TINC)
C
      TOUT = TOLD + TINC
      INDEX = 1
      NSIG = 0
      NGEMAX = 500
      CALL D6RTOL(NSIG,NGEMAX,U10)
C     IDIR =  1 MEANS INTEGRATION TO THE RIGHT.
C     IDIR = -1 MEANS INTEGRATION TO THE LEFT.
      IDIR = 1
      IF (TFINAL.LT.TOLD) IDIR = -1
C
      IPOINT(28) = IDIR
C
C     OUTPUT THE INITIAL SOLUTION.
C
      ITYPE = 1
      CALL DPRINT(N,TOLD,YOLD,DYOLD,NG,JROOT,INDEXG,
     +            ITYPE,LOUT,INDEX,RPAR,IPAR)
      IF (N.LE.0) GO TO 8985
C
C     TURN OFF THE ROOTFINDING UNTIL THE SECOND STEP IS COMPLETED.
C
      IFIRST = 0
      IPOINT(26) = IFIRST
      IFOUND = 0
      IPOINT(27) = IFOUND
C
C     CALCULATE THE INITIAL STEPSIZE IF THE USER CHOSE NOT TO
C     DO SO.
C________________________________________
C
      IF (HINIT .EQ. 0.0D0) THEN
C
         DO 140 I = 1 , N
            ICOMP = I 
            IF (I.GT.ITOL) ICOMP = 1
            WORK(I) = 0.5D0 * (ABSERR(ICOMP)
     +                + RELERR(ICOMP) * ABS(YOLD(I)))
            IF (WORK(I) .EQ. 0.0D0) THEN
               CALL D6MESG(10,'D6DRV4')
               GO TO 8995
            ENDIF
  140    CONTINUE
         MORDER = 6
         SMALL = D1MACH(4)
         BIG = SQRT(D1MACH(2))
         CALL D6HST1(DERIVS, N, TOLD, TOUT, YOLD, DYOLD, WORK(1),
     +               MORDER, SMALL, BIG, WORK(N+1), WORK(2*N+1),
     +               WORK(3*N+1), WORK(4*N+1), WORK(5*N+1),
     +               YLAG, DYLAG, BETA, YINIT, DYINIT,
     +               IHFLAG, HINIT, IPAIR, RPAR, IPAR)
         IF (N.LE.0) GO TO 8985
         IF (IHFLAG .NE. 0) GO TO 8995
C
C     THIS IS THE END OF THE IF HINIT = 0 BLOCK.
C
      ENDIF
C________________________________________
C
C     CHECK IF THE INITIAL STEPSIZE IS TOO SMALL.
C 
      TEMP = MAX(ABS(TOLD),ABS(TOUT))
      TEMP = U13*TEMP
      IF (ABS(HINIT) .LE. TEMP) THEN
         CALL D6MESG(11,'D6INT1')
         GO TO 8995
      ENDIF
C
      CALL D6STAT(N, 2, HINIT, RPAR, IPAR, IDUM, RDUM)
C
C     LIMIT THE INITIAL STEPSIZE TO BE NO LARGER THAN
C     ABS(TFINAL-TOLD) AND NO LARGER THAN HMAX.
C
      HINITM = MIN(ABS(HINIT),ABS(TFINAL-TOLD))
      HINIT = SIGN(HINITM,TFINAL-TOLD)
      HMAX = SIGN(ABS(HMAX),TFINAL-TOLD)
      HNEXT = HINIT
C
C     THIS IS THE END OF THE IF ITFINS = 2 BLOCK.
C
      ENDIF
C*********************************************************
C
C     CALL TO THE INTEGRATOR TO ADVANCE THE INTEGRATION
C     ONE STEP FROM TOLD. REFER TO THE DOCUMENTATION
C     FOR D6DRV3 FOR DESCRIPTIONS OF THE FOLLOWING
C     PARAMETERS.
C
  160 CONTINUE
C
      HLEFT = TFINS(LQUEUE+ITFIN+1) - TOLD
      HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFNSAV-TNEW)
c
      CALL D6DRV3(N,TOLD,YOLD,DYOLD,TNEW,YNEW,DYNEW,TROOT,YROOT,DYROOT,
     +            TFINAL,DERIVS,ITOL,ABSERR,RELERR,NG,NGUSER,D6GRT3,
     +            GUSER,NSIG,JROOT,INDEXG,WORK,LWORK,IPOINT,NFEVAL,
     +            NGEVAL,RATIO,HNEXT,HMAX,YLAG,DYLAG,BETA,BVAL,YINIT,
     +            DYINIT,QUEUE,LQUEUE,IQUEUE,INDEX,EXTRAP,NUTRAL,
     +            RPAR,IPAR,IER,TROOT2,LOUT)
c
C     CHECK FOR USER REQUESTED TERMINATION.
C
      IF (N.LE.0) GO TO 8985
      IF (INDEX.EQ.-14) GO TO 8985
C
C     TERMINATE THE INTEGRATION IF AN ABNORMAL RETURN
C     WAS MADE FROM THE INTEGRATOR.
C
      IF (INDEX.LT.2) GO TO 8995
C
C     MAKE ANY NECESSARY CHANGES IF A ROOT OCCURRED AT
C     THE INITIAL POINT. THEN CALL THE INTEGRATOR AGAIN.
C
C###############################################################
C
      IF (INDEX.EQ.5) THEN
C
         OFFRT = 1
         NTEMP = N
         NGTEMP = NG
         HINIT = HNEXT
         CALL CHANGE(INDEX,JROOT,INDEXG,NG,N,YROOT,DYROOT,HINIT,TROOT,
     +               TINC,ITOL,ABSERR,RELERR,LOUT,RPAR,IPAR)
C
         TINC = D6OUTP(TROOT,TINC)
C
C        TERMINATE IF THE USER WISHES TO.
C
         IF (N.LE.0) GO TO 8985
         IF (HINIT.EQ.0.0D0) THEN
            CALL D6MESG(39,'D6DRV4')
            GO TO 8995
         ENDIF
C
C        CHECK IF ANY ILLEGAL CHANGES WERE MADE.
C        ALSO RESET THE RELATIVE ERROR TOLERANCES
C        IF NECESSARY.
C
         IF (ITOL.NE.1 .AND. ITOL.NE.N) THEN
            CALL D6MESG(37,'D6DRV4')
            GO TO 8995
         ENDIF
         IF (NTEMP.NE.N) THEN
             CALL D6MESG(30,'D6DRV4')
             GO TO 8995
         ENDIF
C
C        MAKE NECESSARY CHANGES IF THE USER TURNED OFF THE
C        AUTOMATIC ROOTFINDING.
C
         IF (NG.EQ.-1) THEN
            NG = NGUSER
            NGTEMP = NG
            OFFRT = 0
            CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                  I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                  IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                  JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,1)
            CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,
     +                  I12,I13,I14,I15,I9M1,I10M1,I11M1,IORDER,IFIRST,
     +                  IFOUND,IDIR,RSTRT,IERRUS,LQUEUE,IQUEUE,
     +                  JQUEUE,KQUEUE,TINIT,WORK,IK7,IK8,IK9,IPAIR,2)
         ENDIF
         IF (NGTEMP.NE.NG) THEN
             CALL D6MESG(31,'D6DRV4')
             GO TO 8995
         ENDIF
         DO 180 I = 1 , N
            IF (I.GT.ITOL) GO TO 180
            IF (RELERR(I).NE.0.0D0 .AND. RELERR(I).LT.RER)
     +      RELERR(I) = RER
            IF (ABSERR(I).LT.0.0D0) THEN
                CALL D6MESG(5,'D6DRV4')
                GO TO 8995
            ENDIF
            IF (RELERR(I).LT.0.0D0) THEN
                CALL D6MESG(4,'D6DRV4')
                GO TO 8995
            ENDIF
            IF ((ABSERR(I).EQ.0.0D0).AND.(RELERR(I).EQ.0.0D0)) THEN
                CALL D6MESG(6,'D6DRV4')
                GO TO 8995
            ENDIF
  180    CONTINUE
         IF (TINC.EQ.0.0D0) THEN
             CALL D6MESG(22,'D6DRV4')
             GO TO 8995
         ENDIF
C
         GO TO 160
C
C        THIS IS THE END OF THE IF INDEX = 5 BLOCK.
C
      ENDIF
C
C###############################################################
C
C     TURN ON THE ROOTFINDING.
C
      IFIRST = 1
      IPOINT(26) = IFIRST
C
C     IS THIS A ROOT RETURN?
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      IF ((INDEX.EQ.3) .OR. (INDEX.EQ.4)) THEN
C
C   *******************************
C   * YES, THIS IS A ROOT RETURN. *
C   *******************************
C
  200     CONTINUE
C
C         HAVE WE PASSED AN OUTPUT POINT?
C
          IDIR = IPOINT(28)
          IF (((TROOT.GT.TOUT).AND. (IDIR.EQ.1)) .OR.
     +        ((TROOT.LT.TOUT).AND. (IDIR.EQ.-1))) THEN
C
C             WE HAVE PASSED THE NEXT OUTPUT POINT.
C             (TROOT IS BEYOND TOUT.)
C
C             INTERPOLATE AT TOUT.
C
              CALL D6INT8(N,NG,TOLD,YOLD,TNEW,WORK,TOUT,YOUT,DYOUT,
     +                    IPOINT)
C
C             MAKE A PRINT RETURN AT TOUT.
C
              ITYPE = 1
              CALL DPRINT(N,TOUT,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,LOUT,
     +                    INDEX,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C             UPDATE THE PRINT RETURN OUTPUT POINT.
C
              TINC = D6OUTP(TOUT,TINC)
C
              TOUT = TOUT + TINC
C
C             LOOP UNTIL THE OUTPUT POINT TOUT LIES BEYOND
C             THE ROOT TROOT.
C
              GO TO 200
C
          ENDIF
C
C         THE ROOT TROOT DOES NOT LIE BEYOND THE NEXT OUTPUT POINT.
C
C         MAKE A PRINT RETURN AT THE ROOT.
C
          ITYPE = 0
          CALL DPRINT(N,TROOT,YROOT,DYROOT,NG,JROOT,INDEXG,ITYPE,LOUT,
     +                INDEX,RPAR,IPAR)
          IF (N.LE.0) GO TO 8985
C
C         DO WE RESTART THE INTEGRATION AT THE ROOT OR
C         CONTINUE THE INTEGRATION FROM TNEW?
C
          RSTRT = IPOINT(29)
C
C         NOTE:
C         IF RSTRT = 0, A CALL TO SUBROUTINE CHANGE WILL BE MADE
C         AT THE INITIAL POINT.
C
          IF (RSTRT.EQ.0) THEN
C
C      -------------------------------------
C        RESTART THE INTEGRATION AT TROOT.
C      -------------------------------------
C
C             FIRST CHECK IF WE ARE CLOSE ENOUGH TO THE FINAL
C             INTEGRATION TIME TO TERMINATE THE INTEGRATION.
C
              HLEFT = TFINAL - TROOT
C
              IF ( (ABS(HLEFT).LE.U26*MAX(ABS(TROOT),
     +                                    ABS(TFINAL)))
     +             .AND. (ITFIN.EQ.MTFINS)) THEN
C
C                 IF WE ARE CLOSE ENOUGH, PERFORM A FORWARD EULER
C                 EXTRAPOLATION TO TFINAL.
C
                  DO 220 I = 1,N
                      YOUT(I) = YROOT(I) + HLEFT*DYROOT(I)
  220             CONTINUE
C
C                 CALCULATE THE DERIVATIVES FOR THE FINAL
C                 EXTRAPOLATED SOLUTION.
C
C                 NOTE:
C                 THE ERROR VECTOR WORK(I8) WILL BE USED
C                 AS SCRATCH STORAGE.
C
                  I8 = IPOINT(8)
                  CALL BETA(N,TFINAL,YOUT,BVAL,RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
                  CALL D6BETA(N,TFINAL,HNEXT,BVAL,IOK)
C
                  IF (IOK.EQ.1) THEN
                      INDEX = -11
                      GO TO 8995
                  ENDIF
C
                  CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
     +                        YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
     +                        IPOINT,WORK,TINIT,EXTRAP,JNDEX,2,1,
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C                 D6INT1 DOES NOT APPROXIMATE DYLAG(I) IF BVAL(I)=T.
                  IF (NUTRAL) THEN
                     DO 240 I = 1,N
                        IF (BVAL(I).EQ.TFINAL) THEN
C                          EXTRAPOLATE FOR DYLAG.
                           DYLAG(I) = DYROOT(I)
                        ENDIF
  240                CONTINUE
                  ENDIF
C
                  IF (JNDEX.LT.0) THEN
                      INDEX = JNDEX
                      GO TO 8995
                  ENDIF
C
                  CALL DERIVS(N,TFINAL,YOUT,YLAG,DYLAG,DYOUT,
     +                        RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C
C                 MAKE A PRINT RETURN FOR THE FINAL INTEGRATION TIME.
C
                  ITYPE = 1
                  CALL DPRINT(N,TFINAL,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,
     +                        LOUT,INDEX,RPAR,IPAR)
                  IF (N.LE.0) GO TO 8985
C
C                 ADJUST THE FINAL MESH TIME IN THE HISTORY QUEUE.
C
                  CALL D6QUE2(N,TFINAL,IPOINT,QUEUE)
C
C                 TERMINATE THE INTEGRATION.
C
                  GO TO 8950
C
C             THIS IS THE END OF THE IF RSTRT = 0 BLOCK.
C
              ENDIF
C
C             IT IS NOT YET TIME TO TERMINATE THE INTEGRATION SO
C             PREPARE FOR THE RESTART AT THE ROOT.
C
C             RESET THE INTEGRATION FLAG.
C
              INDEX = 1
C
C             TURN OFF THE ROOTFINDING UNTIL THE SECOND STEP
C             IS COMPLETED.
C
              IFIRST = 0
              IPOINT(26) = IFIRST
              IFOUND = 0
              IPOINT(27) = IFOUND
C
C             MAKE ANY NECESSARY CHANGES IN THE PROBLEM.
C
              OFFRT = 1
              NTEMP = N
              NGTEMP = NG
              HINIT = HNEXT
              HINIT = SIGN(MAX(ABS(HINIT),U65),HINIT)
              CALL CHANGE(INDEX,JROOT,INDEXG,NG,N,YROOT,DYROOT,HINIT,
     +                    TROOT,TINC,ITOL,ABSERR,RELERR,LOUT,
     +                    RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
              HINIT = SIGN(MAX(ABS(HINIT),U65),HINIT)
C
              TINC = D6OUTP(TOUT,TINC)
C
C             TERMINATE IF THE USER WISHES TO.
C
              IF (INDEX.LT.0 .OR. HINIT.EQ.0.0D0) THEN
                  CALL D6MESG(25,'D6DRV4')
                  GO TO 8950
              ENDIF
C
C             CHECK IF ANY ILLEGAL CHANGES WERE MADE.
C             ALSO RESET THE RELATIVE ERROR TOLERANCES
C             IF NECESSARY.
C
              IF (ITOL.NE.1 .AND. ITOL.NE.N) THEN
                 CALL D6MESG(37,'D6DRV4')
                 GO TO 8995
              ENDIF
              IF (NTEMP.NE.N) THEN
                  CALL D6MESG(30,'D6DRV4')
                  GO TO 8995
              ENDIF
C
C             MAKE NECESSARY CHANGES IF THE USER TURNED OFF THE
C             AUTOMATIC ROOTFINDING.
C
              IF (NG.EQ.-1) THEN
                 NG = NGUSER
                 NGTEMP = NG
                 OFFRT = 0
                 CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,
     +                       I10,I11,I12,I13,I14,I15,I9M1,I10M1,
     +                       I11M1,IORDER,IFIRST,IFOUND,IDIR,RSTRT,
     +                       IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +                       TINIT,WORK,IK7,IK8,IK9,IPAIR,1)
                 CALL D6PNTR(N,NG,IPOINT,I1,I2,I3,I4,I5,I6,I7,I8,I9,
     +                       I10,I11,I12,I13,I14,I15,I9M1,I10M1,
     +                       I11M1,IORDER,IFIRST,IFOUND,IDIR,RSTRT,
     +                       IERRUS,LQUEUE,IQUEUE,JQUEUE,KQUEUE,
     +                       TINIT,WORK,IK7,IK8,IK9,IPAIR,2)
              ENDIF
C
              IF (NGTEMP.NE.NG) THEN
                  CALL D6MESG(30,'D6DRV4')
                  GO TO 8995
              ENDIF
              DO 260 I = 1 , N
                 IF (I.GT.ITOL) GO TO 260
                 IF (RELERR(I).NE.0.0D0 .AND. RELERR(I).LT.RER)
     +           RELERR(I) = RER
                 IF (ABSERR(I).LT.0.0D0) THEN
                     CALL D6MESG(5,'D6DRV4')
                     GO TO 8995
                 ENDIF
                 IF (RELERR(I).LT.0.0D0) THEN
                     CALL D6MESG(4,'D6DRV4')
                     GO TO 8995
                 ENDIF
                 IF ((ABSERR(I).EQ.0.0D0).AND.(RELERR(I).EQ.0.0D0)) THEN
                     CALL D6MESG(6,'D6DRV4')
                     GO TO 8995
                 ENDIF
  260         CONTINUE
              IF (TINC.EQ.0.0D0) THEN
                  CALL D6MESG(22,'D6DRV4')
                  GO TO 8995
              ENDIF
C
C          ______________________________________
C
           IF (OFFRT .NE. 0) THEN
C
C             SHUFFLE THE STORAGE AND INITIALIZE THE NEW LOCATIONS
C             TO ACCOMMODATE THE ADDITIONAL ROOT FUNCTIONS.
C
C             DETERMINE HOW MANY OF THE ROOT FUNCTIONS HAVE
C             A ZERO AT TROOT.
C
              KROOT = 0
              ANCESL = -1
              IF (NG.GT.NGUSER) THEN
                 ANCESL = 10000
                 NGUSP1 = NGUSER + 1
                 DO 280 I = NGUSP1,NG
                    IF (JROOT(I).EQ.1) THEN
                       KROOT = KROOT + 1
C                      MINIMUM LEVEL FROM WHICH THIS ROOT CAME.
                       IF (LEVMAX.GT.0) ANCESL = MIN(ANCESL,LEVEL(I))
                    ENDIF
  280            CONTINUE
                 IF (KROOT .NE. 0) KROOT = N
                 IF (LEVMAX.GT.0) THEN
C                   DO NOT ADD THIS ROOT IF HAVE REACHED THE
C                   MAXIMUM TRACKING LEVEL.
                    IF (ANCESL .GE. LEVMAX) THEN
                       KROOT = 0
                    ENDIF
                 ENDIF
              ENDIF
C
C             CHECK IF THERE IS ENOUGH STORAGE TO CONTINUE.
C
              INEED = 21*N + 1 + 7 * (NG+KROOT)
              IF (INEED.GT.LWORK) THEN
                  CALL D6MESG(9,'D6DRV4')
                  GO TO 8995
              ENDIF
C
              INEED = 2 * (NG + KROOT)
              IF (INEED.GT.LIWORK) THEN
                  CALL D6MESG(21,'D6DRV4')
                  GO TO 8995
              ENDIF
C
C             SHUFFLE THE OLD WORK LOCATIONS.
C
              IF (KROOT.GT.0)
     +        CALL D6GRT4(WORK,NG,KROOT,JROOT,INDEXG,LEVEL,
     +                    TROOT,IPOINT,LEVMAX,ANCESL)
C
C             DEFINE THE NEW NUMBER OF ROOT FUNCTIONS.
C
              NG = NG + KROOT
              IF (OFFRT .EQ. 0) NG = NGUSER
              IPOINT(30) = NG
C
C          THIS IS THE END OF THE IF OFFRT = 0 BLOCK.
C
           ENDIF
C          ______________________________________
C
C             RE-SET THE INITIAL TIME.
C
              TOLD = TROOT
C
C             DEFINE THE INITIAL STEPSIZE FOR THE RESTART.
C
              HNEXT = HINIT
C
C             IF A SECOND ROOT WAS LOCATED IN (TROOT,TNEW)
C             LIMIT THE STEPSIZE SO AS NOT TO MISS IT ON
C             RESTARTING.
C
              IF (TROOT2 .NE. TROOT) THEN
                 HLEFT = 0.25D0 * (TROOT2 -TROOT)
                 HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFINAL-TNEW)
              ENDIF
C
C             USE THE INTERPOLATED SOLUTION AT THE ROOT TO
C             UPDATE THE LEFT SOLUTION.
C
              CALL DCOPY(N,YROOT,1,YOLD,1)
C
C             UPDATE THE LEFT DERIVATIVES WITH THE SYSTEM
C             DERIVATIVES FOR THE NEW LEFT SOLUTION (NOT
C             THE INTERPOLATED DERIVATIVES IN DYROOT).
C
C             NOTE:
C             THE ERROR VECTOR WORK(I8) WILL BE USED
C             AS SCRATCH STORAGE.
C
              I8 = IPOINT(8)
              CALL BETA(N,TOLD,YOLD,BVAL,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
              CALL D6BETA(N,TOLD,HNEXT,BVAL,IOK)
C
              IF (IOK.EQ.1) THEN
                  INDEX = -11
                  GO TO 8995
              ENDIF
C
              CALL D6INT1(N,NG,TNEW,YNEW,HNEXT,BVAL,WORK(I8),YLAG,
     +                    DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,IPOINT,
     +                    WORK,TINIT,EXTRAP,JNDEX,2,1,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C             D6INT1 DOES NOT APPROXIMATE DYLAG(I) IF BVAL(I)=T.
              IF (NUTRAL) THEN
                 DO 300 I = 1,N
                    IF (BVAL(I).EQ.TNEW) THEN
C                      EXTRAPOLATE FOR DYLAG.
                       DYLAG(I) = DYROOT(I)
                    ENDIF
  300            CONTINUE
              ENDIF
C
              IF (JNDEX.LT.0) THEN
                  INDEX = JNDEX
                  GO TO 8995
              ENDIF
C
              CALL DERIVS(N,TOLD,YOLD,YLAG,DYLAG,DYOLD,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C             RESET THE STEPSIZE IF IT WILL THROW US BEYOND
C             THE FINAL INTEGRATION TIME ON THE NEXT STEP.
C
              HLEFT = TFINAL - TROOT
              HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFINAL-TROOT)
C
C             CALL THE INTEGRATOR AGAIN.
C
              GO TO 160
C
C         THIS IS THE END OF THE IF INDEX = 3 OR INDEX = 4 BLOCK.
C
          ENDIF
C
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
C   **********************************
C   * NO, THIS IS NOT A ROOT RETURN. *
C   **********************************
C
      ELSE
C
  360     CONTINUE
C
C         HAVE WE REACHED AN OUTPUT POINT?
C
          IDIR = IPOINT(28)
          IF (((TNEW.GE.TOUT).AND. (IDIR.EQ.1)) .OR.
     +        ((TNEW.LE.TOUT).AND. (IDIR.EQ.-1))) THEN
C
C             WE HAVE PASSED THE NEXT OUTPUT POINT.
C             (TNEW IS BEYOND TOUT.)
C
C             INTERPOLATE AT TOUT.
C
              CALL D6INT8(N,NG,TOLD,YOLD,TNEW,WORK,TOUT,YOUT,DYOUT,
     +                    IPOINT)
C
C             MAKE A PRINT RETURN AT TOUT.
C
              ITYPE = 1
              CALL DPRINT(N,TOUT,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,LOUT,
     +                    INDEX,RPAR,IPAR)
              IF (N.LE.0) GO TO 8985
C
C             UPDATE THE PRINT RETURN OUTPUT POINT.
C
              TINC = D6OUTP(TOUT,TINC)
C
              TOUT = TOUT + TINC
C
C             LOOP UNTIL THE PRINT OUTPUT POINT TOUT LIES BEYOND
C             THE INTEGRATOR OUTPUT POINT TNEW.
C
              GO TO 360
C
C         THIS IS THE END OF THE 360 LOOPING SECTION.
C
          ENDIF
C
C         TNEW DOES NOT LIE BEYOND THE NEXT OUTPUT POINT.
C         PREPARE FOR ANOTHER CALL TO THE INTEGRATOR OR
C         TERMINATE THE INTEGRATION.
C
C         EXTRAPOLATE THE SOLUTION TO TFINAL AND TERMINATE
C         THE INTEGRATION IF CLOSE ENOUGH TO TFINAL.
C
          HLEFT = TFINAL - TNEW
C
          IF (ABS(HLEFT).LE.U26*MAX(ABS(TNEW),ABS(TFINAL)))
     +       THEN
             IF (ITFIN .EQ. MTFINS) THEN
C
C               IF WE ARE CLOSE ENOUGH, PERFORM A FORWARD EULER
C               EXTRAPOLATION TO TFINAL.
C
                DO 380 I = 1,N
                   YOUT(I) = YNEW(I) + HLEFT*DYNEW(I)
  380           CONTINUE
C
C               CALCULATE THE DERIVATIVES FOR THE FINAL
C               EXTRAPOLATED SOLUTION.
C
C               NOTE:
C               THE ERROR VECTOR WORK(I8) WILL BE USED
C               AS SCRATCH STORAGE.
C
                I8 = IPOINT(8)
                CALL BETA(N,TFINAL,YOUT,BVAL,RPAR,IPAR)
                IF (N.LE.0) GO TO 8985
                CALL D6BETA(N,TFINAL,HNEXT,BVAL,IOK)
C
                IF (IOK.EQ.1) THEN
                   INDEX = -11
                   GO TO 8995
                ENDIF
C
                CALL D6INT1(N,NG,TFINAL,YOUT,HNEXT,BVAL,WORK(I8),
     +                      YLAG,DYLAG,YINIT,DYINIT,QUEUE,LQUEUE,
     +                      IPOINT,WORK,TINIT,EXTRAP,JNDEX,2,1,
     +                      RPAR,IPAR)
                IF (N.LE.0) GO TO 8985
C               D6INT1 DOES NOT APPROXIMATE DYLAG(I) IF BVAL(I)=T.
                IF (NUTRAL) THEN
                   DO 400 I = 1,N
                      IF (BVAL(I).EQ.TFINAL) THEN
C                        EXTRAPOLATE FOR DYLAG.
                         DYLAG(I) = DYNEW(I)
                      ENDIF
  400              CONTINUE
                ENDIF
C
                IF (JNDEX.LT.0) THEN
                   INDEX = JNDEX
                   GO TO 8995
                ENDIF
C
                CALL DERIVS(N,TFINAL,YOUT,YLAG,DYLAG,DYOUT,
     +                      RPAR,IPAR)
                IF (N.LE.0) GO TO 8985
C
C               MAKE A PRINT RETURN FOR THE FINAL INTEGRATION TIME.
C
                ITYPE = 1
                CALL DPRINT(N,TFINAL,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,
     +                      LOUT,INDEX,RPAR,IPAR)
                IF (N.LE.0) GO TO 8985
C
C               TERMINATE THE INTEGRATION.
C
                GO TO 8950
C
             ELSE
C
C               WE HAVE NOT REACHED THE FINAL TFINS(LQUEUE+MTFINS).
C               PREPARE FOR THE NEXT TRIP THROUGH THE LOOP.
C
C               MAKE A PRINT RETURN FOR THIS INTEGRATION TIME.
C
                ITYPE = 1
                CALL DCOPY(N,YNEW,1,YOUT,1)
                CALL DCOPY(N,DYNEW,1,DYOUT,1)
                CALL DPRINT(N,TFINAL,YOUT,DYOUT,NG,JROOT,INDEXG,ITYPE,
     +                      LOUT,INDEX,RPAR,IPAR)
                IF (N.LE.0) GO TO 8985
C
                HLEFT = TFINS(LQUEUE+ITFIN+1) - TNEW
                HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFNSAV-TNEW)
                TOLD = TNEW
                TROOT = TNEW
                TFINAL = TFINS(LQUEUE+ITFIN+1)
                CALL DCOPY(N,YNEW,1,YOLD,1)
                CALL DCOPY(N,DYNEW,1,DYOLD,1)
                INDEX = 1
C
                GO TO 8960
C
             ENDIF
C
          ENDIF
C
C         IT IS NOT YET TIME TO TERMINATE THE INTEGRATION.
C         PREPARE TO CALL THE INTEGRATOR AGAIN TO CONTINUE
C         THE INTEGRATION FROM TNEW.
C
C         UPDATE THE INDEPENDENT VARIABLE.
C
          TOLD = TNEW
C
C         UPDATE THE LEFT SOLUTION.
C
          CALL DCOPY(N,YNEW,1,YOLD,1)
C
C         UPDATE THE LEFT DERIVATIVES WITH THE SYSTEM
C         DERIVATIVES FOR THE NEW LEFT SOLUTION.
C
          CALL DCOPY(N,DYNEW,1,DYOLD,1)
C
C         RESET THE STEPSIZE IF IT WILL THROW US BEYOND
C         THE FINAL INTEGRATION TIME ON THE NEXT STEP.
C
          HLEFT = TFINAL - TNEW
          HNEXT = SIGN(MIN(ABS(HNEXT),ABS(HLEFT)),TFINAL-TNEW)
C
C         CALL THE INTEGRATOR AGAIN.
C
          GO TO 160
C
      ENDIF
C
C     ----------------------------------------------------------
C
 8950 CONTINUE
C
C     THE TFINAL OUTPUT LOOP ENDS HERE (AT 8960).
C
 8960 CONTINUE
C
C     RETURNS TO DKLG6C ARE MADE IN THIS SECTION.
C
C     NORMAL TERMINATION.
C
      IFLAG = 0
      GO TO 9005
C
 8985 CONTINUE
C
C     USER TERMINATION.
C
      CALL D6MESG(25,'D6DRV4')
      IFLAG = 1
      GO TO 9005
C
 8995 CONTINUE
C
C     ABNORMAL TERMINATION.
C
      CALL D6MESG(19,'D6DRV4')
      IFLAG = 1
C
 9005 CONTINUE
C
      RETURN
      END
C*DECK D6FIND
      subroutine d6find (tinit, tfinal, maxlev, delays,
     +           numlags, t, maxlen, lent, iflag,
     +           lout)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6FIND
C
C***PURPOSE  Build the discontinuity tree for D6DRV4.
C
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
c     If all delays are known to be constant, d6find
c     may be used to compute the array of times at
c     which discontinuities may occur. The rootfinding
c     used in ddrlag may be bypassed by first having
c     ddrlag step exactly to each of the potential
c     discontinuity times.
c
c              tdisc(2), ... , tdisc(lent).
c
c     Input:
c
c     tinit   - initial integration time
c     tfinal  - final integration time (> tinit)
c     maxlev  - maximum derivative level to flag for
c               possible derivative discontinuities
c               (1 <= maxlev <= order of integration
c               method)
c     delays  - vector of delays (not delay times)
c                 delays(*) = tinit - bval(*)
c               If tinit < tfinal, each delay must be >=0.
c               If tinit > tfinal, each delay must be <=0.
c     numlags - length of the delays vector n (usually
c               n, the number of ddes). Must be >0.
c     maxlen  - maximum length of the discontinuity
c               array
c     t       - array of length maxlen
c     lout    - output unit for error messages
c
c     Output:
c
c     Note: The contents of delays(*) will be altered.
c
c     t       - the discontinuity array
c     lent    - length of the discontinuity array
c               If tinit < tfinal, the discontinuity times are
c               t(1) = tinit < t(2) < ... < t(lent) = tfinal.
c               If tinit > tfinal, the discontinuity times are
c               t(1) = tinit > t(2) > ... > t(lent) = tfinal.
c               Duplicates in the tree are discarded. Values
c               are considered duplicates if they differ by
c               less than tol = u13*max(abs(tinit),abs(tfinal))
c               where u13 is 13 units of roundoff.
c     iflag   - error flag
c               An error occurred if iflag is not 0; refer
c               to the printed error message. The contents
c               of maxlen an t(*) are valid only if iflag = 0.
c
      implicit none
      integer maxlev, numlag, i, j, k, iflag, maxlen,
     +        lent, itend, c, numl, numlags, last,
     +        added, lout, isode, idir, nexmax,
     +        nextra, cp1, csave
      double precision tinit, tfinal, delays, t, tol,
     +        tc, u13, d1mach
      dimension delays(numlags), t(maxlen)
c
      iflag = 0
      lent = 0
      numlag = numlags
      u13 = 13.0d0 * d1mach(4)
      tol = u13 * max(abs(tinit),abs(tfinal))
c
c     Check for errors
c
      if (maxlen .lt. 2) then
         iflag = 4
         CALL D6MESG(49,'D6FIND')
         return
      else
         lent = 2
         t(1) = tinit
         t(2) = tfinal
      endif
      if ((maxlev.lt.1).or.(maxlev.gt.7)) then
         iflag = 1
         CALL D6MESG(46,'D6FIND')
         return
      endif
      if (numlag .lt. 1)then
         iflag = 2
         CALL D6MESG(47,'D6FIND')
         return
      endif
      if (tinit .eq. tfinal) then
         iflag = 5
         CALL D6MESG(50,'D6FIND')
      endif
c
c     Determine the direction of the integration
c
c     Left to right
      idir = -1
c     Right to left
      if (tfinal .lt. tinit) idir = 1
c
      do 20 i = 1 , numlag
         if (dble(idir)*delays(i) .gt. 0.0d0) then
            iflag = 3
            CALL D6MESG(48,'D6FIND')
            return
         endif
   20 continue
c
c     We may be dealing with an ode with all delays = 0
c
      isode = 1
      do 30 i = 1 , numlag
         if (delays(i) .ne. tinit) isode = 0
   30 continue
      if (isode .eq. 1) return
c
c     Sort the delays (decreasing if idir = -1 and
c     increasing if idir = 1) and discard unnecessary
c     values and those too close together
c
      numl = numlag
      if (numlag .gt. 1) then
         call dksort (delays, delays, numlag, idir, iflag)
         if (iflag .ne. 0) return
         do 40 i = numlag, 2, -1
            if ((abs(delays(i-1)-delays(i)) .le. tol) .or.
     +          (delays(i).eq.tinit)) then
                tc = delays(i-1) - delays(i)
                numl = numl - 1
            endif
   40    continue
      endif 
      if (numl .lt. 1) then
         return
      endif
      numlag = numl
c
c     Generate the discontinuity tree
c
      c = 1
      t(c) = tinit
      last = 1
      do 100 k = 1 , numlag
         do 80 i = 1 , last
            added = 0
            do 60 j = 1 , maxlev
               c = c + 1
               if (maxlen .lt. c) then
                  iflag = 4
                  CALL D6MESG(49,'D6FIND')
                  return
               endif
               tc = t(i) + dble(j)*delays(k)
               if ( ( (tc.le.tfinal) .and. (idir.eq.-1) ) .or.
     +              ( (tc.ge.tfinal) .and. (idir.eq. 1) ) )
     +            then
                  t(c) = tc
                  added = added + 1
               else
                  c = c - 1
               endif
   60       continue
   80    continue
         last = last + added
  100 continue
      lent = c
c
c     Allow the inclusion of up to five additional
c     internal break points
c
      if (maxlen .lt. c+5) then
         iflag = 4
         CALL D6MESG(49,'D6FIND')
         return
       endif
       nexmax = 5
       nextra = 0
       cp1 = c + 1
       csave = c
       call d6fins(nextra, t(cp1))
       if ((nextra.lt.0) .or. (nextra.gt.nexmax)) then
          iflag = 8
          CALL D6MESG(53,'D6FIND')
          return
       endif
       if (nextra .gt. 1) then
          do 110 i = 1 , nextra
             if (idir .eq. -1) then
                if ((tinit .lt. t(csave+i)) .and.
     +             (t(csave+i) .lt. tfinal)) then
                   c = c + 1
                   t(c) = t(csave+i)
                endif
             else
                if ((tinit.gt. t(csave+i)) .and.
     +             (t(csave+i) .gt. tfinal)) then
                   c = c + 1
                   t(c) = t(csave+i)
                endif
             endif
  110     continue
      endif
c
c     Add tfinal if necessary
c
      if (t(lent) .ne. tfinal) then
         c = c + 1
         if (maxlen .lt. c) then
            iflag = 4
            CALL D6MESG(49,'D6FIND')
            return
         endif
         lent = c
         t(lent) = tfinal
      endif
c
c     Sort t (increasing if idir = -1 and decreasing 
c     if idir = 1) and drop values beyond tfinal or
c     too close together
c
      if (lent .gt. 1) then
         write(lout,307) (i, t(i),i=1,lent)
  307    format(' Unsorted discontinuity tree:',/,(i3,f10.2))
         call dksort (t, t, lent, -idir, iflag)
         if (iflag .ne. 0) return
         itend = c
         lent = c
         do 120 i = lent , 1 , -1
            if ( ( (t(i).gt.tfinal).and.(idir.eq.-1) ) .or.
     +           ( (t(i).lt.tfinal).and.(idir.eq. 1) ) )
     +         then
               itend = itend - 1
            endif
  120    continue
         lent = itend
         do 160 i = lent, 2, -1
            if (abs(t(i)-t(i-1)) .le. tol) then
c              drop t(i) and move everything below it up
               if (i .ne. lent) then
                  do 140 j = i, itend
                     t(j) = t(j+1)
  140             continue 
               endif
               itend = itend - 1
            endif
  160    continue
         lent = itend
      endif
c
      return
      end
C*DECK DKSORT
      SUBROUTINE DKSORT (DX, DY, N, KFLAG, IFLAG)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  DKSORT
C***PURPOSE  Sort an array and optionally make the same interchanges in
C            an auxiliary array.  The array may be sorted in increasing
C            or decreasing order.  A slightly modified QUICKSORT
C            algorithm is used.
C***LIBRARY   SLATEC
C***CATEGORY  N6A2B
C***TYPE      DOUBLE PRECISION (SSORT-S, DKSORT-D, ISORT-I)
C***KEYWORDS  SINGLETON QUICKSORT, SORT, SORTING
C***AUTHOR  Jones, R. E., (SNLA)
C           Wisniewski, J. A., (SNLA)
C***DESCRIPTION
c
c   Note:
c   This subroutine should never be called directly
c   by the user.
C
C   DKSORT sorts array DX and optionally makes the same interchanges in
C   array DY.  The array DX may be sorted in increasing order or
C   decreasing order.  A slightly modified quicksort algorithm is used.
C
C   Description of Parameters
C      DX - array of values to be sorted   (usually abscissas)
C      DY - array to be (optionally) carried along
C      N  - number of values in array DX to be sorted
C      KFLAG - control parameter
C            =  2  means sort DX in increasing order and carry DY along.
C            =  1  means sort DX in increasing order (ignoring DY)
C            = -1  means sort DX in decreasing order (ignoring DY)
C            = -2  means sort DX in decreasing order and carry DY along.
C
C***REFERENCES  R. C. Singleton, Algorithm 347, An efficient algorithm
C                 for sorting with minimal storage, Communications of
C                 the ACM, 12, 3 (1969), pp. 185-187.
C***ROUTINES CALLED  NONE
C***END PROLOGUE  DKSORT
C     .. Scalar Arguments ..
      INTEGER KFLAG, N, lout
C     .. Array Arguments ..
      DOUBLE PRECISION DX(*), DY(*)
C     .. Local Scalars ..
      DOUBLE PRECISION R, T, TT, TTY, TY
      INTEGER I, IJ, J, K, KK, L, M, NN, IFLAG
C     .. Local Arrays ..
      INTEGER IL(21), IU(21)
C     .. Intrinsic Functions ..
      INTRINSIC ABS, INT
C***FIRST EXECUTABLE STATEMENT  DKSORT
      IFLAG = 0
      NN = N
      IF (NN .LT. 1) THEN
         IFLAG = 6
         CALL D6MESG(51,'DKSORT')
         RETURN
      ENDIF
C
      KK = ABS(KFLAG)
      IF (KK.NE.1 .AND. KK.NE.2) THEN
         IFLAG = 7
         CALL D6MESG(52,'DKSORT')
         RETURN
      ENDIF
C
C     Alter array DX to get decreasing order if needed
C
      IF (KFLAG .LE. -1) THEN
         DO 10 I=1,NN
            DX(I) = -DX(I)
   10    CONTINUE
      ENDIF
C
      IF (KK .EQ. 2) GO TO 100
C
C     Sort DX only
C
      M = 1
      I = 1
      J = NN
      R = 0.375D0
C
   20 IF (I .EQ. J) GO TO 60
      IF (R .LE. 0.5898437D0) THEN
         R = R+3.90625D-2
      ELSE
         R = R-0.21875D0
      ENDIF
C
   30 K = I
C
C     Select a central element of the array and save it in location T
C
      IJ = I + INT((J-I)*R)
      T = DX(IJ)
C
C     If first element of array is greater than T, interchange with T
C
      IF (DX(I) .GT. T) THEN
         DX(IJ) = DX(I)
         DX(I) = T
         T = DX(IJ)
      ENDIF
      L = J
C
C     If last element of array is less than than T, interchange with T
C
      IF (DX(J) .LT. T) THEN
         DX(IJ) = DX(J)
         DX(J) = T
         T = DX(IJ)
C
C        If first element of array is greater than T, interchange with T
C
         IF (DX(I) .GT. T) THEN
            DX(IJ) = DX(I)
            DX(I) = T
            T = DX(IJ)
         ENDIF
      ENDIF
C
C     Find an element in the second half of the array which is smaller
C     than T
C
   40 L = L-1
      IF (DX(L) .GT. T) GO TO 40
C
C     Find an element in the first half of the array which is greater
C     than T
C
   50 K = K+1
      IF (DX(K) .LT. T) GO TO 50
C
C     Interchange these elements
C
      IF (K .LE. L) THEN
         TT = DX(L)
         DX(L) = DX(K)
         DX(K) = TT
         GO TO 40
      ENDIF
C
C     Save upper and lower subscripts of the array yet to be sorted
C
      IF (L-I .GT. J-K) THEN
         IL(M) = I
         IU(M) = L
         I = K
         M = M+1
      ELSE
         IL(M) = K
         IU(M) = J
         J = L
         M = M+1
      ENDIF
      GO TO 70
C
C     Begin again on another portion of the unsorted array
C
   60 M = M-1
      IF (M .EQ. 0) GO TO 190
      I = IL(M)
      J = IU(M)
C
   70 IF (J-I .GE. 1) GO TO 30
      IF (I .EQ. 1) GO TO 20
      I = I-1
C
   80 I = I+1
      IF (I .EQ. J) GO TO 60
      T = DX(I+1)
      IF (DX(I) .LE. T) GO TO 80
      K = I
C
   90 DX(K+1) = DX(K)
      K = K-1
      IF (T .LT. DX(K)) GO TO 90
      DX(K+1) = T
      GO TO 80
C
C     Sort DX and carry DY along
C
  100 M = 1
      I = 1
      J = NN
      R = 0.375D0
C
  110 IF (I .EQ. J) GO TO 150
      IF (R .LE. 0.5898437D0) THEN
         R = R+3.90625D-2
      ELSE
         R = R-0.21875D0
      ENDIF
C
  120 K = I
C
C     Select a central element of the array and save it in location T
C
      IJ = I + INT((J-I)*R)
      T = DX(IJ)
      TY = DY(IJ)
C
C     If first element of array is greater than T, interchange with T
C
      IF (DX(I) .GT. T) THEN
         DX(IJ) = DX(I)
         DX(I) = T
         T = DX(IJ)
         DY(IJ) = DY(I)
         DY(I) = TY
         TY = DY(IJ)
      ENDIF
      L = J
C
C     If last element of array is less than T, interchange with T
C
      IF (DX(J) .LT. T) THEN
         DX(IJ) = DX(J)
         DX(J) = T
         T = DX(IJ)
         DY(IJ) = DY(J)
         DY(J) = TY
         TY = DY(IJ)
C
C        If first element of array is greater than T, interchange with T
C
         IF (DX(I) .GT. T) THEN
            DX(IJ) = DX(I)
            DX(I) = T
            T = DX(IJ)
            DY(IJ) = DY(I)
            DY(I) = TY
            TY = DY(IJ)
         ENDIF
      ENDIF
C
C     Find an element in the second half of the array which is smaller
C     than T
C
  130 L = L-1
      IF (DX(L) .GT. T) GO TO 130
C
C     Find an element in the first half of the array which is greater
C     than T
C
  140 K = K+1
      IF (DX(K) .LT. T) GO TO 140
C
C     Interchange these elements
C
      IF (K .LE. L) THEN
         TT = DX(L)
         DX(L) = DX(K)
         DX(K) = TT
         TTY = DY(L)
         DY(L) = DY(K)
         DY(K) = TTY
         GO TO 130
      ENDIF
C
C     Save upper and lower subscripts of the array yet to be sorted
C
      IF (L-I .GT. J-K) THEN
         IL(M) = I
         IU(M) = L
         I = K
         M = M+1
      ELSE
         IL(M) = K
         IU(M) = J
         J = L
         M = M+1
      ENDIF
      GO TO 160
C
C     Begin again on another portion of the unsorted array
C
  150 M = M-1
      IF (M .EQ. 0) GO TO 190
      I = IL(M)
      J = IU(M)
C
  160 IF (J-I .GE. 1) GO TO 120
      IF (I .EQ. 1) GO TO 110
      I = I-1
C
  170 I = I+1
      IF (I .EQ. J) GO TO 150
      T = DX(I+1)
      TY = DY(I+1)
      IF (DX(I) .LE. T) GO TO 170
      K = I
C
  180 DX(K+1) = DX(K)
      DY(K+1) = DY(K)
      K = K-1
      IF (T .LT. DX(K)) GO TO 180
      DX(K+1) = T
      DY(K+1) = TY
      GO TO 170
C
C     Clean up
C
  190 IF (KFLAG .LE. -1) THEN
         DO 200 I=1,NN
            DX(I) = -DX(I)
  200    CONTINUE
      ENDIF
C
      RETURN
      END
C*DECK D6GRT5
      SUBROUTINE D6GRT5(N,TOLD,YOLD,YNEW,DYNEW,TROOT1,YROOT,DYROOT,TNEW,
     +                  NG,NGUSER,GRESID,GUSER,JROOT,HINTP,INDEX1,IER,
     +                  NGEVAL,GRTOL,K0,K2,K3,K4,K5,K6,K7,K8,K9,GTOLD,
     +                  GTNEW,GROOT,GA,GB,TG,INDEXG,BETA,BVAL,TINTP,
     +                  IORDER,U10,IPAIR,RPAR,IPAR,TROOT2)
c
c     Note:
c     This subroutine should never be called directly
c     by the user.
C
C***BEGIN PROLOGUE  D6GRT5
C***SUBSIDIARY
C***PURPOSE  The purpose of D6GRT5 is to direct the rootfinding for
C            the DKLAG6 differential equation solver. It is called
C            by subroutine D6DRV3 to perform interpolatory
C            rootfinding.
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
C***ROUTINES CALLED  DCOPY, D6POLY, D6GRT2, D6MESG, D6RTOL
C***REVISION HISTORY  (YYMMDD)
C   960201  DATE WRITTEN
C***END PROLOGUE  D6GRT5
C
C     INTERFACE CALLED BY D6DRV3 TO DIRECT THE D6GRT2 ROOTFINDER.
C
C     IN THIS VERSION TWO ROOTFINDING PASSES ARE MADE. THE FIRST
C     SEARCHES FOR ROOTS IN (TOLD,TNEW). IF A ROOT TROOT1 IS
C     FOUND, A SECOND PASS SEARCHES FOR ROOTS IN (TROOT1,TNEW).
C     IF A SECOND ROOT TROOT2 IS FOUND, THE STEPSIZE IS LIMITED
C     BY TROOT2 - TROOT1 WHEN THE INTEGRATION IS RESTARTED TO
C     AVOID SKIPPING TROOT2 WHEN THE INTEGRATION IS CONTINUED.
C
      IMPLICIT NONE
C
      INTEGER I,IDUM1,IDUM2,IER,INDEX,INDEXG,IORDER,JFLAG,JROOT,N,NG,
     +        NGEMAX,NGEVAL,NGUSER,IPAR,NSIG,ICOMP,IPAIR,INDEX1,
     +        IRCALL
      DOUBLE PRECISION BNEW,BOLD,BVAL,DYNEW,DYROOT,GA,GB,GROOT,GRTOL,
     +                 GTNEW,GTOL,GTOLD,HDUM1,HDUM2,HINTP,K0,K2,K3,
     +                 K4,K5,K6,K7,K8,K9,TG,TINTP,TNEW,TOLD,TROOT,
     +                 U10,YNEW,YOLD,YROOT,RPAR,TROOT1,TROOT2
C
      DIMENSION YOLD(*),YNEW(*),DYNEW(*),YROOT(*),DYROOT(*),JROOT(*),
     +          K0(*),K2(*),K3(*),K4(*),K5(*),K6(*),K7(*),K8(*),K9(*),
     +          BVAL(*),GTOLD(*),GTNEW(*),GROOT(*),GA(*),GB(*),TG(*),
     +          TROOT(2),INDEXG(*),RPAR(*),IPAR(*),ICOMP(1),INDEX(2)
C
      EXTERNAL GRESID,GUSER,BETA
C
C***FIRST EXECUTABLE STATEMENT  D6GRT5
C
      INDEX(1) = 2
      INDEX(2) = 2
      TROOT(1) = TOLD
      TROOT(2) = TOLD
      IER = 0
      NSIG = 0
      NGEMAX = 500
      CALL D6RTOL(NSIG,NGEMAX,U10)
C
      DO 1000 IRCALL = 1 , 2
C
      IF (IRCALL .EQ. 2) THEN
C        THE SECOND PASS IS MADE ONLY IF THE FIRST
C        PASS FOUND A ROOT WITH INDEX(1) = 3.
         IF (INDEX(1) .NE. 3) GO TO 2000
      ENDIF
C
C     INITIALIZE THE ROOTFINDER.
C
      IF (IRCALL .EQ. 1) THEN
         BOLD = TOLD
         BNEW = TNEW
      ELSE
         BOLD = TROOT(1)
         BNEW = TNEW
         DO 10 I = 1,NG
            JROOT(I) = 0
   10    CONTINUE
      ENDIF
      JFLAG = 0
      GTOL = GRTOL*MAX(ABS(TOLD),ABS(TNEW))
      GTOL = MAX(GTOL,U10)
C
C     LOAD THE RESIDUALS AT THE ENDPOINTS.
C
      IF (IRCALL .EQ. 1) THEN
         CALL DCOPY(NG,GTOLD,1,GA,1)
         CALL DCOPY(NG,GTNEW,1,GB,1)
      ELSE
         CALL DCOPY(NG,GROOT,1,GA,1)
         CALL DCOPY(NG,GTNEW,1,GB,1)
      ENDIF
C
C     CALL THE ROOTFINDER.
C
   20 CONTINUE
C
      CALL D6GRT2(NG,GTOL,JFLAG,BOLD,BNEW,GA,GB,GROOT,TROOT(IRCALL),
     +            JROOT,HDUM1,HDUM2,IDUM1,IDUM2,IRCALL)
C
C     FOLLOW THE INSTRUCTIONS FROM THE ROOTFINDER.
C
      GO TO (30,40,60,80),JFLAG
C
   30 CONTINUE
C
C     EVALUATE THE RESIDUALS FOR THE ROOTFINDER.
C
C     FIRST APPROXIMATE THE SOLUTION AND THE DERIVATIVE ....
C
      CALL D6POLY(N,TINTP,YOLD,HINTP,K0,K2,K3,K4,K5,K6,K7,K8,K9,
     +            YROOT,DYROOT,TROOT(IRCALL),IORDER,ICOMP,IPAIR,
     +            3,0,0.0D0,0.0D0,0,11)
C
C     NOW CALCULATE THE RESIDUALS ....
C
      CALL GRESID(N,TROOT(IRCALL),YROOT,DYROOT,NG,NGUSER,
     +            GUSER,GROOT,BETA,BVAL,INDEXG,TG,RPAR,IPAR)
      IF (N.LE.0) GO TO 75
C
C     AND UPDATE THE RESIDUAL COUNTER ....
C
      NGEVAL = NGEVAL + 1
C
C     CHECK FOR TOO MANY RESIDUAL EVALUATIONS FOR THIS
C     CALL TO D6GRT5.
C
      IF (NGEVAL.GT.NGEMAX) THEN
          INDEX(1) = -10
          INDEX(2) = -10
          GO TO 90
      ENDIF
C
C     CALL THE ROOTFINDER AGAIN.
C
      GO TO 20
C
   40 CONTINUE
C
C     A ROOT WAS FOUND.
C
C     APPROXIMATE THE SOLUTION AND THE DERIVATIVE
C     AT THE ROOT.
C
      IF (IRCALL .EQ. 1) THEN
         CALL D6POLY(N,TINTP,YOLD,HINTP,K0,K2,K3,K4,K5,K6,
     +               K7,K8,K9,YROOT,DYROOT,TROOT(1),IORDER,
     +               ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,12)
      ENDIF
C
C     SET THE FLAG TO INDICATE A ROOT FOUND.
C
      INDEX(IRCALL) = 3
C
C     LOAD AN EXACT ZERO FOR THE RESIDUAL.
C
      IF (IRCALL .EQ. 1) THEN
         DO 50 I = 1,NG
            IF (JROOT(I).EQ.1) GROOT(I) = 0.0D0
   50    CONTINUE
      ENDIF
C
C     RESTORE THE SOLUTION FOR THE FIRST ROOT
C
      IF (IRCALL .EQ. 2) THEN
         CALL D6POLY(N,TINTP,YOLD,HINTP,K0,K2,K3,K4,K5,K6,
     +               K7,K8,K9,YROOT,DYROOT,TROOT(1),IORDER,
     +               ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,12)
      ENDIF
C
C     RETURN TO D6DRV3.
C
      GO TO 90
C
   60 CONTINUE
C
C     THE RIGHT ENDPOINT IS A ROOT.
C
      IF (IRCALL .EQ. 2) THEN
         WRITE(6,111)
  111    FORMAT(' There must be a bug in D6GRT5. We should',
     +          ' not be here.')
      ENDIF
C
C     LOAD THE SOLUTION AND DERIVATIVE
C     INTO YROOT AND DYROOT.
C
      CALL DCOPY(N,YNEW,1,YROOT,1)
      CALL DCOPY(N,DYNEW,1,DYROOT,1)
C
C     SET THE FLAG TO INDICATE A ROOT WAS FOUND.
C
      INDEX(IRCALL) = 4
C
C     LOAD AN EXACT ZERO FOR THE RESIDUAL.
C
      DO 70 I = 1,NG
          IF (JROOT(I).EQ.1) GROOT(I) = 0.0D0
   70 CONTINUE
C
C     RETURN TO D6DRV3.
C
      GO TO 90
C
C     THE USER TERMINATED THE INTEGRATION.
C
   75 CONTINUE
      INDEX(1) = -14
      INDEX(2) = -14
      CALL D6MESG(25,'D6GRT5')
      RETURN
C
   80 CONTINUE
C
C     NO SIGN CHANGES WERE DETECTED.
C
C     SET THE FLAG TO INDICATE NO SIGN CHANGES WERE DETECTED.
C
      INDEX(IRCALL) = 2
C
C     RESTORE THE SOLUTION FOR THE FIRST ROOT
C
      IF (IRCALL .EQ. 2) THEN
         CALL D6POLY(N,TINTP,YOLD,HINTP,K0,K2,K3,K4,K5,K6,
     +               K7,K8,K9,YROOT,DYROOT,TROOT(1),IORDER,
     +               ICOMP,IPAIR,3,0,0.0D0,0.0D0,0,12)
      ENDIF
C
C     RETURN TO D6DRV3.
C
   90 CONTINUE
C
 1000 CONTINUE
 2000 CONTINUE
C
      TROOT1 = TROOT(1)
      TROOT2 = TROOT(2)
      IF (INDEX(2) .NE. 3) TROOT2 = TROOT1
      INDEX1 = INDEX(1)
C
      RETURN
      END
C*DECK DRKS56
      SUBROUTINE DRKS56(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,
     +                  YINIT,DYINIT,N,W,LW,IW,LIW,QUEUE,LQUEUE,
     +                  IFLAG,LIN,LOUT,RPAR,IPAR,CONLAG)
C
C***BEGIN PROLOGUE  DRKS56
C***PURPOSE  This driver calls DKLAG6 if CONLAG = .FALSE. and
C            calls DKLG6C if CONLAG = .TRUE. DKLAG6 should be
C            for problems with state dependent delays. DKLG6C
C            should be used if all delays are constant.
C            Refer to the documentation prologue for DKLAG6 for
C            complete descriptions of the different parameters.
C
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
C     Refer to the documentation prologue for DKLAG6 for
C     complete descriptions of the different parameters.
C     Usage of DKLG6C is the same as that for DKLAG6.
C
C
      IMPLICIT NONE
C
      INTEGER LIW,LW,LQUEUE,IFLAG,LIN,LOUT,IW,J4SAVE,IPAR,
     +        INEED,J1,J2,J3,J4,J5,J6,J7,J8,J9,J10,J11,J12,
     +        J13,J14,K1,K2,K3,K4,LWORK,LIWORK,N,IDUM,LQ,
     +        NTFINS
      DOUBLE PRECISION W,QUEUE,D6OUTP,RPAR,RDUM
      LOGICAL CONLAG
C
      DIMENSION IW(LIW),IPAR(*)
      DIMENSION W(LW),QUEUE(LQUEUE),RPAR(*)
C
C     DECLARE THE OUTPUT INCREMENT FUNCTION.
C
      EXTERNAL D6OUTP
C
C     DECLARE THE USER SUPPLIED PROBLEM DEPENDENT SUBROUTINES.
C
      EXTERNAL INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,DYINIT
C
C     LWORK IS THE LENGTH OF THE WORK ARRAY (FOR D6DRV2; NEED
C     13*N MORE IF GO THROUGH D6KLAG).
C     LWORK MUST BE AT LEAST 21*N + 1.
C
C     LQUEUE IS THE LENGTH OF THE QUEUE HISTORY ARRAY.
C     LQUEUE SHOULD BE APPROXIMATELY
C         (12*(N+1)) * (TFINAL-TINIT)/HAVG ,
C     WHERE TINIT AND TFINAL ARE THE BEGINNING AND FINAL
C     INTEGRATION TIMES AND HAVG IS THE AVERAGE STEPSIZE
C     THAT WILL BE USED BY THE INTEGRATOR.
C
C***FIRST EXECUTABLE STATEMENT  DRKS56
C
C     RESET THE OUTPUT UNIT NUMBER TO THE ONE SPECIFIED
C     BY THE USER.
C
      IDUM = J4SAVE(3,LOUT,.TRUE.)
C
      CALL D6STAT(N, 1, 0.0D0, RPAR, IPAR, IDUM, RDUM)
C
C     THE DW ARRAY IS PARTITIONED HERE AND IN D6PNTR (FOR
C     D6DRV2) AS FOLLOWS:
C
C     IN D6KLAG:
C                                      STARTS IN DW:
C
C     YOLD       : J1  = 1              = 1
C     DYOLD      : J2  = J1  + N        = 1 +    N
C     YNEW       : J3  = J2  + N        = 1 +  2*N
C     DYNEW      : J4  = J3  + N        = 1 +  3*N
C     YROOT      : J5  = J4  + N        = 1 +  4*N
C     DYROOT     : J6  = J5  + N        = 1 +  5*N
C     YOUT       : J7  = J6  + N        = 1 +  6*N
C     DYOUT      : J8  = J7  + N        = 1 +  7*N
C     YLAG       : J9  = J8  + N        = 1 +  8*N
C     DYLAG      : J10 = J9  + N        = 1 +  9*N
C     BVAL       : J11 = J10 + N        = 1 + 10*N
C     ABSERR     : J12 = J11 + N        = 1 + 11*N
C     RELERR     : J13 = J12 + N        = 1 + 12*N
C     WORK       : J14 = J13 + N        = 1 + 13*N
C
C     IN D6DRV2 WORK(1) = DW(J14) = DW(1+13*N):
C
C                  STARTS IN WORK:     STARTS IN DW:
C
C     K0 , KPRED0: I1    = 1            = 1 + 13*N
C     K1 , KPRED1: I2    = I1  + 2*N    = 1 + 15*N
C     K2 , KPRED2: I3    = I2  + 2*N    = 1 + 17*N
C     K3 , KPRED3: I4    = I3  + 2*N    = 1 + 19*N
C     K4 , KPRED4: I5    = I4  + 2*N    = 1 + 21*N
C     K5 , KPRED5: I6    = I5  + 2*N    = 1 + 23*N
C     K6 , KPRED6: I7    = I6  + 2*N    = 1 + 25*N
C     K7 , KPRED7: IK7   = I7  + 2*N    = 1 + 27*N
C     K8 , KPRED8: IK8   = IK7 + 2*N    = 1 + 29*N
C     K9 , KPRED9: IK9   = IK8 + 2*N    = 1 + 31*N
C     R          : I8    = IK9 + 2*N    = 1 + 33*N
C     TINIT      : I9-1  = I8  + N      = 1 + 34*N
C     G(TOLD)    : I9    = I8  + N + 1  = 2 + 34*N
C     G(TNEW)    : I10   = I9  + NG     = 2 + 34*N +   NG
C     G(TROOT)   : I11   = I10 + NG     = 2 + 34*N + 2*NG
C     GA         : I12   = I11 + NG     = 2 + 34*N + 3*NG
C     GB         : I13   = I12 + NG     = 2 + 34*N + 4*NG
C     G2(TROOT)  : I14   = I13 + NG     = 2 + 34*N + 5*NG
C     TG         : I15   = I14 + NG     = 2 + 34*N + 6*NG
C     TG ENDS IN :                        1 + 34*N + 7*NG
C
C     DEFINE THE LOCAL ARRAY POINTERS INTO THE WORK ARRAY.
C
C     J1   -  YOLD
C     J2   -  DYOLD
C     J3   -  YNEW
C     J4   -  DYNEW
C     J5   -  YROOT
C     J6   -  DYROOT
C     J7   -  YOUT
C     J8   -  DYOUT
C     J9   -  YLAG
C     J10  -  DYLAG
C     J11  -  BVAL
C     J12  -  ABSERR
C     J13  -  RELERR
C     J14  -  WORK
C
C     IF SOME DELAY IS NOT CONSTANT:
C
      IF (.NOT. CONLAG) THEN
C
      J1 = 1
      J2 = J1 + N
      J3 = J2 + N
      J4 = J3 + N
      J5 = J4 + N
      J6 = J5 + N
      J7 = J6 + N
      J8 = J7 + N
      J9 = J8 + N
      J10 = J9 + N
      J11 = J10 + N
      J12 = J11 + N
      J13 = J12 + N
      J14 = J13 + N
      LWORK = LW - (J14-1)
C
      K1 = 1
      K2 = K1 + 34 
      LIWORK = LIW - 34 
C     THE REMAINING INTEGER STORAGE WILL BE SPLIT EVENLY
C     BETWEEN THE JROOT AND INDEXG ARRAYS.
      LIWORK = LIWORK/3
      K3 = K2 + LIWORK
      K4 = K3 + LIWORK
C
C     PRELIMINARY CHECK FOR ILLEGAL INPUT ERRORS.
C
      IFLAG = 0
C
      IF (N.LT.1) THEN
          CALL D6MESG(3,'DRKS56')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      INEED = 12*(N+1)
      IF (INEED.GT.LQUEUE) THEN
          CALL D6MESG(20,'DRKS56')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LIWORK.LT.0 .OR. LIW .LT. 35) THEN
          CALL D6MESG(21,'DRKS56')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LWORK.LT.21*N+1) THEN
          CALL D6MESG(9,'DRKS56')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      CALL D6DRV2(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,
     +            DYINIT,N,W(J1),W(J2),W(J3),W(J4),W(J5),W(J6),
     +            W(J7),W(J8),W(J9),W(J10),W(J11),W(J12),W(J13),
     +            W(J14),QUEUE,IW(K1),IW(K2),IW(K3),IW(K4),LWORK,
     +            LIWORK,LQUEUE,IFLAG,D6OUTP,LIN,LOUT,RPAR,IPAR)
C
      RETURN
C
C     IF ALL DELAYS ARE CONSTANT:
C
      ELSE
C
      J1 = 1
      J2 = J1 + N
      J3 = J2 + N
      J4 = J3 + N
      J5 = J4 + N
      J6 = J5 + N
      J7 = J6 + N
      J8 = J7 + N
      J9 = J8 + N
      J10 = J9 + N
      J11 = J10 + N
      J12 = J11 + N
      J13 = J12 + N
      J14 = J13 + N
      LWORK = LW - (J14-1)
C
C     TELL D6DRV4 THAT THE HISTORY QUEUE AND THE TFINS ARRAY
C     ARE THE SAME SIZE AT THIS POINT AND PASS THE QUEUE
C     ARRAY FOR EACH.
      NTFINS = LQUEUE
      LQ = LQUEUE
C
      K1 = 1
      K2 = K1 + 34 
      LIWORK = LIW - 34 
C     THE REMAINING INTEGER STORAGE WILL BE SPLIT EVENLY
C     BETWEEN THE JROOT AND INDEXG ARRAYS.
      LIWORK = LIWORK/3
      K3 = K2 + LIWORK
      K4 = K3 + LIWORK
C
C     PRELIMINARY CHECK FOR ILLEGAL INPUT ERRORS.
C
      IF (N.LT.1) THEN
          CALL D6MESG(3,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      INEED = 12*(N+1)
      IF (INEED.GT.LQ) THEN
          CALL D6MESG(20,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LIWORK.LT.0 .OR. LIW .LT. 35) THEN
          CALL D6MESG(21,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      IF (LWORK.LT.21*N+1) THEN
          CALL D6MESG(9,'DKLG6C')
          IFLAG = 1
          GO TO 9005
      ENDIF
C
      CALL D6DRV4(INITAL,DERIVS,GUSER,CHANGE,DPRINT,BETA,YINIT,
     +            DYINIT,N,W(J1),W(J2),W(J3),W(J4),W(J5),W(J6),
     +            W(J7),W(J8),W(J9),W(J10),W(J11),W(J12),W(J13),
     +            W(J14),QUEUE,IW(K1),IW(K2),IW(K3),IW(K4),LWORK,
     +            LIWORK,LQ,IFLAG,D6OUTP,LIN,LOUT,RPAR,IPAR,
     +            QUEUE,NTFINS)
C
      ENDIF
C
 9005 CONTINUE
C
      RETURN
      END
