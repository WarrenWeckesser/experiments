      PROGRAM ARGTEST
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
C     VALUES holds: r, h, stoptime, abserr, relerr
C
      CHARACTER*10 SYMBOLS(5)
      DIMENSION DEFVALUES(5)
      DIMENSION VALUES(5)
C
C     Symbols
      SYMBOLS(1) = 'alpha'
      SYMBOLS(2) = 'h'
      SYMBOLS(3) = 'stoptime'
      SYMBOLS(4) = 'abserr'
      SYMBOLS(5) = 'relerr'
C     Default values
      alpha = 0.5
      h = 4.0
      stoptime = 50.0
      abserr   = 1.0D-12
      relerr   = 1.0D-10
      DEFVALUES(1) = alpha
      DEFVALUES(2) = h
      DEFVALUES(3) = stoptime
      DEFVALUES(4) = abserr
      DEFVALUES(5) = relerr
C
      CALL GETARGS(5,SYMBOLS,DEFVALUES,VALUES)
C
      DO 50 I = 1, 5
          PRINT *, SYMBOLS(I),'=',VALUES(I)
50    CONTINUE
      STOP
      END
C
C
      SUBROUTINE GETARGS(N,SYMBOLS,DEFVALUES,VALUES)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      CHARACTER*10 SYMBOLS(5)
      DIMENSION DEFVALUES(5)
      DIMENSION VALUES(5)
C
      CHARACTER*10 ARGSTRING
C
C     Copy the default values to VALUES
C
      DO 100 I = 1,5
          VALUES(I) = DEFVALUES(I)
100   CONTINUE
C
C     Get the command line arguments.
C
      NARG = IARGC()
      IF (2*(NARG/2) .NE. NARG) THEN
          WRITE(*,*) ' Invalid number of arguments.'
          CALL USE(N,SYMBOLS,DEFVALUES)
          STOP
      ENDIF
      M = NARG/2
      DO 200 I = 1, M
          CALL GETARG(2*I,ARGSTRING)
          READ(ARGSTRING,*) ARGVALUE
          CALL GETARG(2*I-1,ARGSTRING)
          DO 250 J = 1, N
              IF (ARGSTRING .EQ. SYMBOLS(J)) THEN
                  VALUES(J) = ARGVALUE
                  GOTO 260
              ENDIF
250       CONTINUE
          PRINT *, 'Invalid argument: ', ARGSTRING
          CALL USE(N,SYMBOLS,DEFVALUES)
          STOP
260       CONTINUE
C
200   CONTINUE
C
      END
C
C
      SUBROUTINE USE(N,SYMBOLS,DEFVALUES)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
C
      DIMENSION DEFVALUES(N)
      CHARACTER*10 SYMBOLS(N)
C
      PRINT *, 'Use: argtest [symbol value]*'
      PRINT *, 'Allowed symbols and their default values:'
      DO 100 I=1,N
          PRINT *, SYMBOLS(I), DEFVALUES(I)
100   CONTINUE
C
      RETURN
      END
