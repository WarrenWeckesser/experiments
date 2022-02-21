export CLASSPATH=.:/usr/share/java/commons-math3.jar
javac -d . RosslerODE.java
javac RosslerSolver.java
java RosslerSolver
