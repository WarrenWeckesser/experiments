DKLAG6_OBJ = dfalt6.o dklag6.o dkutil.o
FC = gfortran

sirdelay: sirdelay.o $(DKLAG6_OBJ)
	$(FC) -o sirdelay sirdelay.o $(DKLAG6_OBJ)

sirdelay2: sirdelay2.o $(DKLAG6_OBJ)
	$(FC) -o sirdelay2 sirdelay2.o $(DKLAG6_OBJ)

sirdelay.o: sirdelay.f
	$(FC) -c sirdelay.f

dfalt6.o: dfalt6.f
	$(FC) -c dfalt6.f

dklag6.o: dklag6.f
	$(FC) -c dklag6.f

dkutil.o: dkutil.f
	$(FC) -c dkutil.f

clean:
	rm -rf *.o sirdelay sirdelay2
