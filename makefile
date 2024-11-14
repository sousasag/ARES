install:
	gcc -o ARES ARES_v2.c -lcfitsio -lgsl -lgslcblas -lm -lgomp -fopenmp

test:
	./ARES > /dev/null
	diff test.ares test.ares_oric

clean:
	rm -f ARES test.ares
