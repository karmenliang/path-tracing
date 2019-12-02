# Build sequential path tracer
trace-seq: trace.cpp
	g++ -Wall -g -o trace-seq trace.cpp -ljpeg

# Run and output to a ppm file
out-seq.ppm: trace-seq
	@rm -f out-seq.ppm
	./trace-seq > out-seq.ppm

# Transform ppm to jpg file
out-seq.jpg: out-seq.ppm
	@rm -f out-seq.jpg
	ppmtojpeg out-seq.ppm > out-seq.jpg

clean:
	@rm -f trace-seq *.o *~
	@echo Made clean.
