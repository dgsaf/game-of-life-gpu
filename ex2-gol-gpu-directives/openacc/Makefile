# here the make accepts the compiler, gpu, calctype
# will add a make print to indicate what types of compilations
# are enabled.
COMPILERTYPE ?= PGI
OPTLEVEL ?= 2
PROFILER ?= OFF

# optmisation flags
OPTFLAGS = -O$(OPTLEVEL)
# profiling flags if desired 
ifeq ($(PROFILER), ON)
	OPTFLAGS += -pg -g
endif

# formatting characters for info output
NULL := 
TAB := $(NULL)  $(NULL)

# lets define a bunch of compilers
CC = pgcc
FORT = pgf90

# openacc flags
OPENACCFLAGS = -acc -Minfo=accel

PREPROCESSFLAGS = -Mpreprocess
COMMONFLAGS = $(OPTFLAGS) $(VISUALFLAGS) 

.PHONY : dir cpu_serial gpu_openacc
.PHONY : gpu_openacc gpu_openacc_fort

all : dirs cpu_serial gpu_openacc 

dirs :
	[ -d obj ] || mkdir obj
	[ -d bin ] || mkdir bin

clean :
	rm obj/*
	rm bin/*

# just make an easier make name to remember
cpu_serial : bin/01_gol_cpu_serial bin/01_gol_cpu_serial_fort
# gpu related 
gpu_openacc : bin/02_gol_gpu_openacc bin/02_gol_gpu_openacc_fort 

obj/common.o : src/common.h src/common.c
	$(CC) $(PREPROCESSFLAGS) $(COMMONFLAGS) $(CFLAGS) -c src/common.c -o obj/common.o

obj/common_fort.o : src/common_fort.f90 
	$(FORT) $(PREPROCESSFLAGS) $(COMMONFLAGS) $(FFLAGS) -c src/common_fort.f90 -o obj/common_fort.o

bin/01_gol_cpu_serial : src/01_gol_cpu_serial.c obj/common.o
	$(CC) $(COMMONFLAGS) $(CFLAGS) -c src/01_gol_cpu_serial.c -o obj/01_gol_cpu_serial.o
	$(CC) $(COMMONFLAGS) $(CFLAGS) -o bin/01_gol_cpu_serial obj/01_gol_cpu_serial.o obj/common.o

bin/01_gol_cpu_serial_fort : src/01_gol_cpu_serial_fort.f90 obj/common_fort.o
	$(FORT) $(PREPROCESSFLAGS) $(COMMONFLAGS) $(FFLAGS) -c src/01_gol_cpu_serial_fort.f90 -o obj/01_gol_cpu_serial_fort.o
	$(FORT) $(COMMONFLAGS) $(FFLAGS) -o bin/01_gol_cpu_serial_fort obj/01_gol_cpu_serial_fort.o obj/common_fort.o

bin/02_gol_gpu_openacc: src/02_gol_gpu_openacc.c obj/common.o
	$(CC) $(COMMONFLAGS) $(OPENACCFLAGS) $(CFLAGS) -c src/02_gol_gpu_openacc.c -o obj/02_gol_gpu_openacc.o
	$(CC) $(COMMONFLAGS) $(OPENACCFLAGS) $(CFLAGS) -o bin/02_gol_gpu_openacc obj/02_gol_gpu_openacc.o obj/common.o 

bin/02_gol_gpu_openacc_fort: src/02_gol_gpu_openacc_fort.f90 obj/common_fort.o
	$(FORT) $(PREPROCESSFLAGS) $(COMMONFLAGS) $(OPENACCFLAGS) $(FFLAGS) -c src/02_gol_gpu_openacc_fort.f90 -o obj/02_gol_gpu_openacc_fort.o
	$(FORT) $(COMMONFLAGS) $(OPENACCFLAGS) $(FFLAGS) -o bin/02_gol_gpu_openacc_fort obj/02_gol_gpu_openacc_fort.o obj/common_fort.o
