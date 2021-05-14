.PHONY : dirs

CC=gcc
COMMONFLAGS=#-DUSEPNG
INCLUDES=#-I/usr/include
LIBS=#-L/usr/lib64/ -lpng

all : dirs bin/game_of_life bin/game_of_life_cuda

dirs :
	[ -d obj ] || mkdir obj
	[ -d bin ] || mkdir bin

clean :
	rm obj/*
	rm bin/*

obj/common.o : src/common.h src/common.c
	cd obj; \
	$(CC) $(COMMONFLAGS) $(INCLUDES) -o common.o -c ../src/common.c $(LIBS)

bin/game_of_life : src/game_of_life.c obj/common.o
	$(CC) $(COMMONFLAGS) $(INCLUDES) -o obj/game_of_life.o -c src/game_of_life.c
	$(CC) $(COMMONFLAGS) $(INCLUDES) -o bin/game_of_life \
	obj/game_of_life.o obj/common.o $(LIBS)

bin/game_of_life_cuda : src/game_of_life_cuda.cu obj/common.o
	nvcc $(COMMONFLAGS) $(INCLUDES) -o bin/game_of_life_cuda \
	src/game_of_life_cuda.cu obj/common.o $(LIBS)
