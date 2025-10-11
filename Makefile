CXX = g++

CXXFLAGS =
TARGET_EXEC =

ifeq ($(mode), debug)
	CXXFLAGS = -Wall -O0 -g -march=native
	TARGET_EXEC = bin.debug
else
	CXXFLAGS = -Wall -O3 -march=native
	TARGET_EXEC = bin
endif


OBJS = utils.o solver.o integrator.o CosseratRod.o main.o

all: $(OBJS)
	$(CXX) -Wall $(OBJS) -o $(TARGET_EXEC)

utils.o: include/utils.hpp src/utils.cpp
	$(CXX) $(CXXFLAGS) -c include/utils.hpp src/utils.cpp
solver.o: utils.o include/solver.hpp src/solver.cpp
	$(CXX) $(CXXFLAGS) -c include/solver.hpp src/solver.cpp
integrator.o: utils.o include/integrator.hpp src/integrator.cpp
	$(CXX) $(CXXFLAGS) -c include/integrator.hpp src/integrator.cpp
CosseratRod.o: utils.o solver.o integrator.o include/CosseratRod.hpp src/CosseratRod.cpp
	$(CXX) $(CXXFLAGS) -c include/CosseratRod.hpp src/CosseratRod.cpp
main.o: utils.o solver.o integrator.o CosseratRod.o include/main.hpp src/main.cpp
	$(CXX) $(CXXFLAGS) -c include/main.hpp src/main.cpp

clean:
	rm -f include/*.gch $(OBJS)
