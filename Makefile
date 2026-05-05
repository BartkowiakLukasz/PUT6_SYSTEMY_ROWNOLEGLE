CXX      = g++
CXXFLAGS = -O2 -fopenmp -std=c++17 -Wall -Wextra
LDFLAGS  = -lgmp -lgmpxx -fopenmp
TARGET   = automorphisms
SRC      = automorphisms.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
