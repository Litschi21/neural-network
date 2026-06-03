CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Wnull-dereference -Wformat=2 -Werror
TARGET = main

all:
	$(MAKE) clean && $(CXX) main.cpp matrix.cpp nn.cpp $(CXXFLAGS) -o $(TARGET).exe

speed:
	$(MAKE) clean && $(CXX) main.cpp matrix.cpp nn.cpp $(CXXFLAGS) -O3 -o $(TARGET).exe

tests:
	rm -f tests.exe && $(CXX) tests.cpp matrix.cpp nn.cpp -std=c++23 -lgtest -lgtest_main -o tests.exe

clean:
	rm -f $(TARGET).exe
