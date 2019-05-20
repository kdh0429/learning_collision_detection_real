NAME	= main
OBJS    = main.o 
CXX     = g++
DEBUG   = 
CXXFLAGS  = -Wall $(DEBUG) -std=c++11 -pthread

SRC_DIR = src


CXXFLAGS += -g -Wall -Wextra -pthread


all : $(NAME)


clean :
	rm -f *.o $(OBJS) $(NAME) 
	
run : 
	./main


EstimateCollisionNN.o : $(SRC_DIR)/EstimateCollisionNN.cpp $(SRC_DIR)/UtilFunction.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(SRC_DIR)/EstimateCollisionNN.cpp

$(NAME).o : $(SRC_DIR)/$(NAME).cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(SRC_DIR) -c $(SRC_DIR)/$(NAME).cpp

$(NAME) : $(OBJS) EstimateCollisionNN.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $^

