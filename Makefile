OPENCV_PACKAGE                  :=opencv4
GENERATED_CODE_DIR		:=offline_optical_flow_CODE
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 			+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: optical_flow

optical_flow: simulator.cc generated_code
	$(CXX) $(CXXFLAGS) `pkg-config --cflags $(OPENCV_PACKAGE)` -I$(GENN_USERPROJECT_INCLUDE) simulator.cc -o optical_flow -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR) -pthread `pkg-config --libs $(OPENCV_PACKAGE)`

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
