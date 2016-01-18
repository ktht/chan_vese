CXX        = g++
WFLAGS     = -Wall -Wextra -Wno-unused -Werror
CXXFLAGS  += $(WFLAGS) -std=c++14
LDFLAGS    = -flto -fuse-linker-plugin

ifdef DEBUG
  CXXFLAGS += -D_GLIBCXX_DEBUG -O0 -g3 -ggdb3
else
  CXXFLAGS += -O3 -g0 -ggdb0 -fno-unsafe-math-optimizations -fno-associative-math
  LDFLAGS  += -s
endif

SOURCE_PATH   = src
INCLUDE_PATH  = include
PYTHON_PATH   = python
OBJ_PATH      = obj
DEP_PATH      = dep
BIN_PATH      = bin
TEST_PATH     = test
TEST_BIN_PATH = $(BIN_PATH)/$(TEST_PATH)

INCLUDES = -I$(INCLUDE_PATH) \
           -I$(BOOST_PATH)/include
LD_PATHS = -L$(BOOST_PATH)/lib
LIBS     = -lopencv_core \
           -lopencv_imgproc \
           -lopencv_highgui

SRCS      = main.cpp
TEST      =
OBJS      = $(SRCS:%.cpp=$(OBJ_PATH)/%.o)
TEST_OBJS = $(TEST:%.cpp=$(OBJ_PATH)/%.o)
DEPS      = $(SRCS:%.cpp=$(DEP_PATH)/%.d)
TEST_DEPS = $(TEST:%.cpp=$(DEP_PATH)/%.d)
TRGT      = chan_vese
TEST_TRGT =

all: $(TRGT)

test: $(TEST_TRGT)

$(TRGT): %: $(OBJS)
	@mkdir -p $(BIN_PATH)
	@$(CXX) $(CXXFLAGS) $^ -o $(BIN_PATH)/$@ $(LD_PATHS) $(LDFLAGS) $(LIBS)

$(OBJ_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	@mkdir -p $(@D)
	@mkdir -p $(DEP_PATH)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MF $(patsubst $(OBJ_PATH)/%.o,$(DEP_PATH)/%.d,$@) -c $< -o $@

$(TEST_TRGT): $(TEST_OBJS)
	@mkdir -p $(TEST_BIN_PATH)
	@$(CXX) $(CXXFLAGS) $< -o $(TEST_BIN_PATH)/$@ $(LD_PATHS) $(LDFLAGS) $(LIBS)

$(OBJ_PATH)/%.o: $(TEST_PATH)/%.cpp
	@mkdir -p $(@D)
	@mkdir -p $(DEP_PATH)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MF $(patsubst $(OBJ_PATH)/%.o,$(DEP_PATH)/%.d,$@) -c $< -o $@

ifneq ($(MAKECMDGOALS),clean)
  ifeq ($(MAKECMDGOALS),test)
    -include $(TEST_DEPS)
  else
    -include $(DEPS)
  endif
endif

.PHONY: clean

clean:
	@rm -rf $(OBJ_PATH) $(BIN_PATH) $(DEP_PATH) $(PYTHON_PATH)/*.pyc
