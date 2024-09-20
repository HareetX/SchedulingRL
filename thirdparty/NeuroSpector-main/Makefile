# Directories
SRCDIR=src
OBJDIR=obj
SUBDIRS=$(dir $(wildcard $(SRCDIR)/*/.))

# Flags
CXX=g++
SUBFLAGS=$(addprefix -I, $(patsubst %/, %, $(SUBDIRS)))
CXXFLAGS=-g -Wall -O3 -std=c++11 -I$(SRCDIR) $(SUBFLAGS) -fPIC

# Get pybind11 and Python include paths
PYBIND11_INCLUDE=$(shell python3 -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

# Add pybind11 and Python to include paths
CXXFLAGS += -I$(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE)
CXXFLAGS += -fopenmp -DOPENMP

LDFLAGS += -shared
LDFLAGS += -fopenmp -lgomp
LIBFLAGS=-lpthread
PYTHON_LIBRARY = $(shell python3-config --ldflags)
LIBFLAGS += $(PYTHON_LIBRARY)

# Sources(/src)
SRCS=$(wildcard $(SRCDIR)/*.cc)
HDRS=$(wildcard $(SRCDIR)/*.h)
OBJS=$(SRCS:$(SRCDIR)/%.cc=$(OBJDIR)/%.o)
# Sources(/src/*)
SUBSRCS=$(wildcard $(SRCDIR)/*/*.cc)
SUBHDRS=$(wildcard $(SRCDIR)/*/*.h)
SUBOBJS=$(addprefix $(OBJDIR)/, $(notdir $(patsubst %.cc, %.o, $(SUBSRCS))))

# Library
LIB=libneurospector.so

# Targets
.PHONY: all clean
all: $(LIB)

$(LIB): $(SUBOBJS) $(OBJS)
	@echo "# Makefile Target: $@"
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc $(HDRS)
	@mkdir -pv $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.o: $(SRCDIR)/*/%.cc $(SUBHDRS)
	@mkdir -pv $(OBJDIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	@rm -f $(OBJS) $(SUBOBJS) $(LIB)
	@rm -rf $(OBJDIR)
	@echo "# Makefile Clean: $(OBJDIR)/'s [ $(notdir $(OBJS) $(SUBOBJS) ] and [ $(LIB)) ] are removed"
