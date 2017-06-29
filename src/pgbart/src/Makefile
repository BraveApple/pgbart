PGABRT_HEADER_DIR = ../include
PROJECT_SRC_DIR = ../..

EXTERNAL_HEADER_DIR = $(PROJECT_SRC_DIR)/external/include

test:
	echo $(PGABRT_HEADER_DIR)
#CXX_STD = C++0x
#CXX_PG = $(CXX1X) $(CXX1XSTD)
#CXXFLAGS = $(CXX1XFLAGS) $(CXX1XPICFLAGS)
PKG_CPPFLAGS = -I $(PROJECT_SRC_DIR)

ALL_CPPFLAGS = $(R_XTRA_CPPFLAGS) $(PKG_CPPFLAGS) $(CPPFLAGS)
#PKG_LIBS =$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS) -lpthread

LOCAL_SOURCES = bart.cpp config.cpp mcmc.cpp particle.cpp random.cpp state.cpp tree.cpp

LOCAL_OBJECTS = bart.o config.o mcmc.o particle.o random.o state.o tree.o

all : libpgbart.a

libpgbart.a : $(LOCAL_OBJECTS)
	$(AR) $(ARFLAGS) libpgbart.a $(LOCAL_OBJECTS)



$(PGABRT_HEADER_DIR)/bart.hpp : $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/control.hpp $(PGABRT_HEADER_DIR)/data.hpp $(PGABRT_HEADER_DIR)/mcmc.hpp

$(PGABRT_HEADER_DIR)/compare.hpp :

$(PGABRT_HEADER_DIR)/config.hpp :

$(PGABRT_HEADER_DIR)/control.hpp : $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/math.hpp

$(PGABRT_HEADER_DIR)/data.hpp : $(PGABRT_HEADER_DIR)/math.hpp

$(PGABRT_HEADER_DIR)/math.hpp : $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/compare.hpp

$(PGABRT_HEADER_DIR)/mcmc.hpp : $(PGABRT_HEADER_DIR)/data.hpp $(PGABRT_HEADER_DIR)/control.hpp $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/particle.hpp $(PGABRT_HEADER_DIR)/compare.hpp

$(PGABRT_HEADER_DIR)/node.hpp : $(PGABRT_HEADER_DIR)/config.hpp

$(PGABRT_HEADER_DIR)/particle.hpp : $(PGABRT_HEADER_DIR)/state.hpp

$(PGABRT_HEADER_DIR)/random.hpp : $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/math.hpp

$(PGABRT_HEADER_DIR)/serialize.hpp : $(PGABRT_HEADER_DIR)/particle.hpp $(PGABRT_HEADER_DIR)/config.hpp

$(PGABRT_HEADER_DIR)/state.hpp : $(PGABRT_HEADER_DIR)/tree.hpp $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/random.hpp $(PGABRT_HEADER_DIR)/math.hpp $(PGABRT_HEADER_DIR)/data.hpp $(PGABRT_HEADER_DIR)/control.hpp $(PGABRT_HEADER_DIR)/compare.hpp

$(PGABRT_HEADER_DIR)/tree.hpp : $(PGABRT_HEADER_DIR)/node.hpp $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/data.hpp

bart.o : bart.cpp $(PGABRT_HEADER_DIR)/bart.hpp
	$(CXX) $(ALL_CPPFLAGS) $(CXXFLAGS) -c bart.cpp -o bart.o

config.o : config.cpp $(PGABRT_HEADER_DIR)/config.hpp $(PGABRT_HEADER_DIR)/math.hpp
	$(CXX) $(ALL_CPPFLAGS) $(CXXFLAGS) -c config.cpp -o config.o

mcmc.o : mcmc.cpp $(PGABRT_HEADER_DIR)/mcmc.hpp
	$(CXX) $(ALL_CPPFLAGS) $(CXXFLAGS) -c mcmc.cpp -o mcmc.o

particle.o : particle.cpp $(PGABRT_HEADER_DIR)/particle.hpp
	$(CXX) $(ALL_CPPFLAGS) $(CXXFLAGS) -c particle.cpp -o particle.o

random.o : random.cpp $(PGABRT_HEADER_DIR)/random.hpp
	$(CXX) $(ALL_CPPFLAGS) $(CXXFLAGS) -c random.cpp -o random.o

state.o : state.cpp $(PGABRT_HEADER_DIR)/state.hpp $(EXTERNAL_HEADER_DIR)/mconf.h $(EXTERNAL_HEADER_DIR)/gdtr.h
	$(CXX) $(ALL_CPPFLAGS) $(CXXFLAGS) -c state.cpp -o state.o

tree.o : tree.cpp $(PGABRT_HEADER_DIR)/tree.hpp $(PGABRT_HEADER_DIR)/tree.hpp $(PGABRT_HEADER_DIR)/math.hpp
	$(CXX) $(ALL_CPPFLAGS) $(CXXFLAGS) -c tree.cpp -o tree.o

.PHONY: clean rebuild

clean :
	-rm -f *.o
	-rm -f *.a

rebuild : clean all
