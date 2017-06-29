EXTERNAL_HEADER_DIR = ../include
PROJECT_SRC_DIR = ../../

PKG_CPPFLAGS = -I $(PROJECT_SRC_DIR)
ALL_CPPFLAGS = $(R_XTRA_CPPFLAGS) $(PKG_CPPFLAGS) $(CPPFLAGS)

LOCAL_SOURCES = const.c gamma.c gdtr.c igam.c mtherr.c polevl.c
LOCAL_OBJECTS = const.o gamma.o gdtr.o igam.o mtherr.o polevl.o

all : libexternal.a

libexternal.a : $(LOCAL_OBJECTS)
	$(AR) $(ARFLAGS) libexternal.a $(LOCAL_OBJECTS)

$(EXTERNAL_HEADER_DIR)/gdtr.h :
$(EXTERNAL_HEADER_DIR)/mconf.h :

const.o : const.c $(EXTERNAL_HEADER_DIR)/mconf.h
	$(CC) $(ALL_CPPFLAGS) $(CFLAGS) -c const.c -o const.o

gamma.o : gamma.c $(EXTERNAL_HEADER_DIR)/mconf.h $(EXTERNAL_HEADER_DIR)/gdtr.h
	$(CC) $(ALL_CPPFLAGS) $(CFLAGS) -c gamma.c -o gamma.o

gdtr.o : gdtr.c $(EXTERNAL_HEADER_DIR)/mconf.h $(EXTERNAL_HEADER_DIR)/gdtr.h
	$(CC) $(ALL_CPPFLAGS) $(CFLAGS) -c gdtr.c -o gdtr.o

igam.o : igam.c $(EXTERNAL_HEADER_DIR)/mconf.h
	$(CC) $(ALL_CPPFLAGS) $(CFLAGS) -c igam.c -o igam.o

mtherr.o : mtherr.c $(EXTERNAL_HEADER_DIR)/mconf.h
	$(CC) $(ALL_CPPFLAGS) $(CFLAGS) -c mtherr.c -o mtherr.o

polevl.o : polevl.c
	$(CC) $(ALL_CPPFLAGS) $(CFLAGS) -c polevl.c -o polevl.o

.PHONY: clean rebuild

clean :
	-rm -f *.o
	-rm -f *.a

rebuild : clean all
