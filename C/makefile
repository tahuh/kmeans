CC = gcc
CFLAGS = -Wall -O3 -g -DDEBUG
CLIBS = -lm
OBJS = main.o kmeans.o
TARGET = kmeans_test

all : $(TARGET)

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(CLIBS)

clean :
	rm *.o kmeans_test
