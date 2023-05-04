NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3
else
NVCC_FLAGS  = -O3 --std=c++03
endif
LD_FLAGS    = -lcudart
EXE         = reduction
OBJ         = main.o support.o

default: body
# i'm not sure what the default should be listed as - but i think this was causing the issue

main.o: main.cu kernel.cu support.h
    $(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
    $(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

body.o: body.cu body.h
	$(NVCC) -c -o $@ body.cu $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE) $(EXE)-body
