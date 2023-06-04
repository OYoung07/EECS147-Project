.RECIPEPREFIX = >
.PHONY: all clean count

NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -rdc=true
else
NVCC_FLAGS  = -O3 --std=c++03 -rdc=true
endif
LD_FLAGS    = -lcudart
EXE         = simulator
OBJ         = main.o support.o body.o

default: $(EXE)

main.o: main.cu kernel.cu support.h body.h
>$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
>$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

body.o: body.cu body.h
>$(NVCC) -c -o $@ body.cu $(NVCC_FLAGS)

$(EXE):$(OBJ)
>$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)
>make clean

clean: 
>rm -rf *.o
