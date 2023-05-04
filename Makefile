C        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3
else
NVCC_FLAGS  = -O3 --std=c++03
endif
LD_FLAGS    = -lcudart
EXE         = reduction
OBJ         = main.o support.o

default: naive optimized


main-optimized.o: main.cu kernel.cu support.h
    $(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) -DOPTIMIZED

main.o: main.cu kernel.cu support.h
    $(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

upport.o: support.cu support.h
    $(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

naive: $(OBJ)
    $(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)


