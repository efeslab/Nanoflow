GENERATED := gemmFactory.cu cutlassGemmExternDeclearation.cuh
all: $(GENERATED)
$(GENERATED): genGEMM.py gemmFactory.in
	rm -f *.cu *.cuh
	python3 genGEMM.py
