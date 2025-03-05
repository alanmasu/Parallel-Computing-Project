# Nome dell'eseguibile principale
TARGET = main

# Directory
SRC_DIR = src
LIB_DIR = lib
OBJ_DIR = build/obj
BIN_DIR = build/bin
OUTPUT_DIR = run
TEST_DIR = test
TEST_BIN_DIR = $(BIN_DIR)/test

# Trova tutti i file sorgenti .cu in src e nelle sottodirectory di lib
SRCS = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(LIB_DIR)/*/*.cu)

# Crea i file oggetto corrispondenti mantenendo la struttura delle directory
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/src/%.o,$(wildcard $(SRC_DIR)/*.cu)) \
       $(patsubst $(LIB_DIR)/%/%.cu,$(OBJ_DIR)/lib/%/%.o,$(wildcard $(LIB_DIR)/*/*.cu))

# Trova tutte le sottodirectory in lib e le aggiunge al percorso degli include
INCLUDE_DIRS = $(shell find $(LIB_DIR) -type d)
INCLUDE_FLAGS = $(addprefix -I, $(INCLUDE_DIRS)) -I$(LIB_DIR)

# Compilatore e flags
NVCC = nvcc
NVCC_FLAGS = -O3 -lineinfo $(INCLUDE_FLAGS) -lcublas -arch=sm_80

main: all

# Crea le directory bin e obj se non esistono
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)/src $(OBJ_DIR)/lib

# Regola di default
all: $(BIN_DIR) $(OBJ_DIR) $(BIN_DIR)/$(TARGET)

# Compilazione dei file oggetto dalla directory src
$(OBJ_DIR)/src/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compilazione dei file oggetto dalle sottodirectory di lib (come lib/matMul)
$(OBJ_DIR)/lib/%/%.o: $(LIB_DIR)/%/%.cu
	mkdir -p $(OBJ_DIR)/lib/$*
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link e generazione dell'eseguibile principale
$(BIN_DIR)/$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@

# --------------------------------
# Sezione per i test
# --------------------------------

# Trova tutte le sottodirectory in test/
TESTS = $(shell find $(TEST_DIR) -mindepth 1 -maxdepth 1 -type d)
TEST_BINS = $(patsubst $(TEST_DIR)/%, $(TEST_BIN_DIR)/%, $(TESTS))

# Regola per creare tutti i test
test: $(TEST_BIN_DIR) $(TEST_BINS)

# Regola per creare la directory dei binari dei test
$(TEST_BIN_DIR):
	mkdir -p $(TEST_BIN_DIR)

# Regola per compilare ogni test con le librerie di lib/
$(TEST_BIN_DIR)/%: $(TEST_DIR)/% $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(wildcard $</*.cu) $(OBJS) -o $@

# Pulizia dei file generati
clean:
	rm -rf $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR)

# Pulizia dei file oggetto e dell'eseguibile
cleanbuild:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

# Pulizia più approfondita
cleanall: clean cleanbuild
