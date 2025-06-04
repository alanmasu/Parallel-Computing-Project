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

# Compilatore e flags
NVCC = nvcc
NVCC_FLAGS = -O3 $(INCLUDE_FLAGS) -lcublas -arch=sm_80 -DWMMA_BATCHED -DTESTING

# Trova tutti i file sorgenti .cu in src e nelle sottodirectory di lib e test
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
LIB_FILES = $(wildcard $(LIB_DIR)/*/*.cu)
TEST_FILES = $(wildcard $(TEST_DIR)/*/*.cu)

# Crea i file oggetto corrispondenti mantenendo la struttura delle directory
SRC_OBJS = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/src/%,$(SRC_FILES:.cu=.o))
LIB_OBJS = $(patsubst $(LIB_DIR)/%,$(OBJ_DIR)/lib/%,$(LIB_FILES:.cu=.o))
TEST_OBJS = $(patsubst $(TEST_DIR)/%,$(OBJ_DIR)/test/%,$(TEST_FILES:.cu=.o))

# Trova tutte le sottodirectory in test/
TESTS = $(shell find $(TEST_DIR) -mindepth 1 -maxdepth 1 -type d)
TEST_BINS = $(patsubst $(TEST_DIR)/%, $(TEST_BIN_DIR)/%, $(TESTS))

# Trova tutte le sottodirectory in lib e le aggiunge al percorso degli include
INCLUDE_DIRS = $(shell find $(LIB_DIR) -type d)
INCLUDE_FLAGS = $(addprefix -I, $(INCLUDE_DIRS)) -I$(LIB_DIR)

main: $(BIN_DIR)/$(TARGET)

# Crea le directory necessarie
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Regola di default
all: main test

# Compilazione dei file oggetto dalla directory src
$(SRC_OBJS): $(SRC_FILES)
	@echo ""
	@echo "Compiling src files..."
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -dc -c $< -o $@

# Compilazione dei file oggetto dalle sottodirectory di lib
$(LIB_OBJS): $(LIB_FILES)
	@echo ""
	@echo "Compiling lib files..."
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -dc -c $< -o $@

# Link e generazione dell'eseguibile principale
$(BIN_DIR)/$(TARGET): $(LIB_OBJS) $(SRC_OBJS)
	@echo ""
	@echo "Linking..."
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# --------------------------------
# Sezione per i test
# --------------------------------

# Regola per creare tutti i test
test: $(TEST_BIN_DIR) $(TEST_BINS)

# Regola per creare la directory dei binari dei test
$(TEST_BIN_DIR):
	mkdir -p $(TEST_BIN_DIR)

# Compilazione dei file oggetto dei test
$(TEST_OBJS): $(TEST_FILES)
	@echo ""
	@echo "Compiling test files..."
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -dc -c $(filter %/$(patsubst %.o,%.cu, $(notdir $@)), $(TEST_FILES)) -o $@

# Link e generazione dei binari dei test
$(TEST_BINS): $(LIB_OBJS) $(TEST_OBJS)
	@echo ""
	@echo "Linking test..."
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(filter %/$(notdir $@).o, $(TEST_OBJS)) $(LIB_OBJS) -o $@

# Pulizia dei file generati
clean:
	rm -rf $(OUTPUT_DIR)
	mkdir -p $(OUTPUT_DIR)

# Pulizia dei file oggetto e dell'eseguibile
cleanbuild:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

# Pulizia più approfondita
cleanall: clean cleanbuild

# Regola per il profiling
NVCC_FLAGS_PROFILE = $(NVCC_FLAGS) -DN_RUNS=1 -DSIZE_END=8192 -DCUDA_PROFILING
profile: 
	$(MAKE) NVCC_FLAGS="$(NVCC_FLAGS_PROFILE)" main
