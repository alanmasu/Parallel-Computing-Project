# Nome dell'eseguibile
TARGET = main

# Directory
SRC_DIR = src
LIB_DIR = lib
OBJ_DIR = build/obj
BIN_DIR = build/bin
OUTPUT_DIR = run

# Trova tutti i file sorgenti .cu in src e nelle sottodirectory di lib
SRCS = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(LIB_DIR)/*/*.cu)

# Crea i file oggetto corrispondenti mantenendo la struttura delle directory
OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.o,$(notdir $(SRCS)))

# Trova tutte le sottodirectory in lib e le aggiunge al percorso degli include
INCLUDE_DIRS = $(shell find $(LIB_DIR) -type d)
INCLUDE_FLAGS = $(addprefix -I, $(INCLUDE_DIRS))

# Compilatore e flags
NVCC = nvcc
NVCC_FLAGS = -O3 $(INCLUDE_FLAGS)

# Regola di default
all: $(BIN_DIR) $(OBJ_DIR) $(BIN_DIR)/$(TARGET)


# Crea le directory bin e obj se non esistono
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compilazione dei file oggetto dalla directory src
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compilazione dei file oggetto dalle sottodirectory di lib
$(OBJ_DIR)/%.o: $(LIB_DIR)/%/*.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link e generazione dell'eseguibile
$(BIN_DIR)/$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@

# Pulizia dei file generati
clean:
	rm -rf $(OUTPUT_DIR) 
	mkdir -p $(OUTPUT_DIR)

# Pulizia più approfondita
cleanall: clean
	rm -rf $(BIN_DIR) $(OBJ_DIR)



