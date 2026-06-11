# ============================================================
# Makefile – Integral Graphs (CUDA, arytmetyka całkowitoliczbowa)
# ============================================================
#
# Użycie:
#   make                      – kompilacja
#   make ARCH=sm_89           – kompilacja z architekturą sm_89
#   make test                 – test z plikiem grafy17L.graph6 (oczekiwane: 233)
#   make test27               – test z filtrem e=27 (oczekiwane: 8)
#   make clean                – czyszczenie
#
# Zmienne środowiskowe:
#   CUDA_PATH         – ścieżka do CUDA Toolkit (domyślnie: /usr/local/cuda)
#   NVCC              – ścieżka do kompilatora nvcc
#   ARCH              – architektura GPU (domyślnie: sm_70)
# ============================================================

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc
ARCH      ?= sm_70

# Flagi kompilacji
NVCC_FLAGS  = -std=c++17 -O3 -arch=$(ARCH)
NVCC_FLAGS += --use_fast_math
NVCC_FLAGS += -Xcompiler -Wall -Xcompiler -pthread

# Biblioteki (tylko CUDA runtime – bez cuSOLVER)
LIBS = -lcudart -lpthread

# Pliki
TARGET = integral_graphs
SRC    = integral_graphs.cu

# ============================================================
# Reguły
# ============================================================

.PHONY: all clean test test27 test_geng

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

# --- Test: wszystkie grafy z pliku (oczekiwane: 233 integralnych) ---
test: $(TARGET)
	@echo "============================================"
	@echo "  TEST: grafy17L.graph6 (wszystkie)"
	@echo "  Oczekiwany wynik: 233 grafów integralnych"
	@echo "============================================"
	LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$LD_LIBRARY_PATH ./$(TARGET) grafy17L.graph6

# --- Test z filtrem e=27 (oczekiwane: 8 integralnych) ---
test27: $(TARGET)
	@echo "============================================"
	@echo "  TEST: grafy17L.graph6 (filtr: e=27)"
	@echo "  Oczekiwany wynik: 8 grafów integralnych"
	@echo "============================================"
	LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$LD_LIBRARY_PATH ./$(TARGET) grafy17L.graph6 -e 27

# --- Test z geng (wymaga nauty) ---
test_geng: $(TARGET)
	@echo "============================================"
	@echo "  TEST: geng -c 17 27:27 | integral_graphs"
	@echo "============================================"
	LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$LD_LIBRARY_PATH geng -c 17 27:27 | ./$(TARGET)

clean:
	rm -f $(TARGET)
