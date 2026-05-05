/**
 * automorphisms.cpp
 *
 * Zliczanie automorfizmów grafu metodą brute-force
 * z równoległością OpenMP i arytmetyką dużych liczb GMP.
 *
 * Wejście:  linie w formacie graph6 (stdin, pipe z geng)
 * Wyjście:  dla każdego grafu — graph6, n, |Aut(G)|, czas
 *
 * Kompilacja:
 *   g++ -O2 -fopenmp -std=c++17 -o automorphisms automorphisms.cpp -lgmp -lgmpxx
 *
 * Użycie:
 *   echo "Dhc" | ./automorphisms
 *   geng 7 | ./automorphisms
 *   OMP_NUM_THREADS=8 geng 8 | ./automorphisms
 */

#include <iostream>
#include <cstring>
#include <omp.h>
#include <gmpxx.h>

static const int MAX_N = 64;

static const int MAX_LINE = 4096;

/* ============================================================
 * Prekomputacja silni (factorial lookup table)
 * 20! = 2 432 902 008 176 640 000  < 2^63
 * 21! = 51 090 942 171 709 440 000 > 2^63 (overflow)
 * Dla n > 20 brute-force jest i tak niewykonalny czasowo.
 * ============================================================ */
static const int MAX_FACT = 21;
static unsigned long long FACT[MAX_FACT + 1];

static void precompute_factorials() {
    FACT[0] = 1;
    for (int i = 1; i <= MAX_FACT; i++)
        FACT[i] = FACT[i - 1] * (unsigned long long)i;
}

/* ============================================================
 * Dekodowanie formatu graph6 → macierz sąsiedztwa
 *
 * Format graph6 (nauty/geng):
 *   - Pierwszy bajt(y) kodują n (liczbę wierzchołków)
 *   - Kolejne bajty zawierają 6-bitowe paczki, wypełniające
 *     dolny trójkąt macierzy sąsiedztwa kolumnami:
 *       for j = 1..n-1:
 *         for i = 0..j-1:
 *           adj[i][j] = adj[j][i] = next_bit
 * ============================================================ */
static int decode_graph6(const char* s, int adj[][MAX_N]) {
    int idx = 0;
    int n;

    /* Dekodowanie liczby wierzchołków */
    if ((unsigned char)s[0] == 126) {         /* '~' → n >= 63 */
        if ((unsigned char)s[1] == 126) {     /* '~~' → n >= 258048 */
            n = 0;
            for (int i = 2; i < 8; i++)
                n = (n << 6) | ((unsigned char)s[i] - 63);
            idx = 8;
        } else {                              /* '~' → 63 <= n < 258048 */
            n = 0;
            for (int i = 1; i < 4; i++)
                n = (n << 6) | ((unsigned char)s[i] - 63);
            idx = 4;
        }
    } else {                                  /* n < 63 */
        n = (unsigned char)s[0] - 63;
        idx = 1;
    }

    /* Zerowanie macierzy */
    std::memset(adj, 0, sizeof(int) * MAX_N * MAX_N);

    /* Wypełnianie macierzy sąsiedztwa z bitów */
    int bit_pos = 5;     /* bieżąca pozycja bitu w bajcie (5 → 0) */
    int byte_idx = idx;  /* bieżący indeks bajtu w stringu */

    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            int val = (unsigned char)s[byte_idx] - 63;
            int bit = (val >> bit_pos) & 1;
            adj[i][j] = adj[j][i] = bit;

            if (bit_pos == 0) {
                bit_pos = 5;
                byte_idx++;
            } else {
                bit_pos--;
            }
        }
    }

    return n;
}

/* ============================================================
 * Sprawdzenie, czy permutacja jest automorfizmem grafu
 *
 * Automorfizm: perm jest automorfizmem ⟺
 *   ∀ i < j: adj[perm[i]][perm[j]] == adj[i][j]
 *
 * Optymalizacja: early exit przy pierwszej niezgodności.
 * ============================================================ */
static inline bool is_automorphism(const int adj[][MAX_N],
                                   const int perm[], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (adj[perm[i]][perm[j]] != adj[i][j])
                return false;
        }
    }
    return true;
}

/* ============================================================
 * Konwersja indeksu na permutację (Lehmer code / system faktoradyczny)
 *
 * Dla indeksu idx ∈ [0, n!) generuje odpowiednią permutację.
 * Każdy indeks odpowiada dokładnie jednej permutacji,
 * co umożliwia niezależne przetwarzanie w OpenMP parallel for.
 *
 * Złożoność: O(n²) — akceptowalne dla n ≤ 20.
 * ============================================================ */
static inline void index_to_permutation(unsigned long long idx, int n,
                                        int perm[],
                                        const unsigned long long fact[]) {
    int available[MAX_N];
    for (int i = 0; i < n; i++)
        available[i] = i;

    for (int i = 0; i < n; i++) {
        unsigned long long f = fact[n - 1 - i];  /* (remaining - 1)! */
        int pos = (int)(idx / f);
        idx %= f;

        perm[i] = available[pos];

        /* Usunięcie wybranego elementu (przesunięcie tablicy) */
        int remaining = n - i;
        for (int k = pos; k < remaining - 1; k++)
            available[k] = available[k + 1];
    }
}

/* ============================================================
 * Zliczanie automorfizmów grafu z użyciem OpenMP
 *
 * Strategia:
 *   - Pętla po indeksach 0..n!-1 z #pragma omp parallel for
 *   - Każdy wątek generuje permutację z indeksu (Lehmer code)
 *   - Wynik (mpz_class) akumulowany per-thread, sumowany
 *     w sekcji krytycznej (redukcja OMP nie wspiera GMP)
 *   - schedule(dynamic, 1024) — równoważenie obciążenia
 * ============================================================ */
static mpz_class count_automorphisms(const int adj[][MAX_N], int n) {
    if (n <= 1)
        return mpz_class(1);

    if (n > 20) {
        std::cerr << "UWAGA: n=" << n
                  << " — n! przekracza zakres unsigned long long. "
                  << "Brute-force jest niewykonalny dla tak dużych grafów."
                  << std::endl;
        return mpz_class(-1);  /* sygnał: nie udało się obliczyć */
    }

    unsigned long long factorial = FACT[n];

    mpz_class total_count(0);

    #pragma omp parallel
    {
        mpz_class local_count(0);
        int perm[MAX_N];

        #pragma omp for schedule(dynamic, 1024)
        for (unsigned long long idx = 0; idx < factorial; idx++) {
            index_to_permutation(idx, n, perm, FACT);

            if (is_automorphism(adj, perm, n))
                local_count++;
        }

        #pragma omp critical
        {
            total_count += local_count;
        }
    }

    return total_count;
}

/* ============================================================
 * Funkcja główna — pętla while(getline) przetwarzająca stdin
 *
 * Obsługuje:
 *   - Wiele grafów (pipe z geng)
 *   - Nagłówek >>graph6<< (pomijany)
 *   - Puste linie (pomijane)
 * ============================================================ */
int main() {
    precompute_factorials();

    char line[MAX_LINE];

    while (std::cin.getline(line, sizeof(line))) {
        /* Usunięcie trailing whitespace / CR / LF */
        int len = (int)std::strlen(line);
        while (len > 0 && (line[len - 1] == '\r' ||
                           line[len - 1] == '\n' ||
                           line[len - 1] == ' '))
            line[--len] = '\0';

        if (len == 0)
            continue;

        /* Pominięcie nagłówka graph6 */
        const char* data = line;
        if (std::strncmp(data, ">>graph6<<", 10) == 0)
            data += 10;

        /* Pominięcie linii sparse6 (zaczyna się od ':') */
        if (data[0] == ':') {
            std::cerr << "UWAGA: sparse6 nie jest obsługiwane, pomijam: "
                      << line << std::endl;
            continue;
        }

        /* Dekodowanie i zliczanie */
        int adj[MAX_N][MAX_N];
        double t0 = omp_get_wtime();

        int n = decode_graph6(data, adj);
        mpz_class count = count_automorphisms(adj, n);

        double t1 = omp_get_wtime();

        std::cout << data
                  << " : n=" << n
                  << " |Aut(G)| = " << count
                  << " time=" << (t1 - t0) << "s"
                  << std::endl;
    }

    return 0;
}
