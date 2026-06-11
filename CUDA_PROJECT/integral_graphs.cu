/**
 * integral_graphs.cu
 *
 * Search for integral graphs (all eigenvalues are integers)
 * among connected graphs of order 17.
 *
 * Thread-Local Buffers & uint64_t GPU Optimization
 *   - No mutex locks for individual graphs during CPU parsing.
 *   - Matrix memory on GPU reduced by half (uint64_t) to prevent spilling to VRAM.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <condition_variable>

#include <cuda_runtime.h>

#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

/* ================================================================
 * Configuration
 * ================================================================ */
static constexpr int GRAPH_N       = 17;
static constexpr int G6_STRIDE     = 32;
static constexpr int BATCH_SIZE    = 131072;   /* 128K graphs per batch */
static constexpr int THREADS_BLK   = 128;
static constexpr int MAX_OUTPUT    = 65536;

static int          g_filter_edges = -1;
static const char*  g_geng_path    = nullptr;
static int          g_graph_n      = GRAPH_N;
static const char*  g_out_path     = nullptr;
static bool         g_quiet        = false;

#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while (0)

/* ================================================================
 * GPU Kernel
 * ================================================================ */
__global__ void integral_kernel(
    const char*  __restrict__ g6_data,
    int          batch_count,
    int          filter_edges,
    int*         __restrict__ d_out_count,
    int*         __restrict__ d_out_indices
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= batch_count) return;

    const char* line = g6_data + (long long)gid * G6_STRIDE;

    uint32_t adj[GRAPH_N];
    for (int i = 0; i < GRAPH_N; i++) adj[i] = 0;

    int pos = 1;
    int bits_left = 0, cur = 0, edge_count = 0;

    for (int col = 1; col < GRAPH_N; col++) {
        for (int row = 0; row < col; row++) {
            if (bits_left == 0) {
                cur = (unsigned char)line[pos++] - 63;
                bits_left = 6;
            }
            bits_left--;
            if ((cur >> bits_left) & 1) {
                adj[row] |= (1u << col);
                adj[col] |= (1u << row);
                edge_count++;
            }
        }
    }

    if (filter_edges > 0 && edge_count != filter_edges) return;

    /* Calculate max degree (needed to bound roots) */
    int max_deg = 0;
    for (int i = 0; i < GRAPH_N; i++) {
        int d = __popc(adj[i]);
        if (d > max_deg) max_deg = d;
    }

    /* ================================================================
     * PHASE 1: Matrix powers with UNIVERSAL EARLY EXIT
     *
     * For any integral graph with E edges, 
     * trace congruences based on Fermat's Little Theorem hold:
     *   - Mod 3: trace(A^k) ≡ trace(A^{k-2})
     *   - Mod 5: trace(A^k) ≡ trace(A^{k-4})
     *   - Mod 7: trace(A^k) ≡ trace(A^{k-6})
     *   - Mod 4 i 8: result from root parity
     * ================================================================ */
    uint64_t P[GRAPH_N][GRAPH_N];
    __int128 traces[GRAPH_N + 1];

    for (int i = 0; i < GRAPH_N; i++)
        for (int j = 0; j < GRAPH_N; j++)
            P[i][j] = (uint64_t)((adj[i] >> j) & 1u);

    traces[1] = 0;

    /* Variables to hold base traces for higher powers */
    uint64_t t3_mod8 = 0, t4_mod8 = 0;
    uint64_t t3_mod5 = 0, t4_mod5 = 0;
    uint64_t t3_mod7 = 0, t4_mod7 = 0, t5_mod7 = 0, t6_mod7 = 0;

    for (int k = 2; k <= GRAPH_N; k++) {
        uint64_t Q[GRAPH_N][GRAPH_N];
        for (int i = 0; i < GRAPH_N; i++) {
            for (int j = 0; j < GRAPH_N; j++) {
                uint64_t s = 0;
                uint32_t col_bits = adj[j];
                while (col_bits) {
                    int l = __ffs(col_bits) - 1;
                    s += P[i][l];
                    col_bits &= col_bits - 1;
                }
                Q[i][j] = s;
            }
        }
        __int128 tr = 0;
        for (int i = 0; i < GRAPH_N; i++) tr += (__int128)Q[i][i];
        traces[k] = tr;

        /* --- EARLY EXIT: universal mathematical integrality conditions --- */
        if (k >= 3) {
            uint64_t lo = (uint64_t)((unsigned __int128)tr);
            uint64_t hi = (uint64_t)((unsigned __int128)tr >> 64);
            
            /* Fast modulo for 128-bit:
               2^64 ≡ 1 (mod 3), 2^64 ≡ 1 (mod 5), 2^64 ≡ 2 (mod 7) */
            uint64_t r3 = ((hi % 3) + (lo % 3)) % 3;
            uint64_t r5 = ((hi % 5) + (lo % 5)) % 5;
            uint64_t r7 = ((hi % 7) * 2 + (lo % 7)) % 7;
            uint64_t r8 = lo & 7;

            /* Save bases (k=3..6) */
            if (k == 3) { t3_mod8 = r8; t3_mod5 = r5; t3_mod7 = r7; }
            if (k == 4) { t4_mod8 = r8; t4_mod5 = r5; t4_mod7 = r7; }
            if (k == 5) { t5_mod7 = r7; }
            if (k == 6) { t6_mod7 = r7; }

            /* Mod 3 Test (Period 2) */
            if (k % 2 != 0) {
                if (r3 != 0) return;
            } else {
                if (r3 != (2 * edge_count) % 3) return;
            }

            /* Mod 4 and Mod 8 Test */
            if (k % 2 != 0) {
                if (k >= 5 && r8 != t3_mod8) return;
            } else {
                if ((lo & 3) != ((2 * edge_count) & 3)) return; /* trace ≡ 2E (mod 4) */
                if (k >= 6 && r8 != t4_mod8) return;
            }

            /* Mod 5 Test (Period 4) */
            if (k >= 5) {
                int m = k % 4;
                if (m == 1 && r5 != 0) return;
                if (m == 2 && r5 != (2 * edge_count) % 5) return;
                if (m == 3 && r5 != t3_mod5) return;
                if (m == 0 && r5 != t4_mod5) return;
            }

            /* Mod 7 Test (Period 6) */
            if (k >= 7) {
                int m = k % 6;
                if (m == 1 && r7 != 0) return;
                if (m == 2 && r7 != (2 * edge_count) % 7) return;
                if (m == 3 && r7 != t3_mod7) return;
                if (m == 4 && r7 != t4_mod7) return;
                if (m == 5 && r7 != t5_mod7) return;
                if (m == 0 && r7 != t6_mod7) return;
            }
        }

        for (int i = 0; i < GRAPH_N; i++)
            for (int j = 0; j < GRAPH_N; j++)
                P[i][j] = Q[i][j];
    }

    /* ================================================================
     * PHASE 2: Newton's Identities -> characteristic polynomial coefficients
     * ================================================================ */
    int64_t c[GRAPH_N + 1];
    c[0] = 1;
    for (int k = 1; k <= GRAPH_N; k++) {
        __int128 sum = traces[k];
        for (int i = 1; i < k; i++)
            sum += (__int128)c[i] * traces[k - i];
        c[k] = (int64_t)(-sum / k);
    }

    /* ================================================================
     * PHASE 3: Finding roots (optimized order)
     *
     * Test 0, ±1, ±2, ..., ±max_deg (most common eigenvalues first).
     * After each deflation check the constant term (Rational Root Theorem):
     * if |constant_term| < next tested value, there are no more roots.
     * ================================================================ */
    int64_t poly[GRAPH_N + 1];
    for (int i = 0; i <= GRAPH_N; i++) poly[i] = c[i];

    int deg = GRAPH_N;
    int total_mult = 0;

    /* Test x = 0 (most common eigenvalue) */
    while (deg > 0 && poly[deg] == 0) {
        total_mult++;
        deg--;
    }

    /* Test ±1, ±2, ..., ±max_deg */
    for (int abs_x = 1; abs_x <= max_deg && deg > 0; abs_x++) {
        for (int sign = 1; sign >= -1 && deg > 0; sign -= 2) {
            int x = sign * abs_x;
            while (deg > 0) {
                __int128 q128[GRAPH_N + 1];
                q128[0] = (__int128)poly[0];
                for (int i = 1; i <= deg; i++)
                    q128[i] = q128[i - 1] * (__int128)x + (__int128)poly[i];

                if (q128[deg] != 0) break;

                total_mult++;
                deg--;
                for (int i = 0; i <= deg; i++)
                    poly[i] = (int64_t)q128[i];
            }
        }

        /* Rational Root Theorem: after deflation check constant term.
           Every remaining integer root r must divide the constant term.
           If |constant_term| < abs_x+1, we have tested all divisors. */
        if (deg > 0 && poly[deg] != 0) {
            int64_t ct = poly[deg] < 0 ? -poly[deg] : poly[deg];
            if (ct < (int64_t)(abs_x + 1)) break;
        }

        /* Handle new x=0 roots created after deflation */
        while (deg > 0 && poly[deg] == 0) {
            total_mult++;
            deg--;
        }
    }

    if (total_mult == GRAPH_N) {
        int pos_out = atomicAdd(d_out_count, 1);
        if (pos_out < MAX_OUTPUT)
            d_out_indices[pos_out] = gid;
    }
}

/* ================================================================
 * GPU Buffer
 * ================================================================ */
struct GpuBuffer {
    char* current_batch_data = nullptr;

    int*  h_out_count  = nullptr;
    int*  h_out_indices = nullptr;
    char* d_g6         = nullptr;
    int*  d_out_count  = nullptr;
    int*  d_out_indices = nullptr;
    cudaStream_t stream = nullptr;

    void allocate() {
        CUDA_CHECK(cudaMallocHost(&h_out_count, sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_out_indices, (size_t)MAX_OUTPUT * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_g6, (size_t)BATCH_SIZE * G6_STRIDE));
        CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_indices, (size_t)MAX_OUTPUT * sizeof(int)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    void release() {
        if (h_out_count)  { cudaFreeHost(h_out_count);  h_out_count = nullptr; }
        if (h_out_indices){ cudaFreeHost(h_out_indices); h_out_indices = nullptr; }
        if (d_g6)         { cudaFree(d_g6);             d_g6 = nullptr; }
        if (d_out_count)  { cudaFree(d_out_count);      d_out_count = nullptr; }
        if (d_out_indices){ cudaFree(d_out_indices);     d_out_indices = nullptr; }
        if (stream)       { cudaStreamDestroy(stream);   stream = nullptr; }
    }
};

/* ================================================================
 * Memory Pool (GraphPool) - Optimized for Thread-Local Buffers
 * CPU threads use it EXCLUSIVELY to fetch empty packets and
 * return fully written batches (lock called only every 131k graphs)
 * ================================================================ */
struct GraphPool {
    static constexpr int NUM_BUFFERS = 32;
    char* buffers[NUM_BUFFERS];

    std::vector<char*> free_buffers;
    struct Batch { char* data; int count; };
    std::queue<Batch> ready_queue;

    std::mutex mtx;
    std::condition_variable cv_gpu;
    std::condition_variable cv_cpu;
    int active_readers = 0;

    void init(int num_readers) {
        for (int i = 0; i < NUM_BUFFERS; i++) {
            char* ptr;
            CUDA_CHECK(cudaMallocHost(&ptr, (size_t)BATCH_SIZE * G6_STRIDE));
            buffers[i] = ptr;
            free_buffers.push_back(ptr);
        }
        active_readers = num_readers;
    }

    void destroy() {
        for (int i = 0; i < NUM_BUFFERS; i++) {
            cudaFreeHost(buffers[i]);
        }
    }

    char* get_free_buffer() {
        std::unique_lock<std::mutex> lk(mtx);
        cv_cpu.wait(lk, [this]{ return !free_buffers.empty(); });
        char* buf = free_buffers.back();
        free_buffers.pop_back();
        return buf;
    }

    void submit_batch(char* data, int count) {
        std::unique_lock<std::mutex> lk(mtx);
        ready_queue.push({data, count});
        cv_gpu.notify_one();
    }

    void reader_done(char* data, int count) {
        std::unique_lock<std::mutex> lk(mtx);
        if (count > 0) {
            ready_queue.push({data, count});
        } else if (data) {
            free_buffers.push_back(data);
            cv_cpu.notify_one();
        }
        active_readers--;
        if (active_readers == 0) {
            cv_gpu.notify_one();
        }
    }

    bool take_batch(Batch& out) {
        std::unique_lock<std::mutex> lk(mtx);
        cv_gpu.wait(lk, [this]{ return !ready_queue.empty() || active_readers == 0; });
        if (ready_queue.empty()) return false;
        out = ready_queue.front();
        ready_queue.pop();
        return true;
    }

    void recycle_buffer(char* ptr) {
        std::unique_lock<std::mutex> lk(mtx);
        free_buffers.push_back(ptr);
        cv_cpu.notify_one();
    }
};

struct Job {
    int   frag_id;
    pid_t pid;
    FILE* fp;
};

struct JobQueue {
    std::queue<Job>             queue;
    std::mutex                  mtx;
    std::condition_variable     cv;
    bool                        producer_done = false;
    int                         total_jobs = 0;
    std::atomic<int>            done_count{0};

    void push(Job job) {
        { std::lock_guard<std::mutex> lk(mtx); queue.push(job); }
        cv.notify_one();
    }

    void finish() {
        { std::lock_guard<std::mutex> lk(mtx); producer_done = true; }
        cv.notify_all();
    }

    bool take(Job& out) {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this]{ return !queue.empty() || producer_done; });
        if (queue.empty()) return false;
        out = queue.front();
        queue.pop();
        return true;
    }

    void mark_done() { done_count.fetch_add(1, std::memory_order_relaxed); }
};

/* ================================================================
 * Orchestration - fork/pipe to geng
 * ================================================================ */
static Job spawn_geng_job(int frag_id, int n, int edges, int total_mod) {
    Job job = {frag_id, -1, nullptr};
    int pipefd[2];
    if (pipe(pipefd) < 0) { perror("pipe"); return job; }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(pipefd[0]); close(pipefd[1]);
        return job;
    }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }

        char n_str[16], edge_range[32], res_mod[32];
        snprintf(n_str, sizeof(n_str), "%d", n);
        snprintf(edge_range, sizeof(edge_range), "%d:%d", edges, edges);
        snprintf(res_mod, sizeof(res_mod), "%d/%d", frag_id, total_mod);

        execlp(g_geng_path, "geng", "-cq", n_str, edge_range, res_mod, nullptr);
        _exit(127);
    }

    close(pipefd[1]);
    job.pid = pid;
    job.fp = fdopen(pipefd[0], "r");
    return job;
}

/* ================================================================
 * Worker State (visible to Monitor)
 * ================================================================ */
struct WorkerState {
    std::atomic<int>       current_frag{-1};
    std::atomic<long long> frags_done{0};
    std::atomic<long long> graphs_pushed{0};
};

/* ================================================================
 * CPU reader thread 
 * Operates completely lock-free (Lock-Free Thread-Local Buffering)
 * ================================================================ */
static void reader_thread_fn(
    int              worker_id,
    JobQueue&        job_queue,
    GraphPool&       pool,
    WorkerState&     state,
    std::atomic<long long>& global_skipped
) {
    long long local_pushed = 0;
    long long local_skip = 0;

    char* local_buf = pool.get_free_buffer();
    int   local_count = 0;

    Job job;
    while (job_queue.take(job)) {
        state.current_frag.store(job.frag_id, std::memory_order_relaxed);

        if (!job.fp) {
            job_queue.mark_done();
            continue;
        }

        char io_buf[131072];
        int io_len = 0;

        while (true) {
            size_t bytes = fread(io_buf + io_len, 1, sizeof(io_buf) - io_len, job.fp);
            if (bytes == 0) break;
            io_len += (int)bytes;

            int start = 0;
            for (int i = 0; i < io_len; i++) {
                if (io_buf[i] == '\n' || io_buf[i] == '\r') {
                    int len = i - start;
                    if (len > 0) {
                        if (io_buf[start] - 63 == GRAPH_N) {
                            char* dst = local_buf + local_count * G6_STRIDE;
                            memset(dst, 0, G6_STRIDE);
                            int cpy = len < G6_STRIDE ? len : G6_STRIDE - 1;
                            memcpy(dst, io_buf + start, cpy);
                            local_count++;

                            /* If local batch is full -> return to pool -> get new one */
                            if (local_count == BATCH_SIZE) {
                                pool.submit_batch(local_buf, local_count);
                                local_buf = pool.get_free_buffer();
                                local_count = 0;
                            }
                            local_pushed++;
                        } else {
                            local_skip++;
                        }
                    }
                    start = i + 1;
                }
            }

            if (start < io_len) {
                memmove(io_buf, io_buf + start, io_len - start);
                io_len -= start;
            } else {
                io_len = 0;
            }
        }

        /* The last graph in the file might not have a newline character at the end (\n) */
        if (io_len > 0) {
            if (io_buf[0] - 63 == GRAPH_N) {
                char* dst = local_buf + local_count * G6_STRIDE;
                memset(dst, 0, G6_STRIDE);
                int cpy = io_len < G6_STRIDE ? io_len : G6_STRIDE - 1;
                memcpy(dst, io_buf, cpy);
                local_count++;

                if (local_count == BATCH_SIZE) {
                    pool.submit_batch(local_buf, local_count);
                    local_buf = pool.get_free_buffer();
                    local_count = 0;
                }
                local_pushed++;
            } else {
                local_skip++;
            }
        }

        if (job.pid > 0) {
            fclose(job.fp);
            waitpid(job.pid, nullptr, 0);
        }

        state.frags_done.fetch_add(1, std::memory_order_relaxed);
        state.graphs_pushed.store(local_pushed, std::memory_order_relaxed);
        job_queue.mark_done();
    }

    state.current_frag.store(-1, std::memory_order_relaxed);
    global_skipped.fetch_add(local_skip, std::memory_order_relaxed);
    
    pool.reader_done(local_buf, local_count);
}

/* ================================================================
 * GPU thread
 * ================================================================ */
static void gpu_thread_fn(
    GraphPool&                  pool,
    std::mutex&                 result_mtx,
    std::vector<std::string>&   global_results,
    std::atomic<long long>&     global_processed,
    std::atomic<long long>&     gpu_batches_done,
    FILE*                       out_fp
) {
    GpuBuffer bufs[2];
    bufs[0].allocate();
    bufs[1].allocate();

    int cur = 0;
    int prev_idx = -1;
    int prev_count = 0;

    GraphPool::Batch batch;
    while (pool.take_batch(batch)) {
        GpuBuffer* b = &bufs[cur];
        b->current_batch_data = batch.data;

        CUDA_CHECK(cudaMemsetAsync(b->d_out_count, 0, sizeof(int), b->stream));
        CUDA_CHECK(cudaMemcpyAsync(b->d_g6, batch.data, (size_t)batch.count * G6_STRIDE,
                                   cudaMemcpyHostToDevice, b->stream));
        int blocks = (batch.count + THREADS_BLK - 1) / THREADS_BLK;
        integral_kernel<<<blocks, THREADS_BLK, 0, b->stream>>>(
            b->d_g6, batch.count, g_filter_edges, b->d_out_count, b->d_out_indices);
        CUDA_CHECK(cudaMemcpyAsync(b->h_out_count, b->d_out_count, sizeof(int),
                                   cudaMemcpyDeviceToHost, b->stream));

        if (prev_idx >= 0) {
            GpuBuffer* p = &bufs[prev_idx];
            CUDA_CHECK(cudaStreamSynchronize(p->stream));
            int found = *(p->h_out_count);
            if (found > 0) {
                if (found > MAX_OUTPUT) found = MAX_OUTPUT;
                CUDA_CHECK(cudaMemcpy(p->h_out_indices, p->d_out_indices,
                                      (size_t)found * sizeof(int), cudaMemcpyDeviceToHost));
                std::lock_guard<std::mutex> lk(result_mtx);
                for (int i = 0; i < found; i++) {
                    int idx = p->h_out_indices[i];
                    std::string g6 = p->current_batch_data + idx * G6_STRIDE;
                    global_results.push_back(g6);
                    if (out_fp) {
                        fprintf(out_fp, "%s\n", g6.c_str());
                    }
                }
                if (out_fp) fflush(out_fp);
            }
            global_processed.fetch_add(prev_count, std::memory_order_relaxed);
            gpu_batches_done.fetch_add(1, std::memory_order_relaxed);
            
            pool.recycle_buffer(p->current_batch_data);
        }

        prev_idx = cur;
        prev_count = batch.count;
        cur = 1 - cur;
    }

    if (prev_idx >= 0) {
        GpuBuffer* p = &bufs[prev_idx];
        CUDA_CHECK(cudaStreamSynchronize(p->stream));
        int found = *(p->h_out_count);
        if (found > 0) {
            if (found > MAX_OUTPUT) found = MAX_OUTPUT;
            CUDA_CHECK(cudaMemcpy(p->h_out_indices, p->d_out_indices,
                                  (size_t)found * sizeof(int), cudaMemcpyDeviceToHost));
            std::lock_guard<std::mutex> lk(result_mtx);
            for (int i = 0; i < found; i++) {
                int idx = p->h_out_indices[i];
                std::string g6 = p->current_batch_data + idx * G6_STRIDE;
                global_results.push_back(g6);
                if (out_fp) {
                    fprintf(out_fp, "%s\n", g6.c_str());
                }
            }
            if (out_fp) fflush(out_fp);
        }
        global_processed.fetch_add(prev_count, std::memory_order_relaxed);
        gpu_batches_done.fetch_add(1, std::memory_order_relaxed);
        pool.recycle_buffer(p->current_batch_data);
    }

    bufs[0].release();
    bufs[1].release();
}

/* ================================================================
 * Producer thread (fork-only)
 * ================================================================ */
static void producer_thread_fn(
    JobQueue& job_queue, int frag_start, int frag_end, int total_mod
) {
    for (int frag = frag_start; frag < frag_end; frag++) {
        Job job = spawn_geng_job(frag, g_graph_n, g_filter_edges, total_mod);
        job_queue.push(job);
    }
    job_queue.finish();
}

/* ================================================================
 * Monitor thread
 * ================================================================ */
static void monitor_thread_fn(
    int                  num_readers,
    WorkerState*         states,
    JobQueue&            job_queue,
    std::atomic<long long>& gpu_batches,
    std::atomic<long long>& global_processed,
    std::atomic<bool>&   running
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    while (running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();

        int done = job_queue.done_count.load(std::memory_order_relaxed);
        int total = job_queue.total_jobs;
        long long proc = global_processed.load(std::memory_order_relaxed);
        long long batches = gpu_batches.load(std::memory_order_relaxed);

        long long pushed = 0;
        for (int i = 0; i < num_readers; i++)
            pushed += states[i].graphs_pushed.load(std::memory_order_relaxed);

        if (!g_quiet) {
            fprintf(stderr, "\r[%5.0fs] Frag: %d/%d (%.1f%%) | Pushed: %lld | GPU Done: %lld (batches: %lld, %.0f gr/s) | ",
                    elapsed, done, total, total > 0 ? 100.0 * done / total : 0.0,
                    pushed, proc, batches,
                    elapsed > 0 ? proc / elapsed : 0.0);

            for (int i = 0; i < num_readers; i++) {
                int f = states[i].current_frag.load(std::memory_order_relaxed);
                if (f >= 0) fprintf(stderr, "R%d:[%d] ", i, f);
                else        fprintf(stderr, "R%d:[--] ", i);
            }
            fprintf(stderr, "   ");
            fflush(stderr);
        }
    }
    if (!g_quiet) fprintf(stderr, "\n");
}

/* ================================================================
 * main()
 * ================================================================ */
int main(int argc, char* argv[]) {
    auto wall_start = std::chrono::high_resolution_clock::now();

    const char* input_path = nullptr;
    int num_readers  = 8;
    int frag_start   = 0;
    int frag_end     = -1;
    int frag_mod     = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i+1 < argc)
            g_filter_edges = atoi(argv[++i]);
        else if (strcmp(argv[i], "-g") == 0 && i+1 < argc)
            g_geng_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            g_graph_n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc)
            num_readers = atoi(argv[++i]);
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc) {
            char* arg = argv[++i];
            if (strchr(arg, '-')) {
                sscanf(arg, "%d-%d/%d", &frag_start, &frag_end, &frag_mod);
            } else {
                sscanf(arg, "%d/%d", &frag_start, &frag_mod);
                frag_end = frag_start + 1;
            }
        }
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)
            g_out_path = argv[++i];
        else if (strcmp(argv[i], "-q") == 0)
            g_quiet = true;
        else
            input_path = argv[i];
    }

    if (g_graph_n != GRAPH_N) {
        fprintf(stderr, "[ERROR] Compiled for n=%d, provided n=%d\n", GRAPH_N, g_graph_n);
        return 1;
    }

    std::mutex result_mtx;
    std::vector<std::string> integral_graphs;
    std::atomic<long long> global_processed{0};
    std::atomic<long long> global_skipped{0};
    std::atomic<long long> gpu_batches_done{0};

    if (frag_end < 0) frag_end = frag_mod;
    if (frag_end > frag_mod && frag_mod > 0) frag_end = frag_mod;
    
    int total_frags = (g_geng_path && frag_mod > 0) ? (frag_end - frag_start) : 1;
    if (num_readers > total_frags && g_geng_path && frag_mod > 0) num_readers = total_frags;
    if (!g_geng_path && !input_path) {
        num_readers = 1; /* stdin */
    }

    GraphPool pool;
    pool.init(num_readers);

    if (!g_quiet) {
        fprintf(stderr, "[INFO] Rząd grafu: %d | Batch: %d | Blok: %d wątków\n",
                g_graph_n, BATCH_SIZE, THREADS_BLK);
        fprintf(stderr, "[INFO] Edge filter: %s\n",
                g_filter_edges >= 0 ? std::to_string(g_filter_edges).c_str() : "NONE");
        if (g_geng_path) {
            fprintf(stderr, "[INFO] Architecture: %d Readers (Thread-Local Buffers), 1 GPU\n", num_readers);
            if (frag_mod > 0) {
                fprintf(stderr, "[INFO] Fragments: %d..%d / %d (total %d)\n",
                        frag_start, frag_end - 1, frag_mod, total_frags);
            }
        } else {
            fprintf(stderr, "[INFO] Architecture: 1 Reader from file/stdin, 1 GPU\n");
        }
    }

    FILE* out_fp = nullptr;
    if (g_out_path) {
        out_fp = fopen(g_out_path, "a");
        if (!out_fp) {
            fprintf(stderr, "[ERROR] Cannot open file for writing: %s\n", g_out_path);
            return 1;
        }
        if (!g_quiet) fprintf(stderr, "[INFO] Saving integral graphs to: %s\n", g_out_path);
    }

    JobQueue job_queue;
    job_queue.total_jobs = total_frags;

    std::thread producer;
    if (g_geng_path && frag_mod > 0) {
        producer = std::thread(producer_thread_fn, std::ref(job_queue), frag_start, frag_end, frag_mod);
    } else {
        FILE* fp = nullptr;
        if (input_path) fp = fopen(input_path, "r");
        else fp = stdin;
        if (!fp) { fprintf(stderr, "[ERROR] Odczyt pliku niemożliwy.\n"); return 1; }
        
        Job job = {0, -1, fp};
        job_queue.push(job);
        job_queue.finish();
    }

    std::vector<WorkerState> reader_states(num_readers);

    std::atomic<bool> monitor_running{true};
    std::thread monitor(monitor_thread_fn,
                        num_readers, reader_states.data(),
                        std::ref(job_queue),
                        std::ref(gpu_batches_done),
                        std::ref(global_processed),
                        std::ref(monitor_running));

    std::vector<std::thread> readers;
    readers.reserve(num_readers);
    for (int i = 0; i < num_readers; i++) {
        readers.emplace_back(
            reader_thread_fn, i,
            std::ref(job_queue),
            std::ref(pool),
            std::ref(reader_states[i]),
            std::ref(global_skipped)
        );
    }

    /* GPU thread */
    std::thread gpu(gpu_thread_fn,
                    std::ref(pool),
                    std::ref(result_mtx),
                    std::ref(integral_graphs),
                    std::ref(global_processed),
                    std::ref(gpu_batches_done),
                    out_fp);

    /* Graceful Shutdown */
    if (producer.joinable()) producer.join();
    for (auto& r : readers) r.join();
    
    gpu.join();
    
    monitor_running.store(false);
    monitor.join();

    if (out_fp) fclose(out_fp);

    pool.destroy();

    /* ============================================================
     * Final report
     * ============================================================ */
    double w_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_start).count();

    if (!g_quiet) {
        printf("\n==================================================\n");
        printf("  Integral graphs search results\n");
        printf("==================================================\n");
        printf("  Graph order (n):            %d\n", g_graph_n);
        printf("  Graphs processed:       %lld\n", (long long)global_processed.load());
        long long skips = global_skipped.load();
        printf("  Skipped (filter/error):    %lld\n", skips);
        printf("  Integral graphs found:   %zu\n", integral_graphs.size());
        printf("  Execution time:            %.3f s\n", w_time);
        if (w_time > 0) {
            printf("  Throughput:             %.0f graphs/s\n", (double)global_processed.load() / w_time);
        }
        printf("==================================================\n\n");
    }

    if (!g_quiet && integral_graphs.size() > 0) {
        printf("Integral graphs (graph6):\n");
        for (size_t i = 0; i < integral_graphs.size(); i++) {
            printf("  [%zu] %s\n", i + 1, integral_graphs[i].c_str());
        }
    }

    return 0;
}
