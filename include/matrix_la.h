#pragma once

#include "matrix.h"
#include <queue>
#include <vector>
#include <utility>
#include <Kokkos_Core.hpp>

// declarations

template <typename T>
struct MSA {
    static enum { UNALLOWED = 0, ALLOWED, SET } msa_states;
    char *state;
    T *value;
    size_t len;

    MSA(size_t n) {
        value = new T[n]();
        state = new char[n]();
        len = n;
    }

    ~MSA() {
        delete[] value;
        delete[] state;
    }
};

template <typename T>
spMtx<T> transpose(const spMtx<T> &A);

template <typename T>
denseMtx<T> transpose(const denseMtx<T> &A);

template <typename T>
void fuseEWiseMultAdd(const denseMtx<T> &A,
                      const denseMtx<T> &B,
                            denseMtx<T> &C);

template <typename T, typename U> 
void eWiseMult(const denseMtx<T> &A,
               const denseMtx<T> &B,
               const spMtx<U> &M,
               const denseMtx<T> &C);


template <typename T, typename U>
void mxmm_spd(const spMtx<T> &A,
              const denseMtx<T> &B,
              const spMtx<U> &M,
              denseMtx<T> &C,
              denseMtx<T> &Cbuf);

template <typename T, typename U>
void fuse_mxmm_eWiseMultAdd(const spMtx<T> &A,
                            const denseMtx<T> &W,
                            const spMtx<U> &M,
                            const denseMtx<T> &Numspd,
                            denseMtx<T> &Bcu);

template <typename T>
void add_nointersect(const spMtx<T> &A,
                     const spMtx<T> &B,
                     spMtx<T> &C,
                     spMtx<T> &Cbuf);

template <typename T>
spMtx<T> eWiseMult(const spMtx<T> &A,
                   const spMtx<T> &B);

template <typename T, typename U>
spMtx<T> eWiseMult(const spMtx<T> &A,
                   const spMtx<T> &B,
                   const spMtx<U> &M);

template <typename MatrixValT, typename ScalarT>
spMtx<MatrixValT> multScalar(const spMtx<MatrixValT> &A,
                             const ScalarT &alpha);

template<typename T, typename U>
spMtx<T> mxmm_mca(bool isParallel,
                  const spMtx<T> &A,
                  const spMtx<U> &B,
                  const spMtx<T> &M);

template<typename T, typename U>
void mxmm_mca(bool isParallel,
              const spMtx<T> &A,
              const spMtx<T> &B,
              const spMtx<U> &M,
              spMtx<T> &C);

template<typename T, typename U>
void _mxmm_mca_parallel(const spMtx<T> &A,
                        const spMtx<T> &B,
                        const spMtx<U> &M,
                        spMtx<T> &C);

template<typename T, typename U>
void _mxmm_mca_sequential(const spMtx<T> &A,
                          const spMtx<T> &B,
                          const spMtx<U> &M,
                          spMtx<T> &C);

template<typename T, typename U>
void mxmm_msa(bool isParallel,
              const spMtx<T> &A,
              const spMtx<T> &B,
              const spMtx<U> &M,
              spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_parallel(const spMtx<T> &A,
                        const spMtx<T> &B,
                        const spMtx<U> &M,
                        spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_sequential(const spMtx<T> &A,
                          const spMtx<T> &B,
                          const spMtx<U> &M,
                          spMtx<T> &C);

template<typename T, typename U>
void mxmm_msa_cmask(bool isParallel,
                    const spMtx<T> &A,
                    const spMtx<T> &B,
                    const spMtx<U> &M,
                    spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_cmask_parallel(const spMtx<T> &A,
                              const spMtx<T> &B,
                              const spMtx<U> &M,
                              spMtx<T> &C);

template<typename T, typename U>
void _mxmm_msa_cmask_sequential(const spMtx<T> &A,
                                const spMtx<T> &B,
                                const spMtx<U> &M,
                                spMtx<T> &C);

template <typename T>
void mxmm_naive(bool isParallel,
                const spMtx<T> &A,
                const spMtx<T> &B,
                const spMtx<T> &M,
                spMtx<T> &C);


// definitions


template <typename T>
spMtx<T> transpose(const spMtx<T> &A) {
    spMtx<T> AT(A.n, A.m, A.nz);

    // filling the column indices array and current column positions array
    for (size_t i = 0; i < A.nz; ++i)
        ++AT.Rst(A.Col(i)+1);
    for (size_t i = 0; i < AT.m; ++i)
        AT.Rst(i+1) += AT.Rst(i);

    // transposing
    for (size_t i = 0; i < A.m; ++i) {
        for (int j = A.Rst(i); j < A.Rst(i+1); ++j) {
            AT.Val(AT.Rst(A.Col(j))) = std::move(A.Val(j));
            AT.Col(AT.Rst(A.Col(j))++) = i;
        }
    }
    // set Rst indices to normal state
    // AT.Rst(AT.m) already has the correct value
    for (int i = AT.m - 1; i > 0; --i)
        AT.Rst(i) = AT.Rst(i-1);
    AT.Rst(0) = 0;

    return AT;
}

// C += A .* B
template <typename T>
void fuseEWiseMultAdd(const denseMtx<T> &A, const denseMtx<T> &B, denseMtx<T> &C) {
#pragma omp parallel for simd schedule(static, 4096)
    for (size_t i = 0; i < A.m * A.n; ++i)
        C.Val[i] += A.Val[i] * B.Val[i];
}

// C<M> = A .* B
template <typename T, typename U>
void eWiseMult(const denseMtx<T> &A, const denseMtx<T> &B, const spMtx<U> &M, denseMtx<T> &C) {
    const T zero = T(0);
// #pragma omp parallel for simd

    parallel_for("EWiseMult", C.m, KOKKOS_LAMBDA(size_t i) {
        for (size_t j = 0; j < C.n; ++j) {
            C.Val(i * C.n + j) = zero;
        }
    });


    parallel_for("EWiseMult",
                 RangePolicy<Schedule<Dynamic>>(0, M.m, ChunkSize(256)),
                 KOKKOS_LAMBDA(size_t i) {
        for (size_t j = M.Rst[i]; j < M.Rst[i+1]; ++j) {
            size_t idx = C.n * i + M.Col[j];
            C.Val[idx] = A.Val[idx] * B.Val[idx];
        }
    });
}

template <typename T>
denseMtx<T> transpose(const denseMtx<T> &A) {
    denseMtx<T> AT(A.n, A.m);
    size_t block_size = 64;

    for (size_t i = 0; i < A.m; i += block_size) {
        for (size_t j = 0; j < A.n; j += block_size) {
            size_t pmax = std::min(A.m, i + block_size);
            size_t qmax = std::min(A.n, j + block_size);
            for (size_t p = i; p < pmax; ++p)
                for (size_t q = j; q < qmax; ++q)
                    AT.Val[AT.n * q + p] = A.Val[A.n * p + q];
        }
    }

    return AT;
}

// C<M> = A * B
// TODO ^T !!!!!!!!!!!!!!!!
template <typename T, typename U>
void mxmm_spd(const spMtx<T> &A, const denseMtx<T> &B, const spMtx<U> &M, denseMtx<T> &C, denseMtx<T> &Cbuf) {
    const T zero = T(0);
    // denseMtx<T> BT = transpose(B);

    Kokkos::parallel_for("MaskedSparseDenseGEMM",
                         Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, C.m, Kokkos::ChunkSize(256)),
                         KOKKOS_LAMBDA(size_t i) {
        // for (size_t q = M.Rst[i]; q < M.Rst[i+1]; ++q) {
        //     size_t j = M.Col[q];
        //     T *b_row = BT.Val + BT.n * j;
        //     T dotpr = zero;
        //     for (size_t k = A.Rst[i]; k < A.Rst[i+1]; ++k) {
        //         dotpr += A.Val[k] * b_row[A.Col[k]];
        //     }
        //     Ccopy.Val[C.n * i + j] = dotpr;
        // }
        for (size_t j = 0; j < C.n; ++j)
            Cbuf.Val(i * Cbuf.n + j) = zero;
        for (size_t q = A.Rst(i); q < A.Rst(i+1); ++q) {
            size_t j = A.Col(q);
            T  a_val = A.Val(q);
        // #pragma omp simd
            for (size_t k = M.Rst(i); k < M.Rst(i+1); ++k)
                Cbuf.Val(i * Cbuf.n + M.Col(k)) += a_val * B.Val(j * B.n + M.Col(k));
        }
    });
    std::swap(C.Val, Cbuf.Val);
}

template <typename T, typename U>
void fuse_mxmm_eWiseMultAdd(const spMtx<T> &A, const denseMtx<T> &W, const spMtx<U> &M,
                            const denseMtx<T> &Numspd, denseMtx<T> &Bcu) {
    const T zero = T(0);

#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < M.m; ++i) {
        for (size_t q = A.Rst[i]; q < A.Rst[i+1]; ++q) {
            size_t j = A.Col[q];
            T  a_val = A.Val[q];
            T *w_row = W.Val + W.n * j;
            T *bcu_row = Bcu.Val + Bcu.n * j;
            T *numspd_row = Numspd.Val + Numspd.n * j;
        #pragma omp simd
            for (size_t k = M.Rst[i]; k < M.Rst[i+1]; ++k)
                bcu_row[M.Col[k]] += a_val * w_row[M.Col[k]] * numspd_row[M.Col[k]];
        }
    }

//#pragma omp parallel for
//    for (size_t i = 0; i < W.m; ++i) {
//        T *w_row = W.Val + W.n * i;
//        for (size_t j = 0; j < W.n; ++j)
//            w_row[j] = zero;
//    }
}

template <typename T>
void add_nointersect(const spMtx<T> &A, const spMtx<T> &B, spMtx<T> &C, spMtx<T> &Cbuf) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    Cbuf.resizeRows(A.m);
    for (size_t i = 0; i <= A.m; ++i)
        Cbuf.Rst(i) = A.Rst(i) + B.Rst(i);
    Cbuf.resizeVals(Cbuf.Rst(C.m));

    Kokkos::parallel_for("SparseAddNointersect",
                         RangePolicy<Schedule<Dynamic>>(0, C.m, ChunkSize(256)),
                         KOKKOS_LAMBDA(size_t i) {
        int aIdx = A.Rst(i), bIdx = B.Rst(i), cIdx;
        for (cIdx = Cbuf.Rst(i); aIdx < A.Rst(i+1) && bIdx < B.Rst(i+1); ++cIdx) {
            if (A.Col(aIdx) < B.Col(bIdx)) {
                Cbuf.Col(cIdx) = A.Col(aIdx);
                Cbuf.Val(cIdx) = A.Val(aIdx++);
            }
            else {
                Cbuf.Col(cIdx) = B.Col(bIdx);
                Cbuf.Val(cIdx) = B.Val(bIdx++);
            }
        }
        if (aIdx < A.Rst[i+1]) {
            for (int j = 0; j < Cbuf.Rst[i+1] - cIdx; ++j) {
                Cbuf.Col(cIdx + j) = A.Col(aIdx + j);
                Cbuf.Val(cIdx + j) = A.Val(aIdx + j);
            }
        }
        else {
            for (int j = 0; j < Cbuf.Rst[i+1] - cIdx; ++j) {
                Cbuf.Col(cIdx + j) = B.Col(aIdx + j);
                Cbuf.Val(cIdx + j) = B.Val(aIdx + j);
            }
        }
    });

    std::swap(C, Cbuf);
}

template <typename T>
spMtx<T> eWiseMult(const spMtx<T> &A, const spMtx<T> &B) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMtx<T> C(A.m, A.n);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        int aMax = A.Rst[i+1], bMax = B.Rst[i+1];
        int colCnt = 0;

        while (aIdx < aMax && bIdx < bMax) {
            if (A.Col[aIdx] == B.Col[bIdx])
                ++aIdx, ++bIdx, ++colCnt;
            while (aIdx < aMax && A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            while (bIdx < bMax && B.Col[bIdx] < A.Col[aIdx])
                ++bIdx;
        }
        C.Rst[i+1] = colCnt;
    }

    for (size_t i = 0; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.resizeVals(C.Rst[C.m]);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        int aMax = A.Rst[i+1], bMax = B.Rst[i+1];

        for (int cIdx = C.Rst[i]; cIdx < C.Rst[i+1]; ++i) {
            if (A.Col[aIdx] == B.Col[bIdx]) {
                C.Col[cIdx] = A.Col[aIdx];
                C.Val[cIdx] = A.Val[aIdx++] * B.Val[bIdx++];
            }
            while (aIdx < aMax && A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            while (bIdx < bMax && B.Col[bIdx] < A.Col[aIdx])
                ++bIdx;
        }
    }
    return C;
}


template <typename T, typename U>
spMtx<T> eWiseMult(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M) {
    if (A.m != B.m || A.n != B.n)
        throw -1;

    spMtx<T> C(A.m, A.n);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i];
        int aMax = A.Rst[i+1], bMax = B.Rst[i+1], mMax = M.Rst[i+1];
        int colCnt = 0;

        for (int mIdx = M.Rst[i]; mIdx < mMax; ++mIdx) {
            if (A.Col[aIdx] == B.Col[bIdx] && B.Col[bIdx] == M.Col[mIdx]) {
                ++aIdx;
                ++bIdx;
                ++mIdx;
                ++colCnt;
            }
            while (aIdx < aMax && A.Col[aIdx] < M.Col[mIdx])
                ++aIdx;
            while (bIdx < bMax && B.Col[bIdx] < M.Col[mIdx])
                ++bIdx;
        }
        C.Rst[i+1] = colCnt;
    }

    for (size_t i = 0; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    C.resizeVals(C.Rst[C.m]);
    
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < A.m; ++i) {
        int aIdx = A.Rst[i], bIdx = B.Rst[i], mIdx = M.Rst[i];
        for (int j = C.Rst[i]; j < C.Rst[i+1]; ++j) {
            if (A.Col[aIdx] == B.Col[bIdx] && B.Col[bIdx] == M.Col[mIdx]) {
                C.Col[j] = A.Col[aIdx];
                C.Val[j] = A.Val[aIdx++] * B.Val[bIdx++];
            } else if (A.Col[aIdx] < B.Col[bIdx])
                ++aIdx;
            else
                ++bIdx;
        }
    }

    return C;
}

template <typename MatrixValT, typename ScalarT>
spMtx<MatrixValT> multScalar(const spMtx<MatrixValT> &A, const ScalarT &alpha) {
#pragma omp parallel for
    for (size_t i = 0; i < A.nz; ++i)
        A.Val[i] *= alpha;
    return A;
}

/*MSpGEMM ù ùùùùùùùùùùùùùù ùùùùùùù (ùùùùùùùùùùù) ùùùùùùùùùùùù*/
template <typename T>
struct MCA {
    static enum {ALLOWED = 0, SET} mca_states;
    // char   *states;
    T      *values;
    size_t  len;

    MCA(size_t n) {
        values = new T[n];
        // states = new char[n];
           len = n;

        std::memset(values, 0, len * sizeof(T));
        // std::memset(states, 0, len * sizeof(char));
    }

    ~MCA() {
        delete[] values;
        // delete[] states;
    }

    inline void clear() {
        std::memset(values, 0, len * sizeof(T));
        // std::memset(states, 0, len * sizeof(char));
    }
};

template<typename T, typename U>
spMtx<T> mxmm_mca(bool isParallel, const spMtx<T> &A, const spMtx<U> &B, const spMtx<T> &M) {
    // ùùùùùùùùùùùùù C
    spMtx<T> C(A.m, B.n, M.nz);
    memcpy(C.Col, M.Col, M.nz * sizeof(int));
    memcpy(C.Rst, M.Rst, (M.m + 1) * sizeof(int));

    if (isParallel)
        _mxmm_mca_parallel(A, B, M, C);
    else
        _mxmm_mca_sequential(A, B, M, C);

    return C;
}

template<typename T, typename U>
void mxmm_mca(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    // ùùùùùùùùùùùùù C
    C.resizeRows(M.m);
    C.resizeVals(M.nz);
    C.n = M.n;
    Kokkos::deep_copy(C.Col, M.Col);
    Kokkos::deep_copy(C.Rst, M.Rst);

    if (isParallel)
        _mxmm_mca_parallel(A, B, M, C);
    else
        _mxmm_mca_sequential(A, B, M, C);
}

template<typename T, typename U>
void _mxmm_mca_parallel(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    int mca_len = 0;
    for (size_t i = 0; i < A.m; ++i)
        if (M.Rst(i+1) - M.Rst(i) > mca_len)
            mca_len = M.Rst(i+1) - M.Rst(i);

#pragma omp parallel
    {
        MCA<T> accum(mca_len);

#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            int m_row_len = M.Rst(i+1) - M.Rst(i);
            int m_pos;

            for (int t = A.Rst(i); t < A.Rst(i+1); ++t) {
                int k = A.Col(t);
                int b_pos = B.Rst(k);
                int b_max = B.Rst(k+1);
                T   a_val = A.Val(t);
                m_pos = M.Rst(i);
                for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                    while (b_pos < b_max && B.Col(b_pos) < M.Col(m_pos))
                        ++b_pos;
                    if (b_pos < b_max && B.Col(b_pos) == M.Col(m_pos))
                        accum.values[j] += a_val * B.Val(b_pos);
                }
            }

            for (size_t j = 0; j < m_row_len; ++j)
                C.Val(C.Rst(i) + j) = accum.values[j];
            memset(accum.values, 0, mca_len * sizeof(T));
        }
    }
}

template<typename T, typename U>
void _mxmm_mca_sequential(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    int mca_len = 0;
    for (size_t i = 0; i < A.m; ++i)
        if (M.Rst(i+1) - M.Rst(i) > mca_len)
            mca_len = M.Rst(i+1) - M.Rst(i);

    MCA<T> accum(mca_len);

    for (size_t i = 0; i < A.m; ++i) {
        int m_row_len = M.Rst(i+1) - M.Rst(i);
        int m_pos;

        for (int t = A.Rst(i); t < A.Rst(i+1); ++t) {
            int k = A.Col(t);
            int b_pos = B.Rst(k);
            int b_max = B.Rst(k+1);
            T   a_val = A.Val(t);
            m_pos = M.Rst(i);
            for (int j = 0; j < m_row_len; ++j, ++m_pos) {
                while (b_pos < b_max && B.Col(b_pos) < M.Col(m_pos))
                    ++b_pos;
                if (b_pos < b_max && B.Col(b_pos) == M.Col(m_pos))
                    accum.values[j] += a_val * B.Val(b_pos);
            }
        }
        for (size_t j = 0; j < m_row_len; ++j)
            C.Val(C.Rst(i) + j) = accum.values[j];
        memset(accum.values, 0, mca_len * sizeof(T));
    }
}

template<typename T, typename U>
void mxmm_msa(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    C.resizeRows(M.m);
    C.resizeVals(M.nz);
    C.n = M.n;
    Kokkos::deep_copy(C.Col, M.Col);
    Kokkos::deep_copy(C.Rst, M.Rst);

    if (isParallel)
        _mxmm_msa_parallel(A, B, M, C);
    else
        _mxmm_msa_sequential(A, B, M, C);
}

template<typename T, typename U>
void _mxmm_msa_parallel(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    // ScratchPadView accum("Accum", total_threads, B.n);

    // DEBUG
    // std::cout << "A:\n";
    // A.print_dense();
    // std::cout << "B:\n";
    // B.print_dense();
    // std::cout << "M:\n";
    // M.print_dense();
    // std::cout << "C:\n";
    // C.print_dense();
    // C.print_crs();
    const size_t num_of_teams = Kokkos::num_threads();
    View<int*> work_distribution("", num_of_teams+1);
    size_t team_i = 1;
    for (size_t i = 0; i <= A.m; ++i) {
        if (A.Rst[i] >= A.nz * team_i / num_of_teams)
            work_distribution(team_i++) = i;
    }
    
    using ScratchPadView =
        View<T*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    int level = 0;
    TeamPolicy<DefaultExecutionSpace, Schedule<Dynamic>> policy(num_of_teams, Kokkos::AUTO, 64);
    policy = policy.set_scratch_size(level, Kokkos::PerThread(ScratchPadView::shmem_size(B.n)));
    // std::cout << ScratchPadView::shmem_size(B.n) << " bytes for accumulator\n";
    parallel_for("MSpGEMM_MSA", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const size_t team_id = team_member.league_rank();
        // std::string out = std::to_string(team_id) + " size: " + std::to_string(team_member.team_size()) + "\n";
        // std::cout << out;

        const size_t team_begin = work_distribution(team_id);
        const size_t team_end   = work_distribution(team_id + 1);
        ScratchPadView accum(team_member.thread_scratch(level), B.n);
        parallel_for(Kokkos::TeamThreadRange(team_member, team_begin, team_end), KOKKOS_LAMBDA(const int i) {
            const T zero = T(0);
            const int m_beg = M.Rst(i);
            const int m_end = M.Rst(i+1);
            for (int j = m_beg; j < m_end; ++j)
                accum(M.Col(j)) = zero;
            for (int t = A.Rst(i); t < A.Rst(i+1); ++t) {
                const int k = A.Col(t);
                const int b_beg = B.Rst(k);
                const int b_end = B.Rst(k+1);
                T   a_val = A.Val(t);
                parallel_for(Kokkos::ThreadVectorRange(team_member, b_beg, b_end), KOKKOS_LAMBDA(const int j) {
                    accum(B.Col(j)) += a_val * B.Val(j);
                });
                // for (int j = b_beg; j < b_end; ++j)
                //     accum(B.Col(j)) += a_val * B.Val(j);
            }
            parallel_for(Kokkos::ThreadVectorRange(team_member, m_beg, m_end), KOKKOS_LAMBDA(const int j) {
                C.Val(j) = accum(M.Col(j));
            });
            // for (int j = m_beg; j < m_end; ++j)
            //     C.Val(j) = accum(M.Col(j));
        });
    });

    // parallel_for("MSpGEMM_MSA", RangePolicy<Schedule<Dynamic>>(0, C.m, ChunkSize(64)).set_scratch_size(), KOKKOS_LAMBDA(size_t i) {
    // 
    //     const T zero = T(0);
    //     int m_min = M.Rst(i);
    //     int m_max = M.Rst(i+1);
    //     for (int j = m_min; j < m_max; ++j)
    //         accum(thread_id, M.Col(j)) = zero;
    //     for (int t = A.Rst(i); t < A.Rst(i+1); ++t) {
    //         int k = A.Col(t);
    //         int b_pos = B.Rst(k);
    //         int b_max = B.Rst(k+1);
    //         T   a_val = A.Val(t);
    //         for (int j = b_pos; j < b_max; ++j)
    //             accum(B.Col(j)) += a_val * B.Val(j);
    //     }
    //     for (int j = m_min; j < m_max; ++j)
    //         C.Val(j) = accum(thread_id, M.Col(j));
    // });
    // 
    // Kokkos::parallel_for("MSpGEMM_MSA", Kokkos::RangePolicy<>(0, total_threads), KOKKOS_LAMBDA(size_t thread_id) {
    //     const T zero = T(0);
    //     size_t begin = A.m * thread_id / total_threads;
    //     size_t end = A.m * (thread_id + 1) / total_threads;
    // 
    //     // DEBUG
    //     // std::string output("# " + std::to_string(thread_id) + " entered, job: [" +
    //     //                    std::to_string(begin) + "; " + std::to_string(end) + ")\n");
    //     // std::cout << output;
    //     for (size_t i = begin; i < end; ++i) {
    //         int m_min = M.Rst(i);
    //         int m_max = M.Rst(i+1);
    //         for (int j = m_min; j < m_max; ++j)
    //             accum(thread_id, M.Col(j)) = zero;
    //         for (int t = A.Rst(i); t < A.Rst(i+1); ++t) {
    //             int k = A.Col(t);
    //             int b_pos = B.Rst(k);
    //             int b_max = B.Rst(k+1);
    //             T   a_val = A.Val(t);
    //             for (int j = b_pos; j < b_max; ++j)
    //                 accum(thread_id, B.Col(j)) += a_val * B.Val(j);
    //         }
    //         for (int j = m_min; j < m_max; ++j)
    //             C.Val(j) = accum(thread_id, M.Col(j));
    // 
    //         // DEBUG
    //         // C.print_crs();
    //         // output = "Row " + std::to_string(i) + ": ";
    //         // std::string out2 = "Accum " + std::to_string(thread_id) + ": (";
    //         // for (int j = 0; j < B.n; ++j)
    //         //     out2 += std::to_string(accum(thread_id, j)) + " ";
    //         // out2 += ")\n";
    //         // for (int j = m_min; j < m_max; ++j)
    //         //     output += "(" + std::to_string(C.Col(j)) + ", " +  std::to_string(C.Val(j)) + "), ";
    //         // output += "\n" + out2;
    //         // std::cout << output;
    //     }
    // });
}

template<typename T, typename U>
void _mxmm_msa_sequential(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    MSA<T> accum(B.n);

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];

        // ùùùùùùùùù ùùùùùùùùùù ùùùùùùùùù ùùùùùùùùùùùù
        for (int j = m_min; j < m_max; ++j)
            accum.state[M.Col[j]] = MSA<T>::ALLOWED;

        // ùùùùùùù i-ù ùùùùùù ùùùùùùù C
        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];

            for (int j = b_pos; j < b_max; ++j) {
                int b_col = B.Col[j];
                if (accum.state[b_col] == MSA<T>::ALLOWED) {
                    accum.state[b_col] = MSA<T>::SET;
                    accum.value[b_col] = a_val * B.Val[j];
                } else if (accum.state[b_col] == MSA<T>::SET)
                    accum.value[b_col] += a_val * B.Val[j];
            }
        }

        // ùùùùùùùùùù ùùùùùù ùùùùùùù C ù ùùùùùùù ùùùùùùùùùùùù
        for (int j = m_min; j < m_max; ++j) {
            C.Val[j] = accum.value[M.Col[j]];
            accum.state[M.Col[j]] = MSA<T>::UNALLOWED;
        }
    }
}

template<typename T, typename U>
void mxmm_msa_cmask(bool isParallel, const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    // ùùùùùùùùùùùùù C
    C.resizeRows(M.m);
    C.n = M.n;

    if (isParallel)
        _mxmm_msa_cmask_parallel(A, B, M, C);
    else
        _mxmm_msa_cmask_sequential(A, B, M, C);
}

template<typename T, typename U>
void _mxmm_msa_cmask_parallel(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
#pragma omp parallel
    {
        MSA<T> accum(B.n);
        std::vector<int> changed_states;
        changed_states.reserve(B.n);
        
#pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < A.m; ++i) {
            int m_begin = M.Rst[i];
            int m_end   = M.Rst[i+1];
            int row_nz = 0;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_begin = B.Rst[k];
                int b_end   = B.Rst[k+1];

                for (int j = b_begin; j < b_end; ++j) {
                    int col = B.Col[j];
                    if (accum.state[col] == MSA<T>::UNALLOWED) {
                        accum.state[col] = MSA<T>::ALLOWED;
                        changed_states.push_back(col);
                        ++row_nz;
                    }
                }
            }
            for (int j = m_begin; j < m_end; ++j) {
                // OPTIMIZATION 1: GET RID OF IF STATEMENT
                row_nz -= accum.state[M.Col[j]];
                // if (accum.state[M.Col[j]] == MSA<T>::ALLOWED)
                //     --row_nz;
            }
            C.Rst[i+1] = row_nz;
            
            for (int col_idx: changed_states)
                accum.state[col_idx] = MSA<T>::UNALLOWED;
            changed_states.clear();
        }
#pragma omp single
    {
        C.Rst[0] = 0;
        for (int i = 1; i < A.m; ++i)
            C.Rst[i+1] += C.Rst[i];
        if (C.Rst[A.m] > C.nz)
            C.resizeVals(C.Rst[A.m]);
        C.nz = C.Rst[A.m];
    }

    constexpr T zero = T(0);
    for (size_t i = 0; i < accum.len; ++i)
        accum.state[i] = MSA<T>::ALLOWED;

#pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < A.m; ++i) {
            int m_begin = M.Rst[i];
            int m_end   = M.Rst[i+1];

            for (size_t j = m_begin; j < m_end; ++j)
                accum.state[M.Col[j]] = MSA<T>::UNALLOWED;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_begin = B.Rst[k];
                int b_end   = B.Rst[k+1];
                T   a_val = A.Val[t];

            #pragma omp simd
                for (int j = b_begin; j < b_end; ++j) {
                    int col = B.Col[j];
                    // if (accum.state[col] == MSA<T>::ALLOWED) {
                    //     accum.state[col] = MSA<T>::SET;
                    //     changed_states.push_back(col);
                    // }

                    accum.state[col] = MSA<T>::SET;

                    accum.value[col] += a_val * B.Val[j];
                }
            }
            for (size_t j = m_begin; j < m_end; ++j) {
                accum.state[M.Col[j]] = MSA<T>::ALLOWED;
                accum.value[M.Col[j]] = zero;
            }
            
            int c_pos = C.Rst[i];
            for (int i = 0; i < accum.len; ++i) {
                if (accum.state[i] == MSA<T>::SET) {
                    C.Col[c_pos] = i;
                    C.Val[c_pos++] = accum.value[i];
                    accum.state[i] = MSA<T>::ALLOWED;
                    accum.value[i] = zero;
                }
            }

            // sort(changed_states.begin(), changed_states.end());
            // for (int col_idx : changed_states) {
            //     C.Col[c_pos] = col_idx;
            //     C.Val[c_pos++] = accum.value[col_idx];
            //     accum.state[col_idx] = MSA<T>::ALLOWED;
            //     accum.value[col_idx] = zero;
            // }
            // changed_states.clear();
        }
    }
}

template<typename T, typename U>
void _mxmm_msa_cmask_sequential(const spMtx<T> &A, const spMtx<T> &B, const spMtx<U> &M, spMtx<T> &C) {
    MSA<T> accum(B.n);
    std::vector<int> changed_states;
    changed_states.reserve(B.n);

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];
        int row_nz = 0;

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_min = B.Rst[k];
            int b_max = B.Rst[k+1];

            for (int j = b_min; j < b_max; ++j) {
                if (accum.state[B.Col[j]] == MSA<T>::UNALLOWED) {
                    accum.state[B.Col[j]] = MSA<T>::ALLOWED;
                    changed_states.push_back(B.Col[j]);
                    ++row_nz;
                }
            }
        }
        for (int j = m_min; j < m_max; ++j) {
            if (accum.state[M.Col[j]] == MSA<T>::ALLOWED)
                --row_nz;
        }
        C.Rst[i+1] = row_nz;
        
        for (int col_idx: changed_states)
            accum.state[col_idx] = MSA<T>::UNALLOWED;
        changed_states.clear();
    }
    C.Rst[0] = 0;
    for (int i = 1; i < A.m; ++i)
        C.Rst[i+1] += C.Rst[i];
    if (C.Rst[A.m] > C.nz)
        C.resizeVals(C.Rst[A.m]);
    C.nz = C.Rst[A.m];

    constexpr T zero = T(0);
    for (size_t i = 0; i < accum.len; ++i)
        accum.state[i] = MSA<T>::ALLOWED;

    for (size_t i = 0; i < A.m; ++i) {
        int m_min = M.Rst[i];
        int m_max = M.Rst[i+1];

        for (size_t j = m_min; j < m_max; ++j)
            accum.state[M.Col[j]] = MSA<T>::UNALLOWED;

        for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
            int k = A.Col[t];
            int b_pos = B.Rst[k];
            int b_max = B.Rst[k+1];
            T   a_val = A.Val[t];

            for (int j = b_pos; j < b_max; ++j) {
                if (accum.state[B.Col[j]] == MSA<T>::ALLOWED) {
                    accum.state[B.Col[j]] = MSA<T>::SET;
                    changed_states.push_back(B.Col[j]);
                    accum.value[B.Col[j]] = a_val * B.Val[j];
                }
                else if (accum.state[B.Col[j]] == MSA<T>::SET)
                    accum.value[B.Col[j]] += a_val * B.Val[j];
            }
        }
        
        int c_pos = C.Rst[i];
        sort(changed_states.begin(), changed_states.end());
        for (int col_idx : changed_states) {
            C.Col[c_pos] = col_idx;
            C.Val[c_pos++] = accum.value[col_idx];
            accum.value[col_idx] = zero;
            accum.state[col_idx] = MSA<T>::ALLOWED;
        }
        changed_states.clear();
        for (size_t j = m_min; j < m_max; ++j)
            accum.state[M.Col[j]] = MSA<T>::ALLOWED;
    }
}

template <typename T>
void mxmm_naive(bool isParallel, const spMtx<T> &A, const spMtx<T> &B,
                                 const spMtx<T> &M, spMtx<T> &C) {
    // ùùùùùùùùùùùùù C
    C.m = A.m;
    C.resizeRows(A.m);
    C.Rst(0) = 0;

    // ùùùùùùùùùù ùùùùùù
    if (isParallel == true) {
#pragma omp parallel
        {
            int count;
            char *is_set = new char[A.m]();
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < A.m; ++i) {
                count = 0;
                for (int k = A.Rst(i); k < A.Rst(i+1); ++k)
                    for (int j = B.Rst(A.Col(k)); j < B.Rst(A.Col(k) + 1); ++j)
                        is_set[B.Col(j)] = 1;
                for (int k = 0; k < A.m; ++k)
                    if (is_set[k])
                        ++count;
                memset(is_set, 0, A.m*sizeof(char));
                C.Rst(i+1) = count;
            }
            delete[] is_set;
        }
    } else {
        int count = 0;
        char *is_set = new char[A.m]();
        for (size_t i = 0; i < A.m; ++i) {
            for (int k = A.Rst(i); k < A.Rst(i+1); ++k)
                for (int j = B.Rst(A.Col(k)); j < B.Rst(A.Col(k) + 1); ++j)
                    is_set[B.Col(j)] = 1;
            for (int k = 0; k < A.m; ++k)
                if (is_set[k])
                    ++count;
            C.Rst[i+1] = count;
            memset(is_set, 0, A.m*sizeof(char));
        }
        Kokkos::resize(C.Col, C.Rst(C.m));
        Kokkos::resize(C.Val, C.Rst(C.m));
    }

    // ùùùùùùùùù ùùùùùù
    if (isParallel == true) {
#pragma omp parallel
        {
            T *rowpr = new T[A.m]();
            char *is_set = new char[A.m]();
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < A.m; ++i) {
                int c_curr = C.Rst(i);
                for (int k = A.Rst(i); k < A.Rst(i+1); ++k) {
                    for (int j = B.Rst(A.Col(k)); j < B.Rst(A.Col(k) + 1); ++j) {
                        is_set[B.Col(j)] = 1;
                        rowpr[B.Col(j)] += A.Val(k) * B.Val(j);
                    }
                }
                for (int k = 0; k < A.m; ++k) {
                    if (is_set[k]) {
                        C.Col(c_curr) = k;
                        C.Val(c_curr++) = rowpr[k];
                    }
                }
                memset(is_set, 0, A.m*sizeof(char));
                memset(rowpr, 0, A.m*sizeof(T));
            }
            delete[] rowpr;
            delete[] is_set;
        }
    } else {
        T *rowpr = new T[A.m]();
        char *is_set = new char[A.m]();
        for (size_t i = 0; i < A.m; ++i) {
            int c_curr = C.Rst(i);
            for (int k = A.Rst(i); k < A.Rst(i+1); ++k) {
                for (int j = B.Rst(A.Col(k)); j < B.Rst(A.Col(k) + 1); ++j) {
                    is_set[B.Col(j)] = 1;
                    rowpr[B.Col(j)] += A.Val(k) * B.Val(j);
                }
            }
            for (int k = 0; k < A.m; ++k) {
                if (is_set[k]) {
                    C.Col(c_curr) = k;
                    C.Val(c_curr++) = rowpr[k];
                }
            }
            memset(is_set, 0, A.m*sizeof(char));
            memset(rowpr, 0, A.m*sizeof(T));
        }
        delete[] rowpr;
        delete[] is_set;
    }

    // ùùùùùùùùùù ùùùùù
    T *c_wgt_new = new T[M.nz]();
    int *c_adj_new = new int[M.nz];
    for (int i = 0; i < M.nz; ++i)
        c_adj_new[i] = M.Col(i);
    
    int c_curr;
    if (isParallel == true) {
#pragma omp parallel for private(c_curr) schedule(dynamic)
        for (size_t i = 0; i < A.m; ++i) {
            c_curr = C.Rst(i);
            for (int j = M.Rst(i); j < M.Rst(i+1); ++j) {
                while (c_curr < C.Rst(i+1) && C.Col(c_curr) < M.Col(j))
                    ++c_curr;
                if (c_curr < C.Rst(i+1) && C.Col(c_curr) == M.Col(j))
                    c_wgt_new[j] = C.Val(c_curr++);
            }
        }
    } else {
        for (size_t i = 0; i < A.m; ++i) {
            c_curr = C.Rst(i);
            for (int j = M.Rst(i); j < M.Rst(i+1); ++j) {
                while (c_curr < C.Rst(i+1) && C.Col(c_curr) < M.Col(j))
                    ++c_curr;
                if (c_curr < C.Rst(i+1) && C.Col(c_curr) == M.Col(j))
                    c_wgt_new[j] = C.Val(c_curr++);
            }
        }
    }

    for (int i = 0; i < M.nz; ++i) {
        C.Val(i) = c_wgt_new[i];
        C.Col(i) = c_adj_new[i];
    }
    Kokkos::deep_copy(C.Rst, M.Rst);
    C.nz = C.Rst(C.m);
}
