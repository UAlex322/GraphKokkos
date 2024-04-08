#pragma once
#include "matrix.h"
#include "matrix_la.h"

// std::vector<float> betweenness_centrality(bool isParallel, const spMtx<int> &A, size_t blockSize) {
//     if (A.m != A.n)
//         throw "non-square matrix; BC is only for square matrices";
// 
//     size_t m = A.m;
//     std::vector<float> bcv(A.m);
//     float *bc = bcv.data();
//     std::vector<spMtx<int>> Sigmas(A.m);
//     spMtx<int> AT = transpose(A);
//     spMtx<int> Front;
//     spMtx<int> Fronttmp;
//     spMtx<int> Numsp(m, 0);
//     spMtx<float> Af = convertType<float>(A);
//     denseMtx<float> Numspd;
//     denseMtx<float> Bcu;
//     denseMtx<float> Nspinv;
//     denseMtx<float> W;
//     denseMtx<float> Wbuf;
// 
//     std::chrono::high_resolution_clock::time_point 
//         eWiseAdd_begin,
//         spMul_begin,
//         eWiseMult_begin,
//         dMul_begin,
//         eWiseMultAdd_begin,
//         eWiseAdd_end,
//         spMul_end,
//         eWiseMult_end,
//         dMul_end,
//         eWiseMultAdd_end;
//     long long eWiseAdd_time = 0,
//               spMul_time = 0,
//               eWiseMult_time = 0,
//               dMul_time = 0,
//               eWiseMultAdd_time = 0;
// 
//     for (size_t i = 0; i < A.n; i += blockSize) {
//         size_t n = std::min(A.n - i, blockSize); //                    
//         size_t mxn = (size_t)m * n;
//         Numsp.resizeVals(n);
//         Wbuf.resize(A.m, n);
//         Numsp.n = n;
//         for (size_t j = 0; j < i; ++j)
//             Numsp.Rst[j] = 0;
//         for (size_t j = 0; j < n; ++j) {
//             Numsp.Rst[i+j] = j;
//             Numsp.Col[j] = j;
//             Numsp.Val[j] = 1;
//         }
//         for (size_t j = i+n; j <= m; ++j)
//             Numsp.Rst[j] = n;
//         Front = transpose(A.extractRows(i, i+n));
// 
//         //               (              )
//         size_t d = 0;
//         do {
//             Sigmas[d] = Front;
// 
//             eWiseAdd_begin = std::chrono::high_resolution_clock::now();
//             Numsp = add_nointersect(Numsp, Front);
//             eWiseAdd_end = std::chrono::high_resolution_clock::now();
//             eWiseAdd_time += (eWiseAdd_end - eWiseAdd_begin).count();
// 
//             spMul_begin = std::chrono::high_resolution_clock::now();
//             mxmm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
//             spMul_end = std::chrono::high_resolution_clock::now();
//             spMul_time += (spMul_end - spMul_begin).count();
// 
//             Front = Fronttmp;
//             ++d;
//         } while (Front.nz != 0);
// 
//         //                
//         Numspd = Numsp; //                                             
//         Nspinv.resize(m, n);
//         Bcu.resize(m, n);
//         W.resize(m, n);
//         for (size_t i = 0; i < mxn; ++i)
//             Bcu.Val[i] = 1.0f;
//         for (size_t i = 0; i < mxn; ++i)
//             Nspinv.Val[i] = 1.0f / Numspd.Val[i];
//         for (size_t k = d-1; k > 0; --k) {
//             eWiseMult_begin = std::chrono::high_resolution_clock::now();
//             eWiseMult(Nspinv, Bcu, Sigmas[k], W);
//             eWiseMult_end = std::chrono::high_resolution_clock::now();
//             eWiseMult_time += (eWiseAdd_end - eWiseAdd_begin).count();
// 
//             dMul_begin = std::chrono::high_resolution_clock::now();
//             mxmm_spd(Af, W, Sigmas[k-1], W, Wbuf);
//             dMul_end = std::chrono::high_resolution_clock::now();
//             dMul_time += (dMul_end - dMul_begin).count();
// 
//             eWiseMultAdd_begin = std::chrono::high_resolution_clock::now();
//             fuseEWiseMultAdd(W, Numspd, Bcu);
//             eWiseMultAdd_end = std::chrono::high_resolution_clock::now();
//             eWiseMultAdd_time += (eWiseMultAdd_end - eWiseMultAdd_begin).count();
//         }
// 
//         //                                 'bc'
// #pragma omp parallel for schedule(dynamic)
//         for (size_t i = 0; i < m; ++i) {
//             float *bcu_ptr = Bcu.Val + (size_t)n*i;
//             for (size_t j = 0; j < n; ++j) {
//                 bc[i] += bcu_ptr[j];
//             }
//             bc[i] -= (float)n;
//         }
// 
//         std::cerr << "Done " << (i+n)*100ull/A.n << "%\n";
//     }
// 
//     std::cerr << "Sparse addition time:            " << eWiseAdd_time    /1000000ll << "ms\n";
//     std::cerr << "Sparse matrix mult time:         " << spMul_time       /1000000ll << "ms\n";
//     std::cerr << "Dense elem-wise mult time:       " << eWiseMult_time   /1000000ll << "ms\n";
//     std::cerr << "Dense-sparse mult time:          " << dMul_time        /1000000ll << "ms\n";
//     std::cerr << "Dense elem-wise mult + add time: " << eWiseMultAdd_time/1000000ll << "ms\n";
// 
//     return bcv;
// }

#ifdef DEBUG_BACKWARD

// template <typename T>
void bc_backward_step(const spMtx<float> &A,
                      spMtx<float> &Front,
                      const spMtx<float> &Next,
                      const denseMtx<float> &Nspinv,
                      const denseMtx<float> &Numsp,
                      denseMtx<float> &Bcu) {
    size_t m = Front.m;
    size_t n = Front.n;

    // element-wise matrix multiplication
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < m; ++i) {
        // T *nspinv_row = Nspinv.Val + i * n;
        // T *bcu_row = Bcu.Val + i * n;
        for (size_t j = Front.Rst(i); j < Front.Rst(i+1); ++j) {
            size_t idx = Front.Col(j);
            Front.Val(j) = Nspinv.Val(i*n + idx) * Bcu.Val(i*n + idx);
        }
    }

    // MSpGEMM + addition straight into dense matrix
#pragma omp parallel
    {
        MSA<float> accum(Front.n);
        const float zero = 0.0f;

    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < Next.m; ++i) {
            int m_min = Next.Rst(i);
            int m_max = Next.Rst(i+1);

            for (int j = m_min; j < m_max; ++j)
                accum.value[Next.Col(j)] = zero;

            for (int t = A.Rst(i); t < A.Rst(i+1); ++t) {
                int k = A.Col(t);
                int b_pos = Front.Rst(k);
                int b_max = Front.Rst(k+1);
                float a_val = A.Val(t);

                for (int j = b_pos; j < b_max; ++j)
                    accum.value[Front.Col(j)] += a_val * Front.Val(j);
            }

            for (int j = m_min; j < m_max; ++j) {
                int idx = Next.Col(j);
                Bcu.Val(i*n + idx) += accum.value[idx] * Numsp.Val(i*n + idx);
            }
        }
    }
}

#else

void bc_backward_step(const spMtx<float> &A,
                      spMtx<float> &Front,
                      const spMtx<float> &Next,
                      const denseMtx<float> &Nspinv,
                      const denseMtx<float> &Numsp,
                      denseMtx<float> &Bcu) {
    size_t m = Front.m;
    size_t n = Front.n;

    // element-wise matrix multiplication
    Kokkos::parallel_for("EWiseMult", 
                         Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>>(0, m, Kokkos::ChunkSize(256)),
                         KOKKOS_LAMBDA(size_t i) {
        for (size_t j = Front.Rst(i); j < Front.Rst(i+1); ++j) {
            size_t idx = Front.Col(j);
            Front.Val(j) = Nspinv.Val(i*n + idx) * Bcu.Val(i*n + idx);
        }
    });

    const size_t num_of_teams = Kokkos::num_threads();
    View<int *> work_distribution("", num_of_teams+1);
    size_t team_i = 1;
    for (size_t i = 0; i <= Next.m; ++i) {
        if (Next.Rst[i] >= Next.nz * team_i / num_of_teams)
            work_distribution(team_i++) = i;
    }

    using ScratchPadView =
        View<float*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    int level = 0;
    TeamPolicy<DefaultExecutionSpace, Schedule<Dynamic>> policy(num_of_teams, Kokkos::AUTO, 64);
    policy = policy.set_scratch_size(level, Kokkos::PerThread(ScratchPadView::shmem_size(n)));

    // MSpGEMM + addition straight into dense matrix
    parallel_for("MSpGEMM_MSA", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const size_t team_id = team_member.league_rank();
        const size_t team_begin = work_distribution(team_id);
        const size_t team_end = work_distribution(team_id + 1);
        ScratchPadView accum(team_member.thread_scratch(level), n);
        parallel_for(Kokkos::TeamThreadRange(team_member, team_begin, team_end), KOKKOS_LAMBDA(const int i) {
            const float zero = 0.0f;
            int m_beg = Next.Rst(i);
            int m_end = Next.Rst(i+1);
            for (int j = m_beg; j < m_end; ++j)
                accum(Next.Col(j)) = zero;
            for (int t = A.Rst(i); t < A.Rst(i+1); ++t) {
                int k = A.Col(t);
                int b_beg = Front.Rst(k);
                int b_end = Front.Rst(k+1);
                float a_val = A.Val(t);
                parallel_for(Kokkos::ThreadVectorRange(team_member, b_beg, b_end), KOKKOS_LAMBDA(const int j) {
                    accum(Front.Col(j)) += a_val * Front.Val(j);
                });
            }
            parallel_for(Kokkos::ThreadVectorRange(team_member, m_beg, m_end), KOKKOS_LAMBDA(const int j) {
                int idx = Next.Col(j);
                Bcu.Val(i*n + idx) += accum(idx) * Numsp.Val(i*n + idx);
            });
        });
    });
}
#endif


#ifdef DEBUG_FORWARD

void bc_forward_step(const spMtx<int> &AT,
                     denseMtx<int> &Numsp,
                     spMtx<int> &Front,
                     spMtx<int> &FrontTmp) {
    size_t m = Front.m;
    size_t n = Front.n;

#pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int j = Front.Rst(i); j < Front.Rst(i+1); ++j)
            Numsp.Val(i * n + Front.Col(j)) += Front.Val(j);
#pragma omp parallel
    {
        MSA<int> accum(n);

    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < m; ++i) {
            for (int t = AT.Rst(i); t < AT.Rst(i+1); ++t) {
                int k = AT.Col(t);
                for (int j = Front.Rst(k); j < Front.Rst(k+1); ++j)
                    accum.state[Front.Col(j)] = MSA<int>::ALLOWED;
            }
            int row_nz = 0;
            for (size_t j = 0; j < n; ++j) {
                if (Numsp.Val(i*n + j) == 0 && accum.state[j] == MSA<int>::ALLOWED)
                    ++row_nz;
                accum.state[j] = MSA<int>::UNALLOWED;
            }
            FrontTmp.Rst(i+1) = row_nz;
        }
    #pragma omp single
        {
            FrontTmp.Rst(0) = 0;
            for (int i = 1; i < m; ++i)
                FrontTmp.Rst(i+1) += FrontTmp.Rst(i);
            FrontTmp.nz = FrontTmp.Rst(m);
            FrontTmp.resizeVals(FrontTmp.nz);
        }

    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < m; ++i) {
            for (int t = AT.Rst(i); t < AT.Rst(i+1); ++t) {
                int k = AT.Col(t);
                int a_val = AT.Val(t);
                for (int j = Front.Rst(k); j < Front.Rst(k+1); ++j)
                    accum.value[Front.Col(j)] += a_val * Front.Val(j);
            }

            int c_pos = FrontTmp.Rst(i);
            // int *numsp_row = Numsp.Val + i * n;
            for (int j = 0; j < accum.len; ++j) {
                if (Numsp.Val(i*n + j) == 0 && accum.value[j] != 0) {
                    FrontTmp.Col(c_pos) = j;
                    FrontTmp.Val(c_pos++) = accum.value[j];
                }
                accum.value[j] = 0;
            }
        }
    }

    std::swap(Front, FrontTmp);
}

#else

void bc_forward_step(const spMtx<int> &AT,
                     denseMtx<int> &Numsp,
                     spMtx<int> &Front,
                     spMtx<int> &FrontTmp) {
    size_t m = Front.m;
    size_t n = Front.n;


    Kokkos::parallel_for("InitNumsp", m, KOKKOS_LAMBDA(size_t i) {
        for (int j = Front.Rst(i); j < Front.Rst(i+1); ++j)
            Numsp.Val(i * n + Front.Col(j)) += Front.Val(j);
    });

    const size_t num_of_teams = Kokkos::num_threads();
    View<int *> work_distribution("", num_of_teams+1);
    size_t team_i = 1;
    for (size_t i = 0; i <= AT.m; ++i) {
        if (AT.Rst[i] >= AT.nz * team_i / num_of_teams)
            work_distribution(team_i++) = i;
    }
    using ScratchPadViewInt =
        View<int *, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    using ScratchPadViewChar =
        View<char *, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    int level = 0;
    TeamPolicy<DefaultExecutionSpace, Schedule<Dynamic>> policy(num_of_teams, Kokkos::AUTO, 64);
    policy = policy.set_scratch_size(level,
                Kokkos::PerThread(ScratchPadViewInt::shmem_size(n) + ScratchPadViewChar::shmem_size(n)));

    parallel_for("MSpGEMMSymbolic",
                 policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const size_t team_id = team_member.league_rank();
        const size_t team_begin = work_distribution(team_id);
        const size_t team_end = work_distribution(team_id + 1);
        ScratchPadViewInt accum_value(team_member.thread_scratch(level), n);
        ScratchPadViewChar accum_state(team_member.thread_scratch(level), n);
        parallel_for(Kokkos::TeamThreadRange(team_member, team_begin, team_end), KOKKOS_LAMBDA(const int i) {
            for (int t = AT.Rst(i); t < AT.Rst(i+1); ++t) {
                int k = AT.Col(t);
                int front_begin = Front.Rst(k);
                int front_end = Front.Rst(k+1);
                parallel_for(Kokkos::ThreadVectorRange(team_member, front_begin, front_end), KOKKOS_LAMBDA(const int j) {
                    accum_state(Front.Col(j)) = MSA<int>::ALLOWED;
                }); 
            }
            int row_nz = 0;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, 0, n), KOKKOS_LAMBDA(const int j, int &row_nz) {
                if (Numsp.Val(i*n + j) == 0 && accum_state(j) == MSA<int>::ALLOWED)
                    ++row_nz;
                accum_state(j) = MSA<int>::UNALLOWED;
            }, row_nz);
            FrontTmp.Rst(i+1) = row_nz;
        });
    });

    FrontTmp.Rst(0) = 0;
    for (int i = 1; i < m; ++i)
        FrontTmp.Rst(i+1) += FrontTmp.Rst(i);
    FrontTmp.nz = FrontTmp.Rst(m);
    FrontTmp.resizeVals(FrontTmp.nz);

    parallel_for("MSpGEMMNumeric",
                 policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const size_t team_id = team_member.league_rank();
        const size_t team_begin = work_distribution(team_id);
        const size_t team_end = work_distribution(team_id + 1);
        ScratchPadViewInt accum_value(team_member.thread_scratch(level), n);
        parallel_for(Kokkos::TeamThreadRange(team_member, team_begin, team_end), KOKKOS_LAMBDA(const int i) {
            for (int t = AT.Rst(i); t < AT.Rst(i+1); ++t) {
                int k = AT.Col(t);
                int front_begin = Front.Rst(k);
                int front_end = Front.Rst(k+1);
                int a_val = AT.Val(t);
                parallel_for(Kokkos::ThreadVectorRange(team_member, front_begin, front_end), KOKKOS_LAMBDA(const int j) {
                    accum_value[Front.Col(j)] += a_val * Front.Val(j);
                });
            }
            int c_pos = FrontTmp.Rst(i);
            for (int j = 0; j < n; ++j) {
                if (Numsp.Val(i*n + j) == 0 && accum_value(j) != 0) {
                    FrontTmp.Col(c_pos) = j;
                    FrontTmp.Val(c_pos++) = accum_value(j);
                }
                accum_value(j) = 0;
            }
        });
    });
    std::swap(Front, FrontTmp);
}

#endif


std::vector<float> betweenness_centrality_batch(bool isParallel, const spMtx<int> &A, size_t batchSize) {
    if (A.m != A.n)
        throw "non-square matrix; BC is only for square matrices";

    size_t m = A.m;
    size_t n = std::min(A.n, batchSize);
    std::vector<float> bcv(m);
    float *bc = bcv.data();
    std::vector<spMtx<float>> Sigmas(m);
    spMtx<int> AT = std::move(transpose(A));
    spMtx<int> Front(m, n);
    spMtx<int> Fronttmp(m, n);
    denseMtx<int> Numsp(m, n);

    spMtx<float> Af = std::move(convertType<float>(A));
    denseMtx<float> Numspd(m, n);
    denseMtx<float> Bcu(m, n);
    denseMtx<float> Nspinv(m, n);
    denseMtx<float> W;
    denseMtx<float> Wbuf;

    std::chrono::high_resolution_clock::time_point
        eWiseAdd_begin,
        spMul_begin,
        eWiseMult_begin,
        dMul_begin,
        eWiseMultAdd_begin,
        eWiseAdd_end,
        spMul_end,
        eWiseMult_end,
        dMul_end,
        eWiseMultAdd_end;
    long long eWiseAdd_time = 0,
        spMul_time = 0,
        eWiseMult_time = 0,
        dMul_time = 0,
        eWiseMultAdd_time = 0;

    size_t mxn = (size_t)m * n;
    for (size_t j = 0; j < n; ++j)
        Numsp.Val(j*(n+1)) = 1;
    Front = std::move(transpose(A.extractRows(0, n)));

    size_t d = 0;
    do {
        Sigmas[d] = std::move(convertType<float, int>(Front));
        // Sigmas[d] = Front;

        eWiseAdd_begin = std::chrono::high_resolution_clock::now();
        bc_forward_step(AT, Numsp, Front, Fronttmp);
        // add_nointersect(Numsp, Front, Numsp, Numspbuf);
        eWiseAdd_end = std::chrono::high_resolution_clock::now();
        eWiseAdd_time += (eWiseAdd_end - eWiseAdd_begin).count();
        
        // spMul_begin = std::chrono::high_resolution_clock::now();
        // mxmm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
        // spMul_end = std::chrono::high_resolution_clock::now();
        // spMul_time += (spMul_end - spMul_begin).count();

        // Front = Fronttmp;
        ++d;
    } while (Front.nz != 0);

    Nspinv.resize(m, n);
    Bcu.resize(m, n);
    // Numspd = Numsp;
#pragma omp parallel for
    for (size_t i = 0; i < mxn; ++i)
        Numspd.Val(i) = float(Numsp.Val(i));

#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Bcu.Val(i) = 1.0f;
#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Nspinv.Val(i) = 1.0f / Numspd.Val(i);

    for (size_t k = d-1; k > 0; --k) {
        dMul_begin = std::chrono::high_resolution_clock::now();
        bc_backward_step(Af, Sigmas[k], Sigmas[k-1], Nspinv, Numspd, Bcu);
        dMul_end = std::chrono::high_resolution_clock::now();
        dMul_time += (dMul_end - dMul_begin).count();
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        // float *bcu_ptr = Bcu.Val + n*i;
        for (size_t j = 0; j < n; ++j) {
            bc[i] += Bcu.Val(i*n + j);
        }
        bc[i] -= (float)n;
    }

    std::cerr << "Sparse addition time:            " << eWiseAdd_time    /1000000ll << "ms\n";
    std::cerr << "Sparse matrix mult time:         " << spMul_time       /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult time:       " << eWiseMult_time   /1000000ll << "ms\n";
    std::cerr << "Dense-sparse mult time:          " << dMul_time        /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult + add time: " << eWiseMultAdd_time/1000000ll << "ms\n";

    return bcv;
}

std::vector<float> betweenness_centrality_batch_old(bool isParallel, const spMtx<int> &A, size_t batchSize) {
    if (A.m != A.n)
        throw "non-square matrix; BC is only for square matrices";

    size_t m = A.m;
    size_t n = std::min(A.n, batchSize);
    std::vector<float> bcv(m);
    float *bc = bcv.data();
    // std::vector<spMtx<float>> Sigmas(m);
    std::vector<spMtx<int>> Sigmas(m);
    spMtx<int> AT = std::move(transpose(A));
    spMtx<int> Front(m, n);
    spMtx<int> Fronttmp(m, n);
    spMtx<int> Numsp(m, n);
    spMtx<int> Numspbuf(m, n);
    // denseMtx<int> Numsp(m, n);

    spMtx<float> Af = std::move(convertType<float>(A));
    denseMtx<float> Numspd(m, n);
    denseMtx<float> Bcu(m, n);
    denseMtx<float> Nspinv(m, n);
    denseMtx<float> W;
    denseMtx<float> Wbuf;

    std::chrono::high_resolution_clock::time_point
        eWiseAdd_begin,
        spMul_begin,
        eWiseMult_begin,
        dMul_begin,
        eWiseMultAdd_begin,
        eWiseAdd_end,
        spMul_end,
        eWiseMult_end,
        dMul_end,
        eWiseMultAdd_end;
    long long eWiseAdd_time = 0,
        spMul_time = 0,
        eWiseMult_time = 0,
        dMul_time = 0,
        eWiseMultAdd_time = 0;

    size_t mxn = (size_t)m * n;
    Numsp.resizeVals(n);
    Numsp.n = n;
    Numspbuf.n = n;
    for (size_t j = 0; j < n; ++j) {
        Numsp.Rst(j) = j;
        Numsp.Col(j) = j;
        Numsp.Val(j) = 1;
    }
    for (size_t j = n; j <= m; ++j)
        Numsp.Rst(j) = n;
    // for (size_t j = 0; j < n; ++j)
    //     Numsp.Val[j*(n+1)] = 1;
    Front = std::move(transpose(A.extractRows(0, n)));

    size_t d = 0;
    do {
        // Sigmas[d] = convertType<float, int>(Front);
        deep_copy(Sigmas[d], Front);

        eWiseAdd_begin = std::chrono::high_resolution_clock::now();
        // bc_forward_step(AT, Numsp, Front, Fronttmp);

        add_nointersect(Numsp, Front, Numsp, Numspbuf);
        eWiseAdd_end = std::chrono::high_resolution_clock::now();
        eWiseAdd_time += (eWiseAdd_end - eWiseAdd_begin).count();

        spMul_begin = std::chrono::high_resolution_clock::now();
        mxmm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
        spMul_end = std::chrono::high_resolution_clock::now();
        spMul_time += (spMul_end - spMul_begin).count();

        // Front = Fronttmp;
        ++d;
    } while (Front.nz != 0);

    Nspinv.resize(m, n);
    Bcu.resize(m, n);
    Numspd = Numsp;
    // #pragma omp parallel for
    //     for (size_t i = 0; i < mxn; ++i)
    //         Numspd.Val[i] = float(Numsp.Val[i]);

#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Bcu.Val(i) = 1.0f;
#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Nspinv.Val(i) = 1.0f / Numspd.Val(i);
    for (size_t k = d-1; k > 0; --k) {
        eWiseMult_begin = std::chrono::high_resolution_clock::now();
        eWiseMult(Nspinv, Bcu, Sigmas[k], W);
        eWiseMult_end = std::chrono::high_resolution_clock::now();
        eWiseMult_time += (eWiseAdd_end - eWiseAdd_begin).count();

        dMul_begin = std::chrono::high_resolution_clock::now();
        mxmm_spd(Af, W, Sigmas[k-1], W, Wbuf);
        dMul_end = std::chrono::high_resolution_clock::now();
        dMul_time += (dMul_end - dMul_begin).count();

        // dMul_begin = std::chrono::high_resolution_clock::now();
        // bc_backward_step(Af, Sigmas[k], Sigmas[k-1], Nspinv, Numspd, Bcu);
        // fuse_mxmm_eWiseMultAdd(Af, W, Sigmas[k - 1], Numspd, Bcu);
        // dMul_end = std::chrono::high_resolution_clock::now();
        // dMul_time += (dMul_end - dMul_begin).count();

        eWiseMultAdd_begin = std::chrono::high_resolution_clock::now();
        fuseEWiseMultAdd(W, Numspd, Bcu);
        eWiseMultAdd_end = std::chrono::high_resolution_clock::now();
        eWiseMultAdd_time += (eWiseMultAdd_end - eWiseMultAdd_begin).count();
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        // float *bcu_ptr = Bcu.Val + n*i;
        for (size_t j = 0; j < n; ++j) {
            bc[i] += Bcu.Val(i*n + j);
        }
        bc[i] -= (float)n;
    }

    std::cerr << "Sparse addition time:            " << eWiseAdd_time    /1000000ll << "ms\n";
    std::cerr << "Sparse matrix mult time:         " << spMul_time       /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult time:       " << eWiseMult_time   /1000000ll << "ms\n";
    std::cerr << "Dense-sparse mult time:          " << dMul_time        /1000000ll << "ms\n";
    std::cerr << "Dense elem-wise mult + add time: " << eWiseMultAdd_time/1000000ll << "ms\n";

    return bcv;
}