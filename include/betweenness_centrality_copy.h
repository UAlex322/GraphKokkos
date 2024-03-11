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
//         size_t n = std::min(A.n - i, blockSize); // ���������� ��������
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
//         // ������ ������ (����� � ������)
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
//         // �������� ������
//         Numspd = Numsp; // �������������� ����������� ������� � �������
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
//         // �������� ����������� �������� � 'bc'
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

template <typename T>
void bc_backward_step(const spMtx<T> &A,
                      spMtx<T> &Front,
                      const spMtx<T> &Next,
                      const denseMtx<T> &Nspinv,
                      const denseMtx<T> &Numsp,
                      denseMtx<T> &Bcu) {
    size_t m = Front.m;
    size_t n = Front.n;

    // element-wise matrix multiplication
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < m; ++i) {
        T *nspinv_row = Nspinv.Val + i * n;
        T *bcu_row = Bcu.Val + i * n;
        for (size_t j = Front.Rst[i]; j < Front.Rst[i+1]; ++j) {
            size_t idx = Front.Col[j];
            Front.Val[j] = nspinv_row[idx] * bcu_row[idx];
        }
    }

    // MSpGEMM + addition straight into dense matrix
#pragma omp parallel
    {
        MSA<T> accum(Front.n);
        const T zero = T(0);

    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < Next.m; ++i) {
            int m_min = Next.Rst[i];
            int m_max = Next.Rst[i+1];

            for (int j = m_min; j < m_max; ++j)
                accum.value[Next.Col[j]] = zero;

            for (int t = A.Rst[i]; t < A.Rst[i+1]; ++t) {
                int k = A.Col[t];
                int b_pos = Front.Rst[k];
                int b_max = Front.Rst[k+1];
                T a_val = A.Val[t];

                for (int j = b_pos; j < b_max; ++j)
                    accum.value[Front.Col[j]] += a_val * Front.Val[j];
            }

            T *bcu_row = Bcu.Val + i * n;
            T *numsp_row = Numsp.Val + i * n;
            for (int j = m_min; j < m_max; ++j) {
                int idx = Next.Col[j];
                bcu_row[idx] += accum.value[idx] * numsp_row[idx];
            }
        }
    }
}

// template <typename T>
void bc_forward_step(const spMtx<int> &AT,
                     denseMtx<int> &Numsp,
                     spMtx<int> &Front,
                     spMtx<int> &FrontTmp) {
    size_t m = Front.m;
    size_t n = Front.n;

#pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int j = Front.Rst[i]; j < Front.Rst[i+1]; ++j)
            Numsp.Val[i * n + Front.Col[j]] += Front.Val[j];
#pragma omp parallel
    {
        MSA<int> accum(n);

    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < m; ++i) {
            for (int t = AT.Rst[i]; t < AT.Rst[i+1]; ++t) {
                int k = AT.Col[t];
                for (int j = Front.Rst[k]; j < Front.Rst[k+1]; ++j)
                    accum.state[Front.Col[j]] = MSA<int>::ALLOWED;
            }
            int row_nz = 0;
            int *numsp_row = Numsp.Val + i * n;
            for (size_t j = 0; j < n; ++j) {
                if (numsp_row[j] == 0 && accum.state[j] == MSA<int>::ALLOWED)
                    ++row_nz;
                accum.state[j] = MSA<int>::UNALLOWED;
            }
            FrontTmp.Rst[i+1] = row_nz;
        }
    #pragma omp single
        {
            FrontTmp.Rst[0] = 0;
            for (int i = 1; i < m; ++i)
                FrontTmp.Rst[i+1] += FrontTmp.Rst[i];
            FrontTmp.nz = FrontTmp.Rst[m];
            FrontTmp.resizeVals(FrontTmp.nz);
        }

    #pragma omp for schedule(dynamic, 256)
        for (size_t i = 0; i < m; ++i) {
            for (int t = AT.Rst[i]; t < AT.Rst[i+1]; ++t) {
                int k = AT.Col[t];
                int a_val = AT.Val[t];
                for (int j = Front.Rst[k]; j < Front.Rst[k+1]; ++j)
                    accum.value[Front.Col[j]] += a_val * Front.Val[j];
            }

            int c_pos = FrontTmp.Rst[i];
            int *numsp_row = Numsp.Val + i * n;
            for (int j = 0; j < accum.len; ++j) {
                if (numsp_row[j] == 0 && accum.value[j] != 0) {
                    FrontTmp.Col[c_pos] = j;
                    FrontTmp.Val[c_pos++] = accum.value[j];
                }
                accum.value[j] = 0;
            }
        }
    }

    std::swap(Front, FrontTmp);
}

std::vector<float> betweenness_centrality_batch(bool isParallel, const spMtx<int> &A, size_t batchSize) {
    if (A.m != A.n)
        throw "non-square matrix; BC is only for square matrices";

    size_t m = A.m;
    size_t n = std::min(A.n, batchSize);
    std::vector<float> bcv(m);
    float *bc = bcv.data();
    std::vector<spMtx<float>> Sigmas(m);
    spMtx<int> AT = transpose(A);
    spMtx<int> Front(m, n);
    spMtx<int> Fronttmp(m, n);
    // spMtx<int> Numsp(m, n);
    // spMtx<int> Numspbuf(m, n);
    denseMtx<int> Numsp(m, n);

    spMtx<float> Af = convertType<float>(A);
    denseMtx<float> Numspd(m, n);
    denseMtx<float> Bcu(m, n);
    denseMtx<float> Nspinv(m, n);

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
    // Numsp.resizeVals(n);
    // Numsp.n = n;
    // Numspbuf.n = n;
    // for (size_t j = 0; j < n; ++j) {
    //     Numsp.Rst[j] = j;
    //     Numsp.Col[j] = j;
    //     Numsp.Val[j] = 1;
    // }
    // for (size_t j = n; j <= m; ++j)
    //     Numsp.Rst[j] = n;
    for (size_t j = 0; j < n; ++j)
        Numsp.Val[j*(n+1)] = 1;
    Front = transpose(A.extractRows(0, n));

    size_t d = 0;
    do {
        Sigmas[d] = convertType<float, int>(Front);

        eWiseAdd_begin = std::chrono::high_resolution_clock::now();
        bc_forward_step(AT, Numsp, Front, Fronttmp);

        // add_nointersect(Numsp, Front, Numsp, Numspbuf);
        eWiseAdd_end = std::chrono::high_resolution_clock::now();
        eWiseAdd_time += (eWiseAdd_end - eWiseAdd_begin).count();
        // 
        // spMul_begin = std::chrono::high_resolution_clock::now();
        // mxmm_msa_cmask(isParallel, AT, Front, Numsp, Fronttmp);
        // spMul_end = std::chrono::high_resolution_clock::now();
        // spMul_time += (spMul_end - spMul_begin).count();

        // Front = Fronttmp;
        ++d;
    } while (Front.nz != 0);
    
    // Nspinv.resize(m, n);
    // Bcu.resize(m, n);
    // Numspd = Numsp;
#pragma omp parallel for
    for (size_t i = 0; i < mxn; ++i)
        Numspd.Val[i] = float(Numsp.Val[i]);

#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Bcu.Val[i] = 1.0f;
#pragma omp parallel for simd
    for (size_t i = 0; i < mxn; ++i)
        Nspinv.Val[i] = 1.0f / Numspd.Val[i];
    for (size_t k = d-1; k > 0; --k) {
        // eWiseMult_begin = std::chrono::high_resolution_clock::now();
        // eWiseMult(Nspinv, Bcu, Sigmas[k], W);
        // eWiseMult_end = std::chrono::high_resolution_clock::now();
        // eWiseMult_time += (eWiseAdd_end - eWiseAdd_begin).count();

        // dMul_begin = std::chrono::high_resolution_clock::now();
        // mxmm_spd(Af, W, Sigmas[k-1], W, Wbuf);
        // dMul_end = std::chrono::high_resolution_clock::now();
        // dMul_time += (dMul_end - dMul_begin).count();

        dMul_begin = std::chrono::high_resolution_clock::now();
        bc_backward_step(Af, Sigmas[k], Sigmas[k-1], Nspinv, Numspd, Bcu);
        // fuse_mxmm_eWiseMultAdd(Af, W, Sigmas[k - 1], Numspd, Bcu);
        dMul_end = std::chrono::high_resolution_clock::now();
        dMul_time += (dMul_end - dMul_begin).count();

        // eWiseMultAdd_begin = std::chrono::high_resolution_clock::now();
        // fuseEWiseMultAdd(W, Numspd, Bcu);
        // eWiseMultAdd_end = std::chrono::high_resolution_clock::now();
        // eWiseMultAdd_time += (eWiseMultAdd_end - eWiseMultAdd_begin).count();
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        float *bcu_ptr = Bcu.Val + n*i;
        for (size_t j = 0; j < n; ++j) {
            bc[i] += bcu_ptr[j];
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