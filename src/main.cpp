#include "matrix.h"
#include "betweenness_centrality.h"
#include "matrix_utils.h"
#include <omp.h>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <functional>
#include <Kokkos_Core.hpp>
using namespace std;

template <typename T>
using mxmOp = void(*)(bool, const spMtx<T>&, const spMtx<T>&, const spMtx<T>&, spMtx<T>&);

int* triangle_counting_vertex(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel) {
    /* PREPARE DATA */
    int *nums_of_tr = new int[A.m];
    int num_of_tr;
    spMtx<int> SQ; // A^2 (A is adjacency matrix)

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, A, A, A, SQ);
    // for each vertex we count the number of triangles it belongs to
    for (size_t i = 0; i < A.m; ++i) {
        num_of_tr = 0;
        for (int j = SQ.Rst(i); j < SQ.Rst(i+1); ++j)
            num_of_tr += SQ.Val(j);
        nums_of_tr[i] = num_of_tr >>= 1;
    }
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << '\n';

    return nums_of_tr;
}


int64_t triangle_counting_masked_lu(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel) {
    int64_t num_of_tr = 0;
    spMtx<int> L = std::move(extract_lower_triangle(A));
    // spMtx<int> L;
    // deep_copy(L, extract_lower_triangle(A));
    spMtx<int> U = std::move(transpose(L));
    spMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, L, U, A, C);

    // Count the total number of triangles
    for (int j = 0; j < C.Rst(C.m); ++j)
        num_of_tr += C.Val(j);
    num_of_tr >>= 1;
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}


int64_t triangle_counting_masked_sandia(const spMtx<int> &A, mxmOp<int> matrixMult, bool isParallel) {
    int64_t num_of_tr = 0;
    spMtx<int> L = std::move(extract_lower_triangle(A));
    spMtx<int> C;

    auto start = chrono::steady_clock::now();

    /* TRIANGLE COUNTING ITSELF */
    matrixMult(isParallel, L, L, L, C);

    // Count the total number of triangles
    Kokkos::parallel_reduce("", C.nz, KOKKOS_LAMBDA(int64_t i, int64_t& sum) {
        sum += C.Val(i);
    }, Kokkos::Sum<int64_t>(num_of_tr));
    /* TRIANGLE COUNTING ITSELF */

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Triangles:  " << num_of_tr << '\n';

    return num_of_tr;
}


/* K-TRUSS */
spMtx<int> k_truss(const spMtx<int> &A, int k, mxmOp<int> matrixMult, bool isParallel) {
    spMtx<int> C = A;  // a copy of adjacency matrix
    spMtx<int> Tmp;
    int n = A.m;
    int totalIterationNum = 0;
    Kokkos::View<int*> tmp_Xdj("", n);
    tmp_Xdj(0) = 0;

    auto start = chrono::steady_clock::now();

    for (int t = 0; t < n; ++t) {
        // Tmp<C> = C*C
        matrixMult(isParallel, C, C, C, Tmp);

        // remove all edges included in less than (k-2) triangles
        // and replace values of remaining entries with 1
        int new_curr_pos = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = Tmp.Rst(i); j < Tmp.Rst(i+1); ++j) {
                if (Tmp.Val(j) >= k-2) {
                    Tmp.Col(new_curr_pos)   = Tmp.Col[j];
                    Tmp.Val(new_curr_pos++) = 1;
                }
            }
            tmp_Xdj(i+1) = new_curr_pos;
        }
        Kokkos::deep_copy(Tmp.Rst, tmp_Xdj);
        Tmp.nz = Tmp.Rst(n);

        // check if the number of edges has changed
        if (Tmp.nz == C.nz) {
            totalIterationNum = ++t;
            break;
        }

        // Assign 'Tmp' to 'C'
        std::swap(C, Tmp);
    }

    Kokkos::resize(C.Col, C.nz);
    Kokkos::resize(C.Val, C.nz);

    auto finish = chrono::steady_clock::now();
    cout << "Time:       " << chrono::duration_cast<chrono::milliseconds>(finish - start).count() << " ms\n";
    cout << "Iterations: " << totalIterationNum << '\n';

    return C;
}


GraphInfo get_graph_info(int argc, char *argv[]) {
    std::time_t current_time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    string current_date_time(ctime(&current_time));
    string graphName;
    string graphPath(argv[1]);
    string logfile(argv[2]);
    string action(argv[3]);
    string format;

    int slashPos = graphPath.size() - 1;
    int dotPos   = graphPath.size() - 1;

    while (graphPath[slashPos] != '/')
        --slashPos;
    while (graphPath[dotPos] != '.')
        --dotPos;

    graphName = graphPath.substr(slashPos + 1, dotPos - slashPos - 1);
    format = graphPath.substr(dotPos + 1, graphPath.size() - dotPos);
    graphPath = graphPath.substr(0, slashPos + 1);

    if (action == "launch") {
        current_date_time.pop_back();
        replace(current_date_time.begin(), current_date_time.end(), ' ', '_');
        replace(current_date_time.begin(), current_date_time.end(), ':', '-');

        stringstream ss;
        ss << logfile << '/' << current_date_time << '_' << graphName;
        for (int i = 4; i < argc; ++i)
            ss << '_' << argv[i];
        ss << '_' << to_string(omp_get_max_threads()) << ".txt";
        logfile = ss.str();
    }

    GraphInfo info {graphName, graphPath, logfile, format};

    return info;
}

int launch_test(const spMtx<int> &gr, const GraphInfo &info, int argc, char *argv[]) {
    string benchmarkAlgorithm(argv[4]),
           parOrSeq(argv[5]),
           batchSizeStr;
    bool isParallel = (parOrSeq == "par");
    size_t batch_size;
    mxmOp<int> mxm_algorithm;
    spMtx<int> MxmResult, TestMtx = gr;
    chrono::high_resolution_clock::time_point start, finish, default_time;
    stringstream alg_ss;

    if (benchmarkAlgorithm == "bc") {
        if (argc < 7 || (batch_size = atoll(argv[6])) == 0) {
            cerr << "incorrect input, 6-th argument: batch has to be positive integer\n";
            return -6;
        }
    }
    else {
        string multiplicationAlgorithm(argv[6]);
        if (multiplicationAlgorithm == "naive")
            mxm_algorithm = mxmm_naive<int>;
        if (multiplicationAlgorithm == "msa")
            mxm_algorithm = mxmm_msa<int>;
        else if (multiplicationAlgorithm == "mca")
            mxm_algorithm = mxmm_mca<int>;
        else {
            cerr << "incorrect input, 6-th argument: has to be 'naive', 'msa' or 'mca')\n";
            return -7;
        }
    }

    if (benchmarkAlgorithm != "bc")
        TestMtx = std::move(build_symm_from_lower(extract_lower_triangle(TestMtx)));

    if (parOrSeq == "par") {
        alg_ss << "Parallel,   " << omp_get_max_threads() << " threads\n";
        // cerr << "PARALLEL " << omp_get_max_threads() << " THREADS\n";
    } else if (parOrSeq == "seq") {
        alg_ss << "Sequential\n";
        // cerr << "SEQUENTIAL \n\n";
    } else {
        cerr << "incorrect input, 5-th argument: has to be 'par' or 'seq'\n";
        return -5;
    }

    if (benchmarkAlgorithm == "mxm") {
        start = chrono::high_resolution_clock::now();
        mxm_algorithm(isParallel, TestMtx, TestMtx, TestMtx, MxmResult);
        finish = chrono::high_resolution_clock::now();
        alg_ss << "Algorithm:  matrix square\n";
    }
    else if (benchmarkAlgorithm == "k-truss") {
        if (argc < 8 || atoi(argv[7]) < 3) {
            cerr << "incorrect input, 7-th argument: has to be positive integer bigger than 2\n";
        }
        alg_ss << "Algorithm:  k-truss, k = " << argv[7] << '\n';
        k_truss(TestMtx, stoi(argv[7]), mxm_algorithm, isParallel);
    }
    else if (benchmarkAlgorithm == "triangle") {
        alg_ss << "Algorithm:  triangle counting\n";
        triangle_counting_masked_sandia(TestMtx, mxm_algorithm, isParallel);
    }
    else if (benchmarkAlgorithm == "bc") {
        std::vector<float> bcVector;
        // size_t cache_fit_size = 1747626; // to size of float matrix to fit into 20 Mb cache 
        // size_t batch_size = (cache_fit_size/Adj.m > 0) ? cache_fit_size/Adj.m : 3;
        start = chrono::high_resolution_clock::now();
        bcVector = betweenness_centrality_batch(isParallel, TestMtx, batch_size);
        finish = chrono::high_resolution_clock::now();
        // bcVector = betweenness_centrality(isParallel, TestMtx, 5);
        float sum = 0.0f;
        for (size_t i = 0; i < bcVector.size(); ++i)
            sum += bcVector[i];
        // for (size_t i = 0; i < bcVector.size(); ++i)
        //     cout << bcVector[i] << '\n';
        
        // cout << '\n';
        alg_ss << "Algorithm:  betweenness centrality\n" << "Batch size: " << batch_size << '\n';
        alg_ss << "Checksum:   " << sum << '\n';
    }
    else {
        cerr << "incorrect input, 4-th argument: has to be 'triangle', 'k-truss', 'mxm' or 'bc')\n";
        return -4;
    }
    long long time = chrono::duration_cast<chrono::milliseconds>(finish - start).count();
    if (start != default_time)
        cout << "Time:       " << time << " ms\n";
    cout << "Graph name: " << info.graphName << '\n';
    cout << "Vertices:   " << TestMtx.m << '\n';
    cout << "Edges:      " << TestMtx.nz << '\n';
    cout << alg_ss.str() << '\n';
    return 0;
}

string get_graph_val_type(const char *filename, const GraphInfo &info) {
    ifstream istr(filename, (info.format == "bin") ? std::ios::in | std::ios::binary
                                                   : std::ios::in);
    string stype;
    if (info.format == "bin") {
        char type;
        istr >> type >> type >> type;
        if (type == 'R')
            stype = "real";
        else if (type == 'I' || type == 'P')
            stype = "integer";
    } else if (info.format == "mtx") {
        string type;
        istr >> type >> type >> type >> type;
        if (type == "complex")
            throw "Can't use complex numbers!";
        if (type == "real")
            stype = "real";
        else
            stype = "integer";
    } else if (info.format == "graph")
        stype = "integer";
    else if (info.format == "rmat")
        stype = "integer";
    else {
        istr.close();
        throw "Unknown format";
    }

    istr.close();
    return stype;
}

// argv[1] - path to graph
// argv[2] - path to log folder
// argv[3] - action (tobinary / launch)
// if 'launch':
//   argv[4] - algorithm to execute (triangle / k-truss / mxm / bc)
//   argv[5] - parallelism mode (seq / par)
//   if 'bc':
//     argv[6] - batch size (how many vertices are used to compute dependencies in BC)
//   if 'triangle' / 'k-truss' / 'mxm':
//     argv[6] - accumulator to use in algorithm (naive / msa / mca / heap)
//     if k-truss:
//       argv[7] - parameter 'k' in k-truss

template <typename ValType>
int read_graph_and_launch_test(const GraphInfo &info, int argc, char *argv[]) {
    string action(argv[3]);
    spMtx<ValType> gr(argv[1], info.format.c_str());
    cerr << "finished reading\n";
    if (action == "to_bin") {
        gr.write_crs_to_bin((info.graphPath + info.graphName + ".bin").c_str());
        cerr << "finished writing to BIN\n";
        return 0;
    }
    else if (action == "to_mtx") {
        gr.write_crs_to_mtx((info.graphPath + info.graphName + ".mtx").c_str());
        cerr << "finished writing to MTX\n";
        return 0;
    }
    else if (action == "launch") {
       return launch_test(build_adjacency_matrix(gr), info, argc, argv);
    } else {
        cerr << "incorrect input (3-nd argument has to be 'tobinary' or 'launch')\n";
        return -3;
    }
}

//#define TEST
#ifndef TEST

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "not enough arguments (have to be at least 3)\n";
        return -1;
    }

    GraphInfo info = get_graph_info(argc, argv);
    string grType = get_graph_val_type(argv[1], info);

    Kokkos::initialize(argc, argv);

    if (grType == "integer")
        return read_graph_and_launch_test<int>(info, argc, argv);
    else if (grType == "real") {
        return read_graph_and_launch_test<double>(info, argc, argv);
    } else {
        cerr << "unknown value type\n";
        return -2;
    }

    Kokkos::finalize();

    return 0;
}
#else
void test_add_nointersect() {
    for (int iter = 0; iter < 10; ++iter) {
        spMtx<int> A = generate_adjacency_matrix(500, 20, 50),
                   C = generate_adjacency_matrix(500, 20, 50),
                   B(A.m, A.n, A.nz + C.nz);
        int bi = 0;
        B.Rst[0] = 0;
        for (int i = 0; i < A.m; ++i) {
            int ai = A.Rst[i];
            for (int ci = C.Rst[i]; ci < C.Rst[i+1]; ++ci) {
                while (ai < A.Rst[i+1] && A.Col[ai] < C.Col[ci])
                    ++ai;
                if (A.Col[ai] != C.Col[ci]) {
                    B.Col[bi] = C.Col[ci];
                    B.Val[bi++] = C.Val[ci];
                }
            }
            B.Rst[i+1] = bi;
        }

        spMtx<int> Sum(A.m, A.n);
        spMtx<int> Sumbuf(A.m, A.n);
        add_nointersect(A, B, Sum, Sumbuf);

        denseMtx<int> denseA = A, denseB = B;
        denseMtx<int> denseSum = A;
        for (size_t i = 0; i < A.m * A.n; ++i)
            denseSum.Val[i] += denseB.Val[i];

        denseMtx<int> spToDenseSum = Sum;
        if (spToDenseSum == denseSum) {
            cout << iter << ": fine\n";
        }
        else {
            cout << iter << ": WRONG ANSWER!\n";
            cout << "A:\n";
            denseA.print();
            cout << "B:\n";
            denseB.print();
        }
    }
}

void test_spmm_cmask() {
    for (int iter = 0; iter < 500; ++iter) {
        spMtx<int> A = generate_adjacency_matrix(500, 50, 70),
                   B = generate_adjacency_matrix(500, 50, 70),
                   M = generate_adjacency_matrix(500, 50, 70),
                   C(A.m, A.n);
        mxmm_msa_cmask(true, A, B, M, C);

        denseMtx<int> denseA = A,
            denseB = B,
            denseM = M,
            denseC = C;
        for (size_t i = 0; i < A.m * A.n; ++i)
            denseM.Val[i] = 1 - denseM.Val[i];
        dense_mtx_mult(denseA, denseB, denseC);
        for (size_t i = 0; i < C.m * C.n; ++i)
            denseC.Val[i] *= denseM.Val[i];

        denseMtx<int> spToDenseC = C;
        if (spToDenseC == denseC) {
            cout << iter << ": fine\n";
        }
        else {
            cout << iter << ": WRONG ANSWER!\n";
            cout << "A:\n";
            denseA.print();
            cout << "B:\n";
            denseB.print();
            cout << "M:\n";
            denseM.print();
        }
    }
}

int main(int argc, const char *argv[]) {
    test_add_nointersect();
    test_spmm_cmask();
}
#endif
