#pragma once

#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>
#include "mmio.h"
#include <Kokkos_StdAlgorithms.hpp>

using Kokkos::TeamPolicy;
using Kokkos::DefaultExecutionSpace;
using Kokkos::View;
using Kokkos::RangePolicy;
using Kokkos::Schedule;
using Kokkos::Dynamic;
using Kokkos::ChunkSize;
using Kokkos::parallel_for;

struct GraphInfo {
    std::string graphName;
    std::string graphPath;
    std::string logPath;
    std::string format;
};

template <typename ValT>
class spMtx {
public:
    size_t m = 0;
    size_t n = 0;
    size_t nz = 0;
    MM_typecode matcode;
    Kokkos::View<int*>  Rst;
    Kokkos::View<int*>  Col;
    Kokkos::View<ValT*> Val;

    spMtx(const char *filename, const std::string &format) {
        if (format == "mtx" && read_mtx_to_crs(filename)) {
            std::cout << "Can't read MTX from file\n";
            throw "Can't read MTX from file";
        } else if (format == "crs" && read_crs_to_crs(filename)) {
            std::cout << "Can't read CRS from file\n";
            throw "Can't read CRS from file";
        } else if (format == "bin" && read_bin_to_crs(filename)) {
            std::cout << "Can't read BIN from file\n";
            throw "Can't read BIN from file";
        } else if (format == "graph" && read_graph_to_crs(filename)) {
            std::cout << "Can't read GRAPH from file\n";
            throw "Can't read GRAPH from file";
        } else if (format == "rmat" && read_rmat_to_crs(filename)) {
            std::cout << "Can't read RMAT from file\n";
            throw "Can't read RMAT from file";
        }
    }

    spMtx() {}

    spMtx(size_t _m, size_t _n): m(_m), n(_n), Rst("", m+1) {
    }

    spMtx(size_t _m, size_t _n, size_t _nz): m(_m), n(_n), nz(_nz),
        Rst("", m+1), Col("", nz), Val("", nz) {
    }

    spMtx(const spMtx &src): m(src.m), n(src.n), nz(src.nz),
            Rst(src.Rst), Col(src.Col), Val(src.Val) {
        memcpy(matcode, src.matcode, sizeof(MM_typecode));
    }

    spMtx(spMtx &&mov): m(mov.m), n(mov.n), nz(mov.nz) {
        std::swap(Col, mov.Col);
        std::swap(Rst, mov.Rst);
        std::swap(Val, mov.Val);
        memcpy(matcode, mov.matcode, sizeof(MM_typecode));
    }

    ~spMtx() {}

    void resizeRows(size_t newM) {
        m = newM;
        Kokkos::resize(Rst, newM+1);
    }

    void resizeVals(size_t newNz) {
        Kokkos::resize(Col, newNz);
        Kokkos::resize(Val, newNz);
        nz = newNz;
    }

    // Копирование структуры матрицы без копирования значений
    template <typename ValT2>
    void deep_copy_pattern(const spMtx<ValT2> &src) {
        Kokkos::resize(Rst, src.Rst.size());
        Kokkos::deep_copy(Rst, src.Rst);
        Kokkos::resize(Col, src.Col.size());
        Kokkos::deep_copy(Col, src.Col);
        Kokkos::resize(Val, src.nz);
        m = src.m;
        n = src.n;
        nz = src.nz;
    }

    void deep_copy(const spMtx &src) {
        m = src.m;
        n = src.n;
        resizeRows(src.m);
        resizeVals(src.nz);
        Kokkos::deep_copy(Rst, src.Rst);
        Kokkos::deep_copy(Col, src.Col);
        Kokkos::deep_copy(Val, src.Val);
        memcpy(matcode, src.matcode, sizeof(MM_typecode));
    }

    spMtx& operator=(const spMtx &src) {
        if (this == &src)
            return *this;

        m = src.m;
        n  = src.n;
        nz = src.nz;
        Rst = src.Rst;
        Col = src.Col;
        Val = src.Val;
        memcpy(matcode, src.matcode, sizeof(MM_typecode));

        return *this;
    }

    spMtx& operator=(spMtx &&src) {
        if (this == &src)
            return *this;

        m  = src.m;
        n  = src.n;
        nz = src.nz;

        std::swap(Col, src.Col);
        std::swap(Rst, src.Rst);
        std::swap(Val, src.Val);
        memcpy(matcode, src.matcode, sizeof(MM_typecode));


        return *this;
    }

    spMtx extractRows(size_t begin, size_t end) const {
        spMtx result(end - begin, n, Rst[end] - Rst[begin]);

        for (size_t i = 0; i <= end - begin; ++i)
            result.Rst(i) = Rst(i + begin) - Rst(begin);
        Kokkos::deep_copy(result.Col, Kokkos::subview(Col, std::make_pair(Rst(begin), Rst(end))));
        Kokkos::deep_copy(result.Val, Kokkos::subview(Val, std::make_pair(Rst(begin), Rst(end))));
        memcpy(result.matcode, matcode, sizeof(MM_typecode));

        return result;
    }

    bool operator==(const spMtx &other) const {
        if (m != other.m || n != other.n || nz != other.nz)
            return false;
        for (size_t i = 0; i <= m; ++i)
            if (Rst(i) != other.Rst(i))
                return false;
        for (size_t j = 0; j < nz; ++j)
            if (Col(j) != other.Col(j) || Val(j) != other.Val(j))
                return false;
        return true;
    }

    void print_crs() const {
        std::cout << m << ' ' << nz << '\n';
        if (Val.size() > 0) {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = Rst(i); j < Rst(i+1); ++j)
                    std::cout << i+1 << ' ' << Col(j)+1 << '\n';
        }
        else {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = Rst(i); j < Rst(i+1); ++j)
                    std::cout << i+1 << ' ' << Col(j)+1 << ' ' << Val(j) << '\n';
        }
    }

    void print_dense() const {
        for (size_t i = 0; i < m; ++i) {
            size_t k = 0;
            for (size_t j = Rst(i); j < Rst(i+1); ++j, ++k) {
                while (k < Col(j)) {
                    std::cerr << 0 << ' ';
                    ++k;
                }
                std::cerr << Val(j) << ' ';
            }
            while (k < n) {
                std::cerr << 0 << ' ';
                ++k;
            }
            std::cerr << '\n';
        }
        std::cerr << '\n';
    }

    int write_crs_to_bin(const char *filename) {
        FILE *fp = fopen(filename, "wb");
        if (fp == NULL)
            return -1;
        
        fwrite(matcode, 1, 1, fp);
        fwrite(matcode + 1, 1, 1, fp);
        fwrite(matcode + 2, 1, 1, fp);
        fwrite(matcode + 3, 1, 1, fp);
        fwrite(&m, sizeof(size_t), 1, fp);
        fwrite(&n, sizeof(size_t), 1, fp);
        fwrite(&nz, sizeof(size_t), 1, fp);
        fwrite(Rst.data(), sizeof(int), m+1, fp);
        fwrite(Rst.data(), sizeof(int), nz, fp);
        fwrite(Rst.data(), sizeof(ValT), nz, fp);

        fclose(fp);
        return 0;
    }

    int write_crs_to_mtx(const char *filename) {
        mm_set_general(&matcode);
        FILE *fp = fopen(filename, "w");
        if (fp == NULL)
            return -1;
        mm_write_banner(fp, matcode);
        fclose(fp);

        std::ofstream ofstream;
        ofstream.open(filename, std::ios::out | std::ios::app);
        if (!ofstream.is_open())
            return -1;
        ofstream << m << ' ' << n << ' ' << nz << '\n';
        if (mm_is_pattern(matcode)) {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = Rst(i); j < Rst(i+1); ++j)
                    ofstream << i+1 << ' ' << Col(j)+1 << '\n';
        }
        else {
            for (size_t i = 0; i < m; ++i)
                for (size_t j = Rst(i); j < Rst(i+1); ++j)
                    ofstream << i+1 << ' ' << Col(j)+1 << ' ' << Val(j) << '\n';
        }

        ofstream.close();
        return 0;
    }

private:
    int read_mtx_to_crs(const char* filename) {
        /* variables */
        size_t row, col, nz_size, curr;
        int *edge_num, *last_el, *row_a, *col_a, m_int, n_int, nz_int;
        ValT val, *val_a;
        FILE *file;
        std::string str;
        std::ifstream ifstream;

        /* mtx correctness check */
        if ((file = fopen(filename, "r")) == NULL) {
            printf("Cannot open file\n");
            return 1;
        }
        if (mm_read_banner(file, &(matcode))) {
            return 1;
        }
        if (mm_read_mtx_crd_size(file, &m_int, &n_int, &nz_int)) {
            return 1;
        }
        m = (size_t)m_int;
        n = (size_t)n_int;
        nz = (size_t)nz_int;
        if (mm_is_complex(matcode) || mm_is_array(matcode)) {
            printf("This application doesn't support %s", mm_typecode_to_str(matcode));
            return 1;
        }
        if (m != n) {
            printf("Is not a square matrix\n");
            return 1;
        }
        fclose(file);

        /* Allocating memmory to store adjacency list */
        last_el  = new int[m];
        edge_num = new int[m];

        if (mm_is_symmetric(matcode)) {
            row_a = new int[2 * nz];
            col_a = new int[2 * nz];
            val_a = new ValT[2 * nz];
        }
        else {
            row_a = new int[nz];
            col_a = new int[nz];
            val_a = new ValT[nz];
        }
        for (size_t i = 0; i < m; i++) {
            edge_num[i] = 0;
        }

        /* Saving value of nz so we can change it */
        nz_size = nz;

        /* Reading file to count degrees of each vertex */
        std::ios_base::sync_with_stdio(false);  // input acceleration
        ifstream.open(filename);
        do {
            std::getline(ifstream, str);
        } while (str[0] == '%');
        curr = 0;
        if (mm_is_pattern(matcode)) {
            for(size_t i = 0; i < nz_size; i++) {
                ifstream >> row >> col;
                row--;
                col--;
                if (row == col) {
                    nz--;
                    continue; //we don't need loops
                }
                row_a[curr] = row;
                col_a[curr++] = col;
                ++edge_num[row];
                if (mm_is_symmetric(matcode)) {
                    ++edge_num[col];
                    ++nz;
                    row_a[curr] = col;
                    col_a[curr++] = row;
                }
            }
        }
        else {
            for (size_t i = 0; i < nz_size; i++) {
                ifstream >> row >> col >> val;
                row--;
                col--;
                if (row == col) {
                    nz--;
                    continue; //we don't need loops
                }
                row_a[curr] = row;
                col_a[curr] = col;
                val_a[curr++] = val;
                ++edge_num[row];
                if (mm_is_symmetric(matcode)) {
                    ++edge_num[col];
                    ++nz;
                    row_a[curr] = col;
                    col_a[curr] = row;
                    val_a[curr++] = val;
                }
            }
        }
        std::ios_base::sync_with_stdio(true); // restoring the state
        ifstream.close();
        
        /* Creating CRS arrays */
        resizeRows(m);
        resizeVals(nz);

        /* Writing data in Rst and last_el */
        Rst(0) = 0;
        for(size_t i = 0; i < m; i++) {
            Rst(i+1) = Rst(i) + edge_num[i];
            last_el[i] = Rst(i);
        }

        /* Reading file to write it's content in crs */
        if (mm_is_pattern(matcode)) {
            for (size_t i = 0; i < nz; ++i) {
                Col(last_el[row_a[i]]++) = col_a[i];
                Val(i) = 1;
            }
        } else {
            for (size_t i = 0; i < nz; ++i) {
                Col(last_el[row_a[i]]) = col_a[i];
                Val(last_el[row_a[i]]++) = val_a[i];
            }
        }

        delete[] edge_num;
        delete[] last_el;
        delete[] row_a;
        delete[] col_a;
        delete[] val_a;
        return 0;
    }

    int read_crs_to_crs(const char *filename) {
        std::ios_base::sync_with_stdio(false);
        std::ifstream ifstream(filename);
        if (!ifstream.is_open())
            return -1;

        ifstream >> m >> nz >> matcode;
        resizeRows(m);
        resizeVals(nz);
        for (size_t i = 0; i < nz; ++i)
            ifstream >> Col(i);
        for (size_t i = 0; i < nz; ++i)
            ifstream >> Val(i);
        for (size_t i = 0; i < m+1; ++i)
            ifstream >> Rst(i);

        ifstream.close();
        std::ios_base::sync_with_stdio(true);
        return 0;
    }

    int read_bin_to_crs(const char *filename) {
        FILE *fp = fopen(filename, "rb");
        if (fp == NULL)
            return -1;

        if (fread(matcode, 1, 1, fp) != 1)          return -2;
        if (fread(matcode + 1, 1, 1, fp) != 1)      return -2;
        if (fread(matcode + 2, 1, 1, fp) != 1)      return -2;
        if (fread(matcode + 3, 1, 1, fp) != 1)      return -2;
        if (fread(&m, sizeof(size_t), 1, fp) != 1)  return -2;
        if (fread(&n, sizeof(size_t), 1, fp) != 1)  return -2;
        if (fread(&nz, sizeof(size_t), 1, fp) != 1) return -2;

        resizeRows(m);
        resizeVals(nz);
        int  *RstTmp = new int[m+1];
        int  *ColTmp = new int[nz];
        ValT *ValTmp = new ValT[nz];
        if (fread(RstTmp, sizeof(int), m+1, fp) != m+1) return -3;
        if (fread(ColTmp, sizeof(int), nz, fp)  != nz ) return -3;
        if (fread(ValTmp, sizeof(ValT), nz, fp) != nz ) return -3;

        // write with parallel for into main arrays to get along with NUMA
    #pragma omp parallel for
        for (size_t i = 0; i <= m; ++i) {
            Rst(i) = RstTmp[i];
        }
    #pragma omp parallel for
        for (size_t i = 0; i < nz; ++i) {
            Col(i) = ColTmp[i];
            Val(i) = ValTmp[i];
        }

        fclose(fp);
        delete[] RstTmp;
        delete[] ColTmp;
        delete[] ValTmp;
        return 0;
    }

    int read_graph_to_crs(const char *filename) {
        std::ifstream ifstr(filename, std::ios::in);
        if (!ifstr.is_open())
            return -1;
        std::ios::sync_with_stdio(false);

        ifstr >> m >> nz;
        n = m;
        resizeRows(m);
        resizeVals(nz);

        Rst(0) = 0;
        std::string s;
        size_t j = 0;
        for (size_t i = 0; i < m; ++i) {
            std::getline(ifstr, s);
            std::istringstream iss(s);
            while (iss >> Col(j)) {
                Val(j) = 1;
                j++;
            }
            Rst(i+1) = j;
        }
        matcode[2] = 'I';

        std::ios::sync_with_stdio(true);
        ifstr.close();
        return 0;
    }

    int read_rmat_to_crs(const char *filename) {
        std::ifstream ifstr(filename, std::ios::in);
        if (!ifstr.is_open())
            return -1;
        std::ios::sync_with_stdio(false);

        ifstr >> m >> n >> nz;
        resizeRows(m);
        resizeVals(2*nz);
        int *TmpX = new int[nz];
        int *TmpY = new int[nz];

        size_t x, y;
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        mm_set_integer(&matcode);
        mm_set_general(&matcode);
        for (size_t j = 0; j < nz; ++j) {
            ifstr >> x >> y;
            TmpX[j] = x;
            TmpY[j] = y;
            ++Rst[x+1];
            ++Rst[y+1];
        }
        Rst(0) = 0;
        for (size_t i = 1; i < m; ++i)
            Rst(i+1) += Rst(i);
        for (size_t j = 0; j < nz; ++j) {
            x = TmpX[j];
            y = TmpY[j];
            Col(Rst[x]++) = y;
            Col(Rst[y]++) = x;
        }
        for (size_t i = m; i > 0; --i)
            Rst(i) = Rst(i-1);
        Rst(0) = 0;
        for (size_t j = 0; j < 2*nz; ++j)
            Val(j) = 1;
        nz *= 2;

        delete[] TmpX;
        delete[] TmpY;
        std::ios::sync_with_stdio(true);
        ifstr.close();
        return 0;
    }
};

template <typename ValT>
void deep_copy(spMtx<ValT> &dst, const spMtx<ValT> &src) {
    dst.m = src.m;
    dst.n = src.n;
    dst.resizeRows(src.m);
    dst.resizeVals(src.nz);
    Kokkos::deep_copy(dst.Rst, src.Rst);
    Kokkos::deep_copy(dst.Col, src.Col);
    Kokkos::deep_copy(dst.Val, src.Val);
    memcpy(dst.matcode, src.matcode, sizeof(MM_typecode));
}

template <typename ValT>
class denseMtx {
public:
    size_t m = 0;
    size_t n = 0;
    Kokkos::View<ValT*> Val;

    denseMtx() {}

    denseMtx(size_t _m, size_t _n) : m(_m), n(_n), Val("",m*n) {}
    denseMtx(const denseMtx &src) : m(src.m), n(src.n), Val(src.Val) {
        // Kokkos::deep_copy(Val, src.Val);
    }
    denseMtx(denseMtx &&mov) : m(mov.m), n(mov.n) {
        swap(Val, mov.Val);
    }
    template <typename ValT2>
    denseMtx(const spMtx<ValT2> &src) : m(src.m), n(src.n), Val("",m*n) {
//#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < src.m; ++i) {
            for (size_t k = src.Rst(i); k < src.Rst(i+1); ++k) {
                size_t j = src.Col(k);
                Val(i * n + j) = (ValT)(src.Val(k));
            }
        }
    }
    denseMtx& operator=(const denseMtx &src) {
        if (this != &src)
            return *this;

        m = src.m;
        n = src.n;
        Val = src.Val;
        // Kokkos::resize(Val, m*n);
        // Kokkos::deep_copy(Val, src.Val);

        return *this;
    }
    template <typename ValT2>
    denseMtx& operator=(const spMtx<ValT2> &src) {
        m = src.m;
        n = src.n;
        Kokkos::resize(Val, m*n);

#pragma omp parallel for schedule(dynamic, 512)
        for (size_t i = 0; i < src.m; ++i) {
            for (size_t j = 0; j < n; ++j)
                Val(i*n + j) = ValT();
            for (int k = src.Rst(i); k < src.Rst(i+1); ++k) {
                size_t j = (size_t)src.Col(k);
                Val(i * n + j) = (ValT)(src.Val(k));
            }
        }
        return *this;
    }
    denseMtx& operator=(denseMtx &&mov) {
        m = mov.m;
        n = mov.n;
        swap(Val, mov.Val);

        return *this;
    }
    ~denseMtx() {}

    bool operator==(const denseMtx &B) {
        return m == B.m && n == B.n &&
            Kokkos::Experimental::equal(Kokkos::DefaultHostExecutionSpace(), Val, B.Val);
    }

    void resize(size_t newM, size_t newN) {
        m = newM;
        n = newN;
        Kokkos::resize(Val, newM * newN);
    }
    void print() {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j)
                std::cout << std::setw(5) << Val(i * n + j) << ' ';
            std::cout << '\n';
        }
        std::cout << '\n';
    }
};

template <typename ValT2, typename ValT1>
spMtx<ValT2> convertType(const spMtx<ValT1> &src) {
    spMtx<ValT2> result;

    result.deep_copy_pattern(src);
    for (size_t j = 0; j < src.nz; ++j)
        result.Val(j) = (ValT2)(src.Val(j));

    return result;
}

template <typename T>
spMtx<int> build_adjacency_matrix(const spMtx<T> &Gr) {
    spMtx<int> Res;
    Res.m = Gr.m;
    Res.n = Gr.n;
    Res.nz = Gr.nz;
    Res.resizeRows(Gr.m);
    Res.resizeVals(Gr.nz);
    Kokkos::deep_copy(Res.Rst, Gr.Rst);
    Kokkos::deep_copy(Res.Col, Gr.Col);
    std::memcpy(Res.matcode, Gr.matcode, 4);
    Res.matcode[2] = 'I';

    for (size_t i = 0; i < Gr.nz; ++i)
        Res.Val(i) = 1;

    return Res;
}

template <typename T>
spMtx<T> build_symm_from_lower(const spMtx<T> &Low) {
    spMtx<T> Res(Low.m, Low.n, 2*Low.nz);
    spMtx<T> Upp = transpose(Low);
    size_t jl = 0;
    size_t ju = 0;
    size_t jr = 0;

    for (size_t i = 0; i < Low.m; ++i) {
        size_t xl = Low.Rst[i+1];
        size_t xu = Upp.Rst[i+1];

        while (jl < xl && ju < xu) {
            if (Low.Col[jl] < Upp.Col[ju]) {
                Res.Col[jr] = Low.Col[jl];
                Res.Val[jr++] = Low.Val[jl++];
            }
            else {
                Res.Col[jr] = Upp.Col[ju];
                Res.Val[jr++] = Upp.Val[ju++];
            }
        }
        while (jl < xl) {
            Res.Col[jr] = Low.Col[jl];
            Res.Val[jr++] = Low.Val[jl++];
        }
        while (ju < xu) {
            Res.Col[jr] = Upp.Col[ju];
            Res.Val[jr++] = Upp.Val[ju++];
        }
        Res.Rst[i+1] = jr;
    }

    return Res;
}

template <typename T>
spMtx<T> extract_lower_triangle(const spMtx<T> &Gr) {
    spMtx<T> Res;

    Res.m = Gr.m;
    Res.n = Gr.n;
    Res.resizeRows(Gr.m);
    Res.Rst(0) = 0;

    for (size_t i = 0; i < Gr.m; ++i) {
        int r = Gr.Rst(i);
        while (r < Gr.Rst(i+1) && Gr.Col(r) < i)
            ++r;
        Res.Rst(i+1) = Res.Rst(i) + (r - Gr.Rst(i));
    }

    Res.nz = Res.Rst(Res.m);
    Res.resizeVals(Res.nz);

    for (size_t i = 0; i < Gr.m; ++i) {
        size_t row_len = Res.Rst(i+1) - Res.Rst(i);
        for (size_t j = 0; j < row_len; ++j) {
            Res.Col(Res.Rst(i) + j) = Gr.Col(Gr.Rst(i) + j);
            Res.Val(Res.Rst(i) + j) = Gr.Val(Gr.Rst(i) + j);
        }
    }   
    
    return Res;
}

template <typename T>
spMtx<T> extract_upper_triangle(const spMtx<T> &Gr) {
    spMtx<T> Res;

    Res.m = Gr.m;
    Res.n = Gr.n;
    Res.Rst = new int[Gr.m + 1];
    Res.Rst[0] = 0;

    for (size_t i = 0; i < Gr.m; ++i) {
        size_t r = Gr.Rst[i];
        while (r < Gr.Rst[i+1] && Gr.Col[r] <= i)
            ++r;
        Res.Rst[i+1] = Res.Rst[i] + (Gr.Rst[i+1] - r);
    }

    Res.nz = Res.Rst[Res.m];
    Res.Col = new int[Res.nz];
    Res.Val = new   T[Res.nz];

    for (size_t i = 0; i < Gr.m; ++i) {
        int row_len = Res.Rst[i+1] - Res.Rst[i];
        int row_offset = Gr.Rst[i+1] - row_len;
        std::memcpy(Res.Col + Res.Rst[i], Gr.Col + row_offset, row_len*sizeof(int));
        std::memcpy(Res.Val + Res.Rst[i], Gr.Val + row_offset, row_len*sizeof(T));
    }

    return Res;
}