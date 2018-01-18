#ifndef _KMEANS_CPP_
#define _KMEANS_CPP_

#include <vector>
#include <cstdlib>
#include <limits.h>
#include <float.h>
#include <iterator>
#include <algorithm>
#include <random>
class KMeans
{
    double **solution; // k x dim dimension
    unsigned int *labels; // k-dim
    unsigned int **rnks;
    int N;
    int dim;
    int iter;
    int k;
public:
    KMeans()
    {}
    KMeans(int N_, int dim_, int k_, int iter_)
    {
        solution = new double*[k_];
        for(int i = 0; i < k_; i++)
        {
            solution[i] = new double[dim_];
        }
        k = k_;
        N = N_;
        dim = dim_;
        iter= iter_;
        labels = new unsigned int[k];
        std::memset(labels, 0, sizeof(unsigned int) * k);
        rnks = new unsigned int*[N];
        for(int i = 0; i < N; i++)
        {
            rnks[i] = new unsigned int[k];
        }
    }
    ~KMeans()
    {
        for(int i = 0; i < k; i++)
        {
            delete[] solution[i];
        }
        delete[] solution;
        for(int i = 0; i < N; i++)
        {
            delete[] rnks[i];
        }
        delete[] rnks;
        delete[] labels;
    }
    double **show_solution()
    {
        return solution;
    }
    unsigned int *show_labels()
    {
        return labels;
    }
    double l2dist(double *v1, double *v2, int n)
    {
        double x = 0;
        for(int i = 0; i < n; i++)
        {
            x += ((v1[i]-v2[i]) * (v1[i]-v2[i]));
        }
        return x;
    }
    std::vector<double> multiply_constant(double *v, double c, int n)
    {
        std::vector<double> vec;
        for(int i = 0; i < n; i++)
        {
            vec.push_back(v[i] * c);
        }
        return vec;
    }
    std::vector<unsigned int> sample_initial()
    {
        std::vector<unsigned int> target;
        std::vector<unsigned int> ret;
        for(int i = 0; i < N; i++)
        {
            target.push_back(i);
        }
        std::sample(target.begin(), taget.end(), std::back_inserter(ret),
                    k, std::mt19937{std::random_device{}()});
        return ret;
    }
    double* vec2arr(std::vector<double> &v)
    {
        // required C++11 or above
        return v.data();
    }
    void solve(double **data)
    {
        /* Sample data */
        std::vector<unsigned int> sampled = sample_initial();
        for(int i = 0; i < k; i++)
        {
            std::memcpy(solution[i], data[sampled[i]], sizeof(double) * dim);
        }
        for(int i = 0; i < iter; i++)
        {
            Estep(data);
            Mstep(data);
            Assign(data);
        }
    }
    unsigned int argmin(double *x, int n)
    {
        unsigned int x = UINT_MAX;
        double max = DBL_MAX;
        for(unsigned int i = 0; i < static_cast<unsigned int>(n); i++)
        {
            if(max > x[i])
            {
                max = x[i];
                x = i;
            }
        }
        return x;
    }
    void Estep(double **data)
    {
        for(int n = 0; n < N; n++)
        {
            double *xn = data[n];
            for(int k_ = 0; k_ < k; k_++)
            {
                std::vector<double> distances;
                for(int i = 0; i < k; i++)
                {
                    double dist = l2dist(xn, solution[i], dim);
                    distances.push_back(dist);
                }
                unsigned int am = argmin(distances.data(), k);
                if(am == static_cast<unsigned int>(k))
                {
                    rnks[n][k_] = 1;
                }
                else
                {
                    rnks[n][k_] = 0;
                }
            }
            
        }
    }
    void Mstep(double **data)
    {
        for(int k_ = 0; k_ < k; k_++)
        {
            double rnk_sum = 0;
            std::vector<double> uk;
            for(int n = 0; n < N; n++)
            {
                rnk_sum += (double)rnk[n][k_];
                double *xn = data[n];
                std::vector<double> rnk_xn_v = multiply_constant(xn, rnk[n][k_], dim);
                //double *rnk_xn = vec2arr(rnk_xn_v);
                if(uk.size() == 0)
                {
                    std::copy(rnk_xn_v.begin(), rnk_xn_v.end(),
                    std::back_inserter(uk));
                    continue;
                }
                else
                {
                    for(int i = 0; i < dim; i++)
                    {
                        uk[i] = uk[i] / rnk_sum;
                    }
                }
            }
            double *dat = vec2arr(uk);
            std::memcpy(solution[k_] , dat, sizeof(double) * dim);
        }
    }
    void Assign(double **data)
    {
        std::vector<double> distances;
        for(int n = 0; n < N; n++)
        {
            double *xn = data[n];
            for(int k_ = 0; k_ < k; k_++)
            {
                double d = l2dist(xn, solution[k_], dim);
                distances.push_back(d);
            }
            unsigned int am = argmin(distances.data(), k);
            labels[n] = am;
        }
    }
};
#endif
