/*
 * kmeans.c
 * K-means clustering in C language
 * Using EM method
 * Author : Sunghoon Heo
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"

// initialize solution block
kmeans_sol_t *kmeans_init(double **data, unsigned int num, unsigned int dim,
                          unsigned int k, unsigned int iter)
{
    kmeans_t *sol = (kmeans_t*)malloc(sizeof(kmeans_t));
    // Constant domain
    sol->vector_dims_ = dim;
    sol->k_ = k;
    sol->iters_ = iter;
    sol->num_train_ = num;
    // Memory allocated domain
    // k x dim marix
    sol->centers_ = (double **)malloc(sizeof(double *) * k);
    size_t i = 0;
    //unsigned int *samples = kmeans_sample_initial(k);
    unsigned int *samples = kmeans_resorvior(num, k);
    // Randomly select centers
    for(i = 0; i < k; i++)
    {
        sol->centers_[i] = (double *)malloc(sizeof(double) * dim);
        memcpy(sol->centers_[i], data[samples[i]], sizeof(double) * dim);
#ifdef DEBUG
        size_t ppp = 0;
        fprintf(stdout, "Randomly selected centers - %d\n", samples[i]);
        for(ppp = 0; ppp < dim; ppp++)
        {
            fprintf(stdout, "%f ", sol->centers_[i][ppp]);
        }
        fprintf(stdout, "\n");
#endif
    }
    // release memory
    free(samples);
    sol->rnks_ = (double **)malloc(sizeof(double *) * num);
    for(i = 0; i < num; i++)
    {
        sol->rnks_[i] = (double *)malloc(sizeof(double) * k);
        memset(sol->rnks_[i], 0.0, sizeof(double) * k);
    }
    sol->labels_ = (unsigned int *)malloc(sizeof(unsigned int) * num);
    memset(sol->labels_, 0, sizeof(unsigned int) * num);
    return sol;
}

void kmeans_destroy(kmeans_sol_t *sol)
{
    free(sol->labels_);
    size_t i = 0;
    for(i = 0; i < sol->k_; i++)
    {
        free(sol->centers_[i]);
    }
    free(sol->centers_);
    for(i = 0; i < sol->num_train_; i++)
    {
        free(sol->rnks_[i]);
    }
    free(sol->rnks_);
    free(sol);
}

double kmeans_distance(double *v1, double *v2, unsigned int dim)
{
    size_t i = 0;
    double dist = 0.0;
    for( i = 0; i < dim; i++)
    {
        dist += ((v1[i] - v2[i]) * (v1[i] - v2[i]));
    }
    return dist;
}

void kmeans_multiply_constant(double *src, double* dst, double c, size_t n)
{
    size_t i = 0;
    // dst must be pre-allocated
    //memcpy(dst, src, n * sizeof(double));
    for(i = 0; i < n; i++)
    {
        dst[i] = src[i] * c;
    }
}

// data : allocated outside
// sol  : knn solution box
void kmeans_estep(kmeans_t *sol, double **data)
{
    unsigned int N = sol->num_train_;
    unsigned int K = sol->k_;
    size_t n = 0;
    size_t k = 0;
    size_t j = 0;
    double *xn;
    double *distances = (double *)malloc(sizeof(double) * sol->k_);
    double dist = 0.0;
    unsigned int argmin;
    memset(distances, 0.0, sizeof(double) * sol->k_);
    for(n = 0; n < N; n++)
    {
        xn = data[n];
        for(k = 0; k < K; k++)
        {
            memset(distances, 0.0, sizeof(double) * sol->k_);
            for(j = 0; j < K; j++)
            {
                if( j == k )
                {
                    dist = FLT_MAX;
                }
                dist = kmeans_distance(xn, sol->centers_[j], sol->vector_dims_);
                distances[j] = dist;
            }
            argmin = kmeans_argmin(distances, K);
            if(argmin == k)
            {
                sol->rnks_[n][k] = 1;
            }
            else
            {
                sol->rnks_[n][k] = 0;
            }
        }
    }
    free(distances);
}
void kmeans_mstep(kmeans_t *sol, double **data)
{
    unsigned int N = sol->num_train_;
    unsigned int K = sol->k_;
    unsigned int dim = sol->vector_dims_;
    unsigned int n = 0;
    unsigned int k = 0;
    unsigned int i = 0;
    //double **uks = (double **)malloc(sizeof(double *) * K);
    double *uk = (double *)malloc(sizeof(double) * dim);
    double *rnk_xn = (double *)malloc(sizeof(double) * dim);
    double *xn = NULL;
    double rnk_sum = 0.0;
    memset(uk, 0.0, sizeof(double) * dim);
    for(k = 0; k < K; k++)
    {
        rnk_sum = 0.0;
        memset(uk, 0.0, sizeof(double) * dim);
        for(n = 0; n < N; n++)
        {
            rnk_sum += sol->rnks_[n][k];
            xn = data[n];
            kmeans_multiply_constant(xn, rnk_xn, sol->rnks_[n][k], dim);
            for(i = 0; i < dim; i++)
            {
                uk[i] = uk[i] + rnk_xn[i];
            }
        }
        for(i = 0; i < dim; i++)
        {
            uk[i] = uk[i] / rnk_sum;
        }
        memcpy(sol->centers_[k], uk, sizeof(double) * dim);
    }   

    // memory release
    free(uk);
    free(rnk_xn);
}
void kmeans_EM(kmeans_t *sol, double **data)
{
    size_t i = 0;
#ifdef DEBUG
    fprintf(stdout , "Iterate over %d times\n", sol->iters_);
#endif
    for(i = 0; i < sol->iters_; i++)
    {
        kmeans_estep(sol, data);
        kmeans_mstep(sol, data);
        kmeans_assign_labels(sol, data);
    }
}

void kmeans_assign_labels(kmeans_t *sol, double **data)
{
    unsigned int n = 0;
    unsigned int k = 0;
    unsigned int K = sol->k_;
    unsigned int N = sol->num_train_;
    double *distances = (double *)malloc(sizeof(double) * K);
    double *xn;
    unsigned int dim = sol->vector_dims_;
    unsigned int argmin;
    double dist;
    memset(distances, 0.0, sizeof(double) * K);
    for(n = 0; n < N; n++)
    {
        for(k = 0; k < K; k++)
        {
            dist = kmeans_distance(xn, sol->centers_[k], dim);
            distances[k] = dist;
        }
        argmin = kmeans_argmin(distances, K);
        sol->labels_[n] = argmin;
    }
    free(distances);
}

double ** kmeans_show_solution(kmeans_t *sol)
{
    return sol->centers_;
}

unsigned int *kmeans_show_label(kmeans_t *sol)
{
    return sol->labels_;
}
// Generates uniform random integer between [0,n)
// https://stackoverflow.com/questions/822323/how-to-generate-a-random-number-in-c
unsigned int kmeans_randint(unsigned int n)
{
    if((n-1) == RAND_MAX)
    {
        return rand();
    }
    else
    {
        long end = RAND_MAX / n;
        assert(end > 0L);
        end *= n;
        int r;
        while((r = rand()) >= end);
        return r % n;
    }
}

unsigned int *kmeans_resorvior(unsigned int n, unsigned int k)
{
    int i = 0; int j;
    unsigned int *res = (unsigned int*)malloc(sizeof(unsigned int) * k);
    for(i = 0; i < k; i++)
    {
        res[i] = i;
    }

    srand(0);
    for(; i < n; i++)
    {
        j = rand() % (i+1);
        if(j < k)
        {
            res[j] = i;
        }
    }
    return res;
}

unsigned int *kmeans_sample_initial(unsigned int k)
{
    unsigned int *numbers = (unsigned int *)malloc(sizeof(unsigned int) * k);
    memset(numbers, 0, sizeof(int) * k);
    int i = 0;
    int j = 0;
    int cnt = 0;
    unsigned int rn, v;
    for(i = 0; i < k; i++)
    {
        rn = kmeans_randint(k);
        if( i == 0)
        {
            numbers[i] = rn;
            continue;
        }
        cnt = k;
        while(1)
        {
            cnt = 0;
            for(j = 0; j < i; j++)
            {
                v = numbers[j];
                if(v == rn)
                {
                    cnt++;
                }
            }
            if(cnt == k)
            {
                rn = kmeans_randint(k);
                continue;
            }
            else
            {
                break;
            }
        }
        numbers[i] = rn;
    }
    return numbers;
}
unsigned int kmeans_argmin(double *a, unsigned int dim)
{
    unsigned int i = 0;
    unsigned int argmin = UINT_MAX;
    double value = FLT_MAX;
    for(i = 0; i < dim; i++)
    {
        if(a[i] < value)
        {
            value = a[i];
            argmin = i;
        }
    }
    return argmin;
}
