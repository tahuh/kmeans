/*
 * kmeans.h
 * K-means clustering in C language
 * Using EM method
 * Author : Sunghoon Heo
 */
#ifndef _KMEANS_C_
#define _KMEANS_C_
#ifdef __cplusplus
extern "C"
{
#endif

struct kmeans_solution_
{
    double **centers_;
    double **rnks_;
    unsigned int *labels_;
    unsigned int vector_dims_;
    unsigned int num_train_;
    unsigned int k_;
    unsigned int iters_;
};

// The solution struct
typedef struct kmeans_solution_ kmeans_sol_t;
typedef kmeans_sol_t kmeans_t;

kmeans_sol_t *kmeans_init(double **data, unsigned int num, unsigned int dim,
                          unsigned int k, unsigned int iter);
void kmeans_destroy(kmeans_sol_t* s);
double kmeans_distance(double* v1, double* v2, unsigned int dim);
void kmeans_multiply_constant(double *src, double *dst, double c, size_t n);
void kmeans_estep(kmeans_t *sol, double **data);
void kmeans_mstep(kmeans_t *sol, double **data);
void kmeans_EM(kmeans_t *sol, double **data);
void kmeans_assign_label(kmeans_t *sol, double **data);
double **kmeans_show_solution(kmeans_t *sol);
unsigned int *kmeans_show_labels(kmeans_t *sol);
// Helper
unsigned int kmeans_randint(unsigned int n);
unsigned int *kmeans_resorvior(unsigned int n, unsigned int k);
unsigned int *kmeans_sample_initial(unsigned int k);
unsigned int kmeans_argmin(double * ai, unsigned int dim);
#ifdef __cplusplus
}
#endif
#endif
