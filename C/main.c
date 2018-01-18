/*
 * main.c
 * A driver program for kmeans.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kmeans.h"


int main()
{
    double data[20][2] = {{-14,-5},{13,13},{20,23},{-19,-11},{-9,-16},{21,27},{-49,15},{26,13},{-46,5},{-34,-1},{11,15},{-49,0},{-22,-16},{19,28},{-12,-8},{-13,-19},{-41,8},{-11,-6},{-25,-9},{-18,-3}};
    double **tmp = (double **)malloc(sizeof(double *) * 20);
    size_t x = 0;
    fprintf(stdout, "Input data\n");
    for( x = 0; x < 20; x++)
    {
        tmp[x] = (double *)malloc(sizeof(double) * 2);
        memcpy(tmp[x] , data[x], sizeof(double) * 2);
        fprintf(stdout , "[%f,%f]\n", tmp[x][0], tmp[x][1]);
    }
    unsigned int k = 3;
    unsigned int N = 20;
    unsigned int dim = 2;
    unsigned int iter = 1000;
    kmeans_t *engine = kmeans_init(tmp, N, dim, k, iter);
    kmeans_EM(engine, tmp);
    double **solution = kmeans_show_solution(engine);
    size_t i = 0, j = 0;
    fprintf(stdout, "Solution\n");
    for(i = 0; i < k; i++)
    {
        fprintf(stdout , "[");
        for(j = 0; j < dim; j++)
        {
            fprintf(stdout , "%lf ", solution[i][j]);
        }
        fprintf(stdout, "]\n");
    }
    fprintf(stdout, "Free up the memory\n");
    for(x = 0; x < 20; x++)
    {
        free(tmp[x]);
    }
    free(tmp);
    kmeans_destroy(engine);
    return 0;
}
