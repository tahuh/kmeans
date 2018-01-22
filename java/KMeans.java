/*
 * KMeans.java
 * A Java API for K-means clustering
 * Using EM-algorithm
 * Author : Sunghoon Heo
 */
package kmeans;
import java.util.Random;
public class KMeans {
	public static double [][] trainData;
	public static double [][] centers;
	public static int [][] rnks;
	public static int [] labels;
	public static int numTrain;
	public static int dim;
	public static int K;
	public static int iteration;
	/*
	* data : The training set
	* n    : number of training set
	* d    : Dimension of each vector
	* k    : Number of clusters
	* iter : Number of iterations to perform EM-algorithm
	*/
	public KMeans(double [][]data, int n, int d, int k, int iter){
		trainData = new double[n][d];
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < d; j++) {
				trainData[i][j] = data[i][j];
			}
		}
		numTrain = n;
		dim = d;
		K = k;
		iteration = iter;
		labels = new int[k];
		for(int i = 0; i < k; i++) {
			labels[i] = 0;
		}
		rnks = new int[n][k];
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < k; j++) {
				rnks[i][j] = 0;
			}
		}
		InitCenters(); 
	}
	public void Solve(){
		for(int i = 0; i < iteration; i++){
			Estep();
			Mstep();
			Assign();
		}
	}
	public int[] showLabels(){
		return labels;
	}
	public double[][] showSolution(){
		return centers;
	}
	/* Random samples Centers for K-means clustering*/
	public void InitCenters(){
		centers = new double[K][dim];
		for(int i = 0; i < K; i++){
			int index = new Random().nextInt(K);
			double[] C = trainData[index];
			for(int j = 0; j < C.length; j++){
				centers[i][j] = C[j];
			}
		}
	}
	public double euclidean(double[] v1, double[] v2, int n){
		double dist = 0.0;
		for(int i = 0; i < n; i++){
			dist += ( (v1[i]-v2[i]) * (v1[i]-v2[i]) );
		}
		return dist;
	}
	public double[] multiplyConstantToVector(double[] v, double c, int n){
		double [] ret = new double[n];
		for(int i = 0; i < n; i++){
			ret[i] = v[i] * c;
		}
		return ret;
	}
	public int ArgMin(double[] v, int n){
		int loc = 0;
		double dist = Double.MAX_VALUE;
		for(int i = 0; i < n; i++){
			if(dist > v[i]){
				dist = v[i];
				loc = i;
			}
		}
		return loc;
	}
	public void Estep(){
		for(int n = 0; n < numTrain; n++){
			double[] xn = trainData[n];
			for(int k = 0; k < K; k++){
				double[] distances = new double[K];
				for(int i = 0; i < K; i++){
					double dist;
					if (i == k) {
						dist = Double.MAX_VALUE;
					} else { 
						dist = euclidean(xn, centers[i], dim);
					}
					distances[i] = dist;
				}
				int argmin = ArgMin(distances, K);
				if(argmin == k){
					rnks[n][k] = 1;
				}else{
					rnks[n][k] = 0;
				}
			}
		}
	}
	public void Mstep(){
		int n = 0;
		int k = 0;
		int i = 0;
		double rnk_sum = 0.0;
		for(k = 0; k < K; k++){
			rnk_sum = 0.0;
			double[] uk = new double[dim];
			for(n = 0; n < numTrain; n++){
				rnk_sum += rnks[n][k];
				double[] xn = trainData[n];
				double[] rnk_xn = multiplyConstantToVector(xn, rnks[n][k], dim);
				for(i = 0; i < dim; i++){
					uk[i] = uk[i] + rnk_xn[i];
				}
			}
			for(i = 0; i < dim; i++){
				uk[i] = uk[i] / rnk_sum;
			}
			for(i = 0; i < K; i++){
				for(int j = 0; j < dim; j++){
					centers[i][j] = uk[j];
				}
			}
		}
	}
	public void Assign(){
		int n = 0;
		int k = 0;
		for(n = 0; n < numTrain; n++){
			double[] xn = trainData[n];
			double[] distances = new double[K];
			for(k = 0; k < K; k++){
				double dist = euclidean(xn, centers[k], dim);
				distances[k] = dist;
			}
			int argmin = ArgMin(distances, K);
			labels[n] = argmin;
		}
	}
}
