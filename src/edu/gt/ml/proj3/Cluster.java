package edu.gt.ml.proj3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
 
import weka.clusterers.SimpleKMeans;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.EuclideanDistance;
import weka.core.ManhattanDistance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
 
public class Cluster {
 
	static int seed[] = {10, 50, 100, 200};
	//static int cluster[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18}; 
	static int cluster[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200}; 

	static double sqErr[][] = new double[seed.length][cluster.length];
	static double log[][] = new double[seed.length][cluster.length];
	static double time[][] = new double[seed.length][cluster.length];
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static void runEM(String file) throws Exception {
		System.out.println(file);
		BufferedReader datafile = readDataFile(file); 
		Instances data = new Instances(datafile);
		 
		for (int i = 0; i < seed.length; i++) {
			for (int j = 0; j < cluster.length; j++) {
			
				EM em = new EM();
				em.setSeed(seed[i]);
			

				//important parameter to set: preserver order, number of cluster.
				em.setNumClusters(cluster[j]);

				double start = System.nanoTime(), end, trainingTime;
				em.buildClusterer(data);
				ClusterEvaluation eval = new ClusterEvaluation();
				eval.setClusterer(em);
				eval.evaluateClusterer(data);
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);


				log[i][j] = eval.getLogLikelihood();
				time[i][j] = trainingTime;

				System.out.print(log[i][j] + "\t");
			}
			System.out.print("\n");
		}

		for (int i = 0; i < seed.length; i++) {
			for (int j = 0; j < cluster.length; j++) {
				System.out.print(time[i][j] + "\t");
			}
			System.out.print("\n");
		}

	}
	
	public static void runKmeans(String file, int dist) throws Exception {
		System.out.println(file);

		for (int i = 0; i < seed.length; i++) {
			for (int j = 0; j < cluster.length; j++) {
				SimpleKMeans kmeans = new SimpleKMeans();
				kmeans.setSeed(seed[i]);

				NormalizableDistance df = new EuclideanDistance();
				NormalizableDistance df2 = new ManhattanDistance();
				//important parameter to set: preserver order, number of cluster.
				kmeans.setPreserveInstancesOrder(true);

				if (dist == 1) 
					kmeans.setDistanceFunction(df);
				else
					kmeans.setDistanceFunction(df2);

				kmeans.setNumClusters(cluster[j]);

				BufferedReader datafile = readDataFile(file); 
				Instances data = new Instances(datafile);

				double start = System.nanoTime(), end, trainingTime;
				kmeans.buildClusterer(data);
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);

				sqErr[i][j] = kmeans.getSquaredError();

				time[i][j] = trainingTime;
				System.out.print(sqErr[i][j] + "\t");
			}
			System.out.print("\n");
		}


		for (int i = 0; i < seed.length; i++) {
			for (int j = 0; j < cluster.length; j++) {
				System.out.print(time[i][j] + "\t");
			}
			System.out.print("\n");

		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("Adult Income Kmeans with EuclideanDistance:");
		runKmeans("./data/adult.data.arff", 1);
		System.out.println("Adult Income Kmeans with ManhattanDistance:");
		runKmeans("./data/adult.data.arff", 2);
		System.out.println("Credit Card Kmeans with EuclideanDistance:");
		runKmeans("./data/creditcard-training.arff", 1);
		System.out.println("Credit Card Kmeans with ManhattanDistance:");
		runKmeans("./data/creditcard-training.arff", 2);

		System.out.println("Adult Income EM:");
		runEM("./data/adult.data.arff");
		System.out.println("Credit Card EM:");
		runEM("./data/creditcard-training.arff");
	}
	
}