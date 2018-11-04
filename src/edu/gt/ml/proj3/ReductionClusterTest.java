package edu.gt.ml.proj3;

import java.io.*;

import weka.core.*;
import weka.classifiers.Evaluation;

import weka.classifiers.trees.J48;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.*;
import weka.classifiers.functions.*;

import weka.filters.unsupervised.attribute.*;

import weka.filters.Filter;

public class ReductionClusterTest {

	static String labels[] = {"PCA", "ICA", "RP", "IG"}; //{24, 22};
	static int numAttrs[] = {24, 22, 20, 18, 16, 14, 12, 10, 8, 6};
	static double filterTimes[][] = new double [4][numAttrs.length];
	static double trainingTimes[][] = new double [4][numAttrs.length];
	static double testTimes[][] = new double[4][numAttrs.length];
	static double errorRates[][] = new double[4][numAttrs.length];
	static double sqErrs[][] = new double[4][numAttrs.length];
	static double log[][] = new double[4][numAttrs.length];
	
	public static void training(int type){

		try{

			FileReader trainreader = null; 
			//FileReader testreader = null; new FileReader("C:/Users/cheny1/Documents/GaTech/ML3/data/creditcard-test.arff");

			if (type == 0) {
				trainreader = new FileReader("./data/adult.data.arff");
				System.out.println("adult.data.arff");
				//testreader = new FileReader("C:/Users/cheny1/Documents/GaTech/ML3/data/adult.data.test.arff");
			} else {
				trainreader = new FileReader("./data/creditcard-training.arff");
				System.out.println("creditcard-training.arff");
			}
			
			Instances train = new Instances(trainreader);
			//Instances test = new Instances(testreader);
			train.setClassIndex(train.numAttributes() - 1);
			//test.setClassIndex(test.numAttributes() - 1);


			for (int i = 0; i < numAttrs.length; i++) {

				double start = System.nanoTime(), end, trainingTime;
				Instances newPCATrain = PCA(train, numAttrs[i]);
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);
				filterTimes[0][i] = trainingTime;
				
				start = System.nanoTime();
				Instances newICATrain = ICA(train, numAttrs[i]);
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);
				filterTimes[1][i] = trainingTime;
				
				start = System.nanoTime();
				Instances newRPTrain = RP(train, numAttrs[i]);
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);
				filterTimes[2][i] = trainingTime;
				
				start = System.nanoTime();
				Instances newIGTrain = IG(train, numAttrs[i]);
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);
				filterTimes[3][i] = trainingTime;
				
				
				Instances pca = removeClass(newPCATrain);
				Instances ica = removeClass(newICATrain);
				Instances rp = removeClass(newRPTrain);
				Instances ig = removeClass(newIGTrain);
				
				System.out.println("KMeans: ");
				if (type == 0) {
				runKMeans(pca, 0, i, 20);
				runKMeans(ica, 1, i, 20);
				runKMeans(rp, 2, i, 20);
				runKMeans(ig, 3, i, 20);
				} else {
					runKMeans(pca, 0, i, 30);
					runKMeans(ica, 1, i, 30);
					runKMeans(rp, 2, i, 30);
					runKMeans(ig, 3, i, 30);
					
				}
				System.out.println("EM: ");
				runEM(pca, 0, i, 30);
				runEM(ica, 1, i, 30);
				runEM(rp, 2, i, 30);
				runEM(ig, 3, i, 30);

				trainreader.close();

				//testreader.close();

			} 
		}catch (Exception ex) {
			ex.printStackTrace();
		}
		
	}
	
	public static Instances removeClass(Instances data) {
		Instances dataClusterer = null;
		weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
		filter.setAttributeIndices("" + (data.classIndex() + 1));
		try {
			filter.setInputFormat(data);
			dataClusterer = Filter.useFilter(data, filter);
			return dataClusterer;
		} catch (Exception e1) {
			e1.printStackTrace();
			return null;
		}
	}
	public static void runEM(Instances data, int type, int i, int size) throws Exception {
		EM em = new EM();
		em.setSeed(100);
	

		//important parameter to set: preserver order, number of cluster.
		em.setNumClusters(size);

		em.buildClusterer(data);
		ClusterEvaluation eval = new ClusterEvaluation();
		eval.setClusterer(em);
		eval.evaluateClusterer(data);
		log[type][i] = eval.getLogLikelihood();

		System.out.print(log[type][i] + "\t");
	}

	
	public static void runKMeans(Instances data, int type, int i, int size) throws Exception {
		SimpleKMeans kmeans = new SimpleKMeans();
		
		kmeans.setSeed(100);

		NormalizableDistance df = new EuclideanDistance();
		kmeans.setPreserveInstancesOrder(true);

		kmeans.setDistanceFunction(df);
		kmeans.setNumClusters(size);

		kmeans.buildClusterer(data);

		sqErrs[type][i] = kmeans.getSquaredError();
		System.out.print(sqErrs[type][i] + "\t");
	}
	
	public static Instances PCA(Instances trainingData, int numAttr) throws Exception {
		PrincipalComponents pca = new PrincipalComponents();

		pca.setInputFormat(trainingData);

		pca.setMaximumAttributes(numAttr);
		
		Instances newData = Filter.useFilter(trainingData, pca);
		//pca.
		return newData;
	}

	public static Instances ICA(Instances trainingData, int numAttr) throws Exception {
		IndependentComponents ica = new IndependentComponents();

		ica.setInputFormat(trainingData);

		ica.setOutputNumAtts(numAttr);
		//ica.
		for (int i = 0; i < trainingData.numInstances(); i++) {
			ica.input(trainingData.instance(i));
		}
		ica.batchFinished();
		Instances newData = ica.getOutputFormat();
		Instance processed;
		while ((processed = ica.output()) != null) {
			newData.add(processed);
		}

		return newData;
	
	}

	public static Instances RP(Instances trainingData, int numAttr) throws Exception {
		RandomProjection rp = new RandomProjection();
		rp.setNumberOfAttributes(numAttr);
		rp.setInputFormat(trainingData);
		Instances data = Filter.useFilter(trainingData, rp);
		return data;
	}

	public static Instances RS(Instances trainingData, int numAttr) throws Exception {
		RandomSubset rand=new RandomSubset();
		rand.setInputFormat(trainingData);
		rand.setNumAttributes(numAttr);
		Instances data = Filter.useFilter(trainingData, rand);
		return data;
	}
	public static Instances IG(Instances trainingData, int numAttr) throws Exception {
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		search.setOptions(new String[] { "-T", "0.001" });	// information gain threshold
		search.setNumToSelect(numAttr);
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		
		// apply attribute selection
		attSelect.SelectAttributes(trainingData);
	
		// remove the attributes not selected in the last run
		Instances data = attSelect.reduceDimensionality(trainingData);
		return data;
	}

	public static void printStats(String name, double[][] stats) {
		for (int i = 0; i < 4; i++) {
			System.out.print(name + "\t" + labels[i] + "\t");
			for (int j = 0; j < numAttrs.length; j++) {
				System.out.print(stats[i][j] + "\t");

			}
			System.out.print("\n");
		}
	}
	
	public static void main(String args[]) {

		training(0); 
		System.out.println("\n");
		printStats("FilterTime", filterTimes);
		printStats("SqErrs", sqErrs);
		printStats("Log", log);
		
		training(1); 
		System.out.println("\n");
		printStats("FilterTime", filterTimes);
		printStats("SqErrs", sqErrs);
		printStats("Log", log);
	}
}
