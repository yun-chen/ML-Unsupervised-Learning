package edu.gt.ml.proj3;


import java.io.*;

import weka.core.*;

import weka.core.Instances;

import weka.classifiers.Evaluation;

import weka.classifiers.trees.J48;

import weka.classifiers.*;
import weka.classifiers.functions.*;

import weka.filters.unsupervised.attribute.*;
import weka.filters.unsupervised.attribute.RandomSubset;
import weka.filters.Filter;
import weka.attributeSelection.*;

public class SimpleNNTest{

	static String labels[] = {"PCA", "ICA", "RP", "IG"};
	static int numAttrs[] = {24, 22, 20, 18, 16, 14, 12, 10, 8, 6};
	static double filterTimes[][] = new double [4][numAttrs.length];
	static double trainingTimes[][] = new double [4][numAttrs.length];
	static double testTimes[][] = new double[4][numAttrs.length];
	static double errorRates[][] = new double[4][numAttrs.length];
	
	public static void trainingBase(){

		try{

			FileReader trainreader = new FileReader("./data/creditcard-training.arff");

			FileReader testreader = new FileReader("./data/creditcard-test.arff");

			//FileReader trainreader = new FileReader("./data/adult.data.arff");

			//FileReader testreader = new FileReader("./data/adult.data.test.arff");
			Instances train = new Instances(trainreader);
			Instances test = new Instances(testreader);
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);
			
			trainNN(train, test, 0, 0);
			System.out.println("Base NN: Training Time:" + trainingTimes[0][0]);
			System.out.println("Base NN: Testing Time:" + testTimes[0][0]);
			System.out.println("Base NN: Error rate:" + errorRates[0][0]);
			
			trainreader.close();
			testreader.close();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
		
	public static void training(){

		try{

			FileReader trainreader = new FileReader("C:/Users/cheny1/Documents/GaTech/ML3/data/creditcard-training.arff");

			FileReader testreader = new FileReader("C:/Users/cheny1/Documents/GaTech/ML3/data/creditcard-test.arff");

			//FileReader trainreader = new FileReader("C:/Users/cheny1/Documents/GaTech/ML3/data/adult.data.arff");

			//FileReader testreader = new FileReader("C:/Users/cheny1/Documents/GaTech/ML3/data/adult.data.test.arff");
			Instances train = new Instances(trainreader);
			Instances test = new Instances(testreader);
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);


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
				//Instances newRSTrain = RS(train, numAttrs[i]);
				Instances newIGTrain = IG(train, numAttrs[i]);
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10,9);
				filterTimes[3][i] = trainingTime;
				
				Instances newPCATest = PCA(test, numAttrs[i]);
				Instances newICATest = ICA(test, numAttrs[i]);
				Instances newRPTest = RP(test, numAttrs[i]);
				//Instances newRSTest = RS(test, numAttrs[i]);
				Instances newIGTest = IG(test, numAttrs[i]);

				trainNN(newPCATrain, newPCATest, 0, i);
				trainNN(newICATrain, newICATest, 1, i);
				trainNN(newRPTrain, newRPTest, 2, i);
				trainNN(newIGTrain, newIGTest, 3, i);
				
				trainreader.close();

				testreader.close();

			} 
		}catch (Exception ex) {
			ex.printStackTrace();
		}
		
	}
	
	public static void trainNN(Instances train, Instances test, int i, int j) {
		try {
		
			MultilayerPerceptron mlp = new MultilayerPerceptron();
			//Setting Parameters
			mlp.setLearningRate(0.3);
			mlp.setMomentum(0.2);
			//mlp.setTrainingTime(2000);
			//mlpsetHiddenLayers(3);
			
			double start = System.nanoTime(), end, trainingTime;
			mlp.buildClassifier(train);
			
			end = System.nanoTime();
			trainingTime = end - start;
			trainingTime /= Math.pow(10,9);
			
			System.out.println("Trainig time: " + trainingTime);
			trainingTimes[i][j] = trainingTime;
			
			// evaluate classifier and print some statistics

			Evaluation eval = new Evaluation(train);

			start = System.nanoTime();
			eval.evaluateModel(mlp, test);
			end = System.nanoTime();
			trainingTime = end - start;
			trainingTime /= Math.pow(10,9);
			System.out.println("Testing time: " + trainingTime);
			
			testTimes[i][j] = trainingTime;
			
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));

			System.out.println("Error: " + eval.errorRate());

			errorRates[i][j] = eval.errorRate();
			
		} catch(Exception ex){

			ex.printStackTrace();

		}

	}

	public static Instances PCA(Instances trainingData, int numAttr) throws Exception {
		weka.filters.unsupervised.attribute.PrincipalComponents pca = new weka.filters.unsupervised.attribute.PrincipalComponents();

		pca.setInputFormat(trainingData);

		pca.setMaximumAttributes(numAttr);
		
		Instances newData = Filter.useFilter(trainingData, pca);

		return newData;
	}

	public static Instances ICA(Instances trainingData, int numAttr) throws Exception {
		IndependentComponents ica = new IndependentComponents();

		ica.setInputFormat(trainingData);

		ica.setOutputNumAtts(numAttr);
		
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
		trainingBase(); 
		training(); 
		printStats("FilterTime", filterTimes);
		printStats("TrainTime", trainingTimes);
		printStats("TestingTime", testTimes);
		printStats("ErrorRate", errorRates);
	}
}