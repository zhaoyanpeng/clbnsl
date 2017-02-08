package edu.shanghaitech.ai.ml.clbnsl.bnsl.main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.curricula.Curricula;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.estimator.CLBayesNetEstimator;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.search.local.CLSearcher;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.search.local.TabuSearcher;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.BayesNet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.converters.ArffLoader;

/**
 * To be used to reproduce the results reported in the paper.
 * 
 * @author Yanpeng Zhao
 * 3/25/2015
 */
@SuppressWarnings("unused")
public class ExperimentSet {
	
	private String arrfFilePath;
	private Instances instances;
	private ArffLoader arffLoader;
	private BayesNet bayesNet;

	private Instances testSet;
	private String gtHeader = null;
	CLBayesNetEstimator ec = null;
	
	
	public ExperimentSet() {
		this.arffLoader = new ArffLoader();
	}
	
	
	public ExperimentSet(String arrfFilePath) {
		this.arrfFilePath = arrfFilePath;
		this.arffLoader = new ArffLoader();

		initInstances();
	}
	
	
	protected void updateInstances(String arrfFilePath) {
		this.arrfFilePath = arrfFilePath;
		initInstances();
	}
	
	
	protected void updateEstimator(String gtHeader, String gtStruct, String testSetPath) throws Exception {
		this.gtHeader = gtHeader;
		this.ec = new CLBayesNetEstimator(
				gtHeader, gtStruct, AUtils.ESS, true, AUtils.STRUCT_FROM_ALIST); 
		try {
			File file = new File(testSetPath);
			arffLoader.setFile(file);
			this.testSet =  arffLoader.getDataSet();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			this.testSet =  null;
		}
	}
	
	
	protected void initInstances() {
		try {
			File file = new File(this.arrfFilePath);
			arffLoader.setFile(file);
			this.instances = arffLoader.getDataSet();
			
			// @debug
			System.out.println("Instances are ready..Good luck!");
			
			// must be done before everything 
			this.instances.setClassIndex(this.instances.numAttributes() - 1);
			
			 /*  BayesNet  */
	     	this.bayesNet = new BayesNet();
		} catch ( Exception e ) {
			e.printStackTrace();
		}
	}
	
	
	protected void getAverageRunningTime(double[] times, String timeFile) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(timeFile)));
			String line = reader.readLine();
			String[] timesTmp = line.split(" ");
			if ( timesTmp.length != times.length ) {
				System.out.println("err.");
			}
			for ( int i = 0; i < times.length; i++ ) {
				times[i] = Double.valueOf(timesTmp[i]);
			}
			reader.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("time file: " + timeFile);
			e.printStackTrace();
		}
	}
	
	
    protected void getPCSets(Vector<Vector<Integer>> pcSets, String pcSetsPath) {
    	try {
    		pcSets.clear(); // insure
			BufferedReader reader = new BufferedReader(new FileReader(new File(pcSetsPath)));
			
			String line = "";
			String[] tmp = null;
			for ( int i = 0; i < instances.numAttributes(); i++ ) {
				Vector<Integer> pcSet = new Vector<Integer>();
				line = reader.readLine();
				tmp = line.split("\t");
				for ( int j = 0; j < tmp.length; j++ ) {
					if ( !tmp[j].trim().equals("") ) {
						pcSet.add(new Integer(tmp[j].trim()));
					}
				}
				pcSets.add(pcSet);
			}
			reader.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("PCSetFile: " + pcSetsPath);
			e.printStackTrace();
		}
    }
    
    
    protected void getFullConnectedDAG(ArrayList<ArrayList<Integer>> fcDAG, String fcDAGPath, int nVariable) {
    	try {
    		fcDAG.clear(); // insure
			BufferedReader reader = new BufferedReader(new FileReader(new File(fcDAGPath)));
			
			String line = "";
			String[] tmp = null;
			for ( int i = 0; i < nVariable; i++ ) {
				ArrayList<Integer> pcSet = new ArrayList<Integer>();
				line = reader.readLine();
				tmp = line.split(" ");
				for ( int j = 0; j < tmp.length; j++ ) {
					if ( !tmp[j].trim().equals("") ) {
						pcSet.add(new Integer(tmp[j].trim()));
					}
				}
				fcDAG.add(pcSet);
			}
			reader.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("fcDAGPath: " + fcDAGPath);
			e.printStackTrace();
		}
    }
	

    public void experientOnDS(
			int seed, 
			int[] scales, 
			boolean bDebug,
			String basePath) throws Exception {
    	
    	// the reported results are produced with the third configuration (see boolean values below)
    	int nAlg = 3;
    	
    	// customized according to the datasets in experiments
    	int[] iAlgs   = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
//    	int[] iAlgs   = {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//    	int[] iAlgs   = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    	int[] iScales = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//    	int[] nScales = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
//    	int[] iScales = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    	int[] nScales = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    	int[] iSteps  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//    	int[] nSteps  = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    	int[] nSteps  = {4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    	
    	// whether the following settings are used or not depends on the configuration of the algorithm 
    	int[] nMaxPars= {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
    	int[] nCandidates = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
    	
    	// each combination of the following boolean values represent an configuration of the algorithm
    	boolean[] partition = {true,  true,  true,  true,  true,  true};
    	boolean[] random    = {false, false, false, false, false, false};
    	boolean[] candidate = {true,  true,  true,  true,  false, false};
    	boolean[] limitadd  = {false, false, false, false,  false, true};
    	
    	// curriculum sequence can be produced under different metrics
    	int[] clmethod  = {    0,     0,    0,     0,     0,     0};
    	
    	// customized, here we just experiment on the first dataset
    	int iTest = 0; 
    	int nTest = 1;
    	
    	// dataset names
    	String[] names = AUtils.nameOfDS;
    	
    	for ( int ii = iTest; ii < nTest; ii++ ) {
    		
    		int iScale  = iScales[ii],  nScale = nScales[ii];
    		int iStep   = iSteps[ii],   nStep  = nSteps[ii];
    		int nParent = nMaxPars[ii], nCandidate = nCandidates[ii];
    		
    		String nameOfDS = names[ii];
    		
    		String gtHeader = basePath + "res/" + nameOfDS + "/" + nameOfDS + "_" + AUtils.FSUFFIX_HEADER;
    		String gtStruct = basePath + "res/" + nameOfDS + "/" + nameOfDS + "_" + AUtils.FSUFFIX_GT;
    		String testSetPath = basePath + "res/" + nameOfDS + "/seed" + 
    				String.valueOf(5) + "/" + String.valueOf(5000) + "/" + 
    				nameOfDS + String.valueOf(5000) + AUtils.FSUFFIX_ARFF;
    		
    		updateEstimator(gtHeader, gtStruct, testSetPath);
    		
    		String prefix = "_AVERAGE_";
     	 	String timeFile = basePath + "res/" + nameOfDS + "/" + nameOfDS + prefix + AUtils.FSUFFIX_RUNTIME;
     	 	double[] times = new double[6];
     	 	getAverageRunningTime(times, timeFile);
     	 	
//     	 	for ( int i = 0; i < 6; i++ ) {
//     	 		System.out.print(times[i] + " ");
//     	 	}
//     	 	System.out.println();
//     	 	System.exit(0);
    		
    		// read into fully connected DAG
     	 	String fcdagid = "_0_";
     	 	String fcDAGPath = basePath + "res/" + nameOfDS + "/" + 
     	 			nameOfDS + fcdagid + AUtils.FSUFFIX_FNDAG;
     	 	ArrayList<ArrayList<Integer>> fcDAG = new ArrayList<ArrayList<Integer>>();
     	 	getFullConnectedDAG(fcDAG, fcDAGPath, AUtils.N_VARIABLE[ii]);
     	 	
//     	 	for ( int i = 0; i < fcDAG.size(); i++ ) {
//	 	 		System.out.println(fcDAG.get(i));
//	 	 	}
//	 	 	System.exit(0);

    		for ( int jj = iAlgs[ii]; jj < nAlg; jj++ ) {
    			
    			boolean uPartition = partition[jj];
    			boolean uRandom    = random[jj]; 
    			boolean uCandidate = candidate[jj];
    			boolean uLimitadd  = limitadd[jj];
    			int iMethod         = clmethod[jj];

    			for ( int kk = iScale; kk < nScale; kk++ ) {
    				
    				String scale = String.valueOf((scales[kk]));
    				String arrfFilePath = basePath + "res/" + nameOfDS + "/seed" + 
    						String.valueOf(seed) + "/" + scale + "/" + nameOfDS + 
    						scale + AUtils.FSUFFIX_ARFF;
    				
    				this.updateInstances(arrfFilePath);
    				this.bayesNet.setTimeBoundary(times[kk - iScale]);
    				
//    				System.out.print(times[kk - iScale] + " ");
//    				continue;
    				
    				// @debug
    				System.out.print("->" + nameOfDS);
    				/*if ( uCandidate ) { System.out.print(" nCandidate " +  nCandidate); }*/
    				System.out.println(" iAlg " + jj + " scale " + scale + " seed " + seed);
    				
    				this.trainBayesNN(
    						seed,
    						iStep,
    						nStep, 
    						jj, 
    						nCandidate, 
    						iMethod,
    						nameOfDS, 
    						scale, 
    						basePath, 
    						bDebug, 
    						uRandom,
    						uPartition, 
    						uCandidate,
    						uLimitadd,
    						fcDAG);
    			}
    		}
    	}
    }
    
    public void trainBayesNNTabu(
    		int seed,
    		int iStep,
			int nStep, 
	    	int iAlgorithm,
	    	int nCandidate,
	    	int iMethod,
	    	String nameOfDS,
			String scaleOfDS, 
			String basePath,
			boolean bDebug,
			boolean bUseRandom,
			boolean bPartition,
			boolean bCandidate,
			boolean bLimitadd,
			ArrayList<ArrayList<Integer>> fcDAG) throws Exception {
	 	
	 	FileWriter outfile;
	 	int nFirstAdded = 3;
	 	String IA = AUtils.getAlgorithmFlag(iAlgorithm) + String.valueOf((int) AUtils.ESS_CL) + "_";

	 	String path = basePath + "res/" + nameOfDS + "/seed" + String.valueOf(seed) + "/" + scaleOfDS + "/";
	 	
	 	// obviously, it's not valid
	 	int initStepLength = -1;
	 	
	 	// curriculum list
     	int[] listOfCL = null;
     	int[][] candidateSet = null;
     	
     	// read into parents & children set
     	String pcsetid = "_MMHC_";
 	 	String pcSetsPath = path + nameOfDS + pcsetid + AUtils.FSUFFIX_PCSET;
 	 	Vector<Vector<Integer>> pcSets = new Vector<Vector<Integer>>();
 	 	getPCSets(pcSets, pcSetsPath);
 	 	
// 	 	for ( int i = 0; i < pcSets.size(); i++ ) {
// 	 		System.out.println(pcSets.get(i));
// 	 	}
// 	 	System.exit(0);	
 	 
     	LocalScoreSearchAlgorithm searcher = new TabuSearcher(
     			path + nameOfDS,
     			instances, 
     			initStepLength, 
     			nFirstAdded,
     			candidateSet,
     			bDebug,
     			bUseRandom,
     			bPartition, 
     			bCandidate,
     			bLimitadd);
     	
     	searcher.setHowToMakeScore(AUtils.SCORE_EMPTY);
 		searcher.setPCSets(pcSets);
 		searcher.setFullConnectedDAG(fcDAG);
 		
     	bayesNet.setSearchAlgorithm(searcher);

	 	// train using different step
     	int SHD = Integer.MAX_VALUE;
     	int M = 0, E = 0, R = 0, RC = 0, step = -1;
     	int preM = 0, preE = 0, preR = 0, preRC = 0;
     	
     	String learntNet = null;
     	double BDeuScore = -Double.MAX_VALUE, buildTime = 0;
     	
     	// training and timing
     	long startTime = System.currentTimeMillis();
     	
		for ( int i = iStep; i < nStep; i++ ) {
			
			int stepLength = i;
			// update step length and CL list
			bayesNet.updateSearchAlgorithm(0);
	 		bayesNet.buildClassifier(this.instances);

	 		/* write learned net */
	 		learntNet = path + nameOfDS + "_step_" + 
	     			String.valueOf(stepLength) + IA + AUtils.FSUFFIX_LN;
	 		outfile = new FileWriter(learntNet);
	 		outfile.write((bayesNet.toString()));
	 		outfile.close();
	 		
	 		// @debug
			System.out.println("step " + i + " over..");
			
			/*evaluation*/
			ec.updateLearnednet(gtHeader, learntNet, AUtils.STRUCT_FROM_ALIST); 
			ec.evaluate(testSet); // evaluate learned net
			M = ec.getNMissingArc();
			E = ec.getNExtraArc();
			R = ec.getNReversedArc(true);
			RC = ec.getNReversedArc(false);		
			
			if ( BDeuScore < ec.getBDeuScore() ) {
				SHD = M + E + RC;
				preM = M;
				preE = E;
				preR = R;
				preRC = RC;
				step = stepLength;
				BDeuScore = ec.getBDeuScore();
			}	
		}
		
		long endTime = System.currentTimeMillis();
		buildTime = (double)(endTime - startTime) / 1000;
		
		String shd = "M: " + String.valueOf(preM) + " E: " + String.valueOf(preE) + 
					" R: " + String.valueOf(preR) + " RC: " + String.valueOf(preRC) + "\n";
		
		// BDeuScore and Consumed time @ step
		String scoreAndTime = path + nameOfDS + IA + AUtils.FSUFFIX_ST;
		outfile = new FileWriter(scoreAndTime);
		outfile.write(BDeuScore + " & " + buildTime + " @ " + 0 + "\n");
		outfile.close();
		
		// learned optimal shd
		String shdDetail = path + nameOfDS + IA + AUtils.FSUFFIX_SHD;
		outfile = new FileWriter(shdDetail);
		outfile.write(shd);
		outfile.close();
    }
    
    
    public void trainBayesNN(
    		int seed,
    		int iStep,
    		int nStep, 
	    	int iAlgorithm,
	    	int nCandidate,
	    	int iMethod,
	    	String nameOfDS,
			String scaleOfDS, 
			String basePath,
			boolean bDebug,
			boolean bUseRandom,
			boolean bPartition,
			boolean bCandidate,
			boolean bLimitadd,
			ArrayList<ArrayList<Integer>> fcDAG) throws Exception {
		
    	if ( nStep < 0 ) { System.exit(0); }
	 	
	 	FileWriter outfile;
	 	int nFirstAdded = 3;
	 	String IA = AUtils.getAlgorithmFlag(iAlgorithm) + String.valueOf((int) AUtils.ESS_CL) + "_";

	 	String path = basePath + "res/" + nameOfDS + "/seed" + String.valueOf(seed) + "/" + scaleOfDS + "/";
	 	
	 	// obviously, it's not valid
	 	int initStepLength = -1;
	 	
	 	// curriculum list
	 	int[] listOfCL = null;
	 	int[][] candidateSet = null;
	 	if ( nStep > 1 ) {
		 	Curricula curricula = new Curricula(false, instances, nCandidate);
	     	listOfCL = curricula.getCList(iMethod);
//	     	candidateSet = curricula.getCandidateSet(iMethod);
	 	}

     	
     	// read into parents & children set
     	String pcsetid = "_MMHC_";
 	 	String pcSetsPath = path + nameOfDS + pcsetid + AUtils.FSUFFIX_PCSET;
 	 	Vector<Vector<Integer>> pcSets = new Vector<Vector<Integer>>();
 	 	getPCSets(pcSets, pcSetsPath);
 	 	
// 	 	for ( int i = 0; i < pcSets.size(); i++ ) {
// 	 		System.out.println(pcSets.get(i));
// 	 	}
// 	 	System.exit(0);	
 	 
     	LocalScoreSearchAlgorithm searcher = new CLSearcher(
     			path + nameOfDS,
     			instances, 
     			initStepLength, 
     			nFirstAdded,
     			candidateSet,
     			bDebug,
     			bUseRandom,
     			bPartition, 
     			bCandidate,
     			bLimitadd);
     	
     	// use penalty or not
     	searcher.setHowToMakeScore(AUtils.SCORE_DEFAULT);
 		searcher.setPCSets(pcSets);
 		searcher.setFullConnectedDAG(fcDAG);
 		
     	bayesNet.setSearchAlgorithm(searcher);

	 	// train using different step
     	int SHD = Integer.MAX_VALUE;
     	int M = 0, E = 0, R = 0, RC = 0, step = -1;
     	int preM = 0, preE = 0, preR = 0, preRC = 0;
     	
     	String learntNet = null;
     	double BDeuScore = -Double.MAX_VALUE, buildTime = 0;
     	
     	// training and timing
     	long startTime = System.currentTimeMillis();
     	
		for ( int i = iStep; i < nStep; i++ ) {
			
			int stepLength = i;
			// update step length and CL list
			bayesNet.updateSearchAlgorithm(stepLength, listOfCL);
	 		bayesNet.buildClassifier(this.instances);

	 		/* write learned net */
	 		learntNet = path + nameOfDS + "_step_" + 
	     			String.valueOf(stepLength) + IA + AUtils.FSUFFIX_LN;
	 		outfile = new FileWriter(learntNet);
	 		outfile.write((bayesNet.toString()));
	 		outfile.close();
	 		
	 		// @debug
			System.out.println("step " + i + " over..");
			
			/*evaluation*/
			ec.updateLearnednet(gtHeader, learntNet, AUtils.STRUCT_FROM_ALIST); 
			ec.evaluate(this.testSet); // evaluate learned net
			M = ec.getNMissingArc();
			E = ec.getNExtraArc();
			R = ec.getNReversedArc(true);
			RC = ec.getNReversedArc(false);		
			
			if ( BDeuScore < ec.getBDeuScore() ) {
				SHD = M + E + RC;
				preM = M;
				preE = E;
				preR = R;
				preRC = RC;
				step = stepLength;
				BDeuScore = ec.getBDeuScore();
			}	
		}
		
		long endTime = System.currentTimeMillis();
		buildTime = (double)(endTime - startTime) / 1000;
		
		String shd = "M: " + String.valueOf(preM) + " E: " + String.valueOf(preE) + 
					" R: " + String.valueOf(preR) + " RC: " + String.valueOf(preRC) + "\n";
		
		// BDeuScore and Consumed time @ step
		String scoreAndTime = path + nameOfDS + IA + AUtils.FSUFFIX_ST;
		outfile = new FileWriter(scoreAndTime);
		outfile.write(BDeuScore + " & " + buildTime + " @ " + step + "\n");
		outfile.close();
		
		// learned optimal shd
		String shdDetail = path + nameOfDS + IA + AUtils.FSUFFIX_SHD;
		outfile = new FileWriter(shdDetail);
		outfile.write(shd);
		outfile.close();
    }
    
    
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		int seed = 1;
		int nScale = 8;
		boolean bDebug = false;
		int[] scales = AUtils.getScaleSequence(0, nScale);
		String basePath = "E:/NewData/";
		ExperimentSet ExpSet = new ExperimentSet();
		
		long startTime = System.currentTimeMillis();
/*		
		for ( int i = 0; i < 1; i++ ) {
			
			seed = i;
			ExpSet.experientOnDS(
					seed, 
					scales,
					bDebug,
					basePath);
		}
*/		
 		long endTime = System.currentTimeMillis();
 		double totalTime = (double)(endTime - startTime) / 1000;
 		System.out.println("Time on Training: " + totalTime + "s");

	}

}

