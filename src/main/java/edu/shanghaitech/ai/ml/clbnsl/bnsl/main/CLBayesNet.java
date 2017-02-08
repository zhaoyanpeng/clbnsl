package edu.shanghaitech.ai.ml.clbnsl.bnsl.main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Vector;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.curricula.Curricula;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.curricula.MMPC;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.estimator.CLBayesNetEstimator;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.search.local.CLSearcher;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.BayesNet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * Curriculum Learning of Bayesian Network Structures. Given dataset in the format of ARFF as the input,
 * the CL algorithm outputs the most probable Bayesian Network Structure. Note that the algorithm needs
 * MMPC to provide candidate parents and children. The current implementation of MMPC is not as good as
 * <a href="http://discover.mc.vanderbilt.edu/discover/public/causal_explorer/index.html">Causal Explorer</a>.
 * 
 * @author Yanpeng Zhao
 * 7/17/2016
 */
public class CLBayesNet {
	
	private class Option {
		// root directory
		protected static final String ROOT = 
					"E:/NewData";
		// ensure the format of ARFF
		protected static final String FIN  = 
					"alarm10000.arff";
		// to identify a run
		protected static final String ID_RUN = 
					"20160718";
		
		
		// please leave it as it is
		protected static final int N_CANDIDATE = 5;
		// range of the step size [0, N_STEP]
		protected static final int N_STEP = 4;
		// please leave it as it is 
		protected static final int INIT_STEP = -1;
		// the first curriculum
		protected static final int N_FIRSTADDED = 3;
		
		
		// partition the training data
		protected static final boolean B_PARTITION = true;
		// please leave it as it is
		protected static final boolean B_RANDOM    = false;
		// choose the parents from the candidate set
		protected static final boolean B_CANDIDATE = true;
		// limit the max number of parents of the node
		protected static final boolean B_LIMITADD  = false;
		// print running informations
		protected static final boolean B_DEBUG     = false;
		
	}

	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		String fin  = Option.ROOT + "/" + Option.FIN;
		
		String[] strs = Option.FIN.split("\\.");
		if ( strs.length > 0 && !strs[0].equals("") ) {
			String name  = strs[0]; 
			String head  = Option.ROOT + "/" + name + "_" + AUtils.FSUFFIX_HEADER;
			String pcset = Option.ROOT + "/" + name + "_" + AUtils.FSUFFIX_PCSET;
			// String pcset = Option.ROOT + "/" + name + "_MMHC_" + AUtils.FSUFFIX_PCSET;
			
			clbnsl(name, fin, head, pcset);
			
		} else {
			System.err.println("File Configuration Error!\n-->" + Option.FIN);
			return;
		}	
	}
	
	
	private static void clbnsl(String name, String fin, String fhead, String fpcsets) throws Exception {
		
		Instances insts = AUtils.loadInsts(fin);
		
		// write the head of ARFF into file
		if ( !new File(fhead).exists() ) {
			System.out.println("\nCreating the head of ARFF file...");
			AUtils.writeArffHeader(insts, fhead);
		} else {
			System.out.println("\nReading head from: " + fhead);
		}
		
		int [] listOfCL 		= null;
		int [][] candidateSet	= null;
		double[][] mutualInfo	= null;
		Vector<Vector<Integer>> pcsets = new Vector<Vector<Integer>>();
		
		Curricula curricula = new Curricula(false, insts, Option.N_CANDIDATE);
		
		// curriculum list
     	listOfCL = curricula.getCList(0);
     	
     	// discarded
     	// candidateSet = curricula.getCandidateSet(0);
		
		// generate PC (parents and children) set
     	if ( new File(fpcsets).exists() ) {
     		System.out.println("\nReading parents and children set from: " + fpcsets);
     		getPCSets(pcsets, fpcsets, insts.numAttributes());
     	} else {
     		System.out.println("\nCreating parents and children set...");
			mutualInfo = curricula.getMutualInfo();
			MMPC mmpc = new MMPC(insts, mutualInfo);
			pcsets = mmpc.getPCsets(fpcsets);
     	}

		// initialize sercher
		LocalScoreSearchAlgorithm searcher = new CLSearcher(
				Option.ROOT,
				insts,
				Option.INIT_STEP,
				Option.N_FIRSTADDED,
				candidateSet,
				Option.B_DEBUG,
				Option.B_RANDOM,
				Option.B_PARTITION,
				Option.B_CANDIDATE,
				Option.B_LIMITADD);
		
		// use penalty or not
		searcher.setHowToMakeScore(AUtils.SCORE_DEFAULT);
		searcher.setPCSets(pcsets);
		
		// Bayesian Network
		BayesNet bayesNet = new BayesNet();
		bayesNet.setSearchAlgorithm(searcher);
		
		// learning using different steps
		int optimalStep	 = -1;
		String learntNet = null;
		FileWriter outfile = null;
		double BDeuScore = -Double.MAX_VALUE, buildTime = 0;
		
		// timer
		long startTime = System.currentTimeMillis();
		
		for ( int i = 0; i < Option.N_STEP; i++ ) {
			int stepLength = i;
			
			// update step length and CL list
			bayesNet.updateSearchAlgorithm(stepLength, listOfCL);
			bayesNet.buildClassifier(insts);
			
			// save learned network
			learntNet = Option.ROOT + "/" + name + "_step_" + String.valueOf(stepLength) +
						"_" + Option.ID_RUN + "_" + AUtils.FSUFFIX_LN;
			outfile = new FileWriter(learntNet);
			outfile.write(bayesNet.toString());
			outfile.close();
			
			// @debug
			System.out.println("step " + i + " over..");
			
			// evaluation, we should use cross validation or some other similar 
			// model selection methods, here we just show the algorithm flow.
			CLBayesNetEstimator bne = new CLBayesNetEstimator();
			bne.updateLearnednet(fhead, learntNet, AUtils.STRUCT_FROM_ALIST);
			bne.evaluate(insts);
			
			// it would be better if we use Bdeu, BIC and LL to vote. In other words, 
			// only when at least two of the three metrics are better than the old 
			// ones, will we think the current structure is better and replace the 
			// old with it. That is just an alternative, implement it if you would 
			// like to see the differences.
			if ( BDeuScore < bne.getBDeuScore() ) {
				optimalStep = stepLength; 
				BDeuScore   = bne.getBDeuScore();
			}	
		}
		
		long endTime = System.currentTimeMillis();
		buildTime = (double)(endTime - startTime) / 1000;
		
		// write results into file
		String fresult = Option.ROOT + "/" + name + "_" + Option.ID_RUN + "_" + AUtils.FSUFFIX_ST;
		outfile = new FileWriter(fresult);
		outfile.write(BDeuScore + " & " + buildTime + " @ " + optimalStep + "\n");
		outfile.close();
	}
	
	
	private static void getPCSets(Vector<Vector<Integer>> pcsets, String fpcsets, int nLine) {
    	try {
    		pcsets.clear(); // ensure
			BufferedReader reader = new BufferedReader(new FileReader(new File(fpcsets)));
			
			String line = "";
			String[] tmp = null;
			for ( int i = 0; i < nLine; i++ ) {
				Vector<Integer> pcset = new Vector<Integer>();
				line = reader.readLine();
				tmp = line.split("\t");
				for ( int j = 0; j < tmp.length; j++ ) {
					if ( !tmp[j].trim().equals("") ) {
						pcset.add(new Integer(tmp[j].trim()));
					}
				}
				pcsets.add(pcset);
			}
			reader.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.out.println("PCSetFile: " + fpcsets);
			e.printStackTrace();
		}
    }
}
