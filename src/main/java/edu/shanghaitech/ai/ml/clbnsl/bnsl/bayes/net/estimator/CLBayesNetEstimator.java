package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.estimator;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.BayesNetNode;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.converters.ArffLoader;

/**
 * Integrated estimations including SHD, Distribution, and Score based metrics.
 * 
 * @author Yanpeng Zhao
 * 5/5/2015
 */
public class CLBayesNetEstimator {
	private BayesNetEstimator gtEstimator;
	private BayesNetEstimator lnEstimator;
	private BayesNetEstimator estimator;

	private double ess = AUtils.ESS; 
	
	/**
	 * Use this constructor to estimate the distributions without caring about SHD.
	 * 
	 */
	public CLBayesNetEstimator() {}
	
	
	/**
	 * Must initialize the ground-truth first when estimating SHD. 
	 * 
	 * @param headerPath 
	 * 		file of descriptions about the nodes
	 * @param netPath    
	 * 		skeleton of the net, adjacent list or matrix
	 * @param ess        
	 * 		equivalent sample size, 1 default
	 * @param isGT       
	 * 		see info above
	 * @param sourceid   
	 * 		type of the net
	 * 
	 */
	public CLBayesNetEstimator(
			String headerPath, 
			String netPath,
			double ess,
			boolean isGT, 
			int sourceid) throws Exception {
		if ( isGT ) {
			this.gtEstimator = new BayesNetEstimator(headerPath, netPath, ess, sourceid);
//			this.nNode = gtEstimator.getNNode();
			this.estimator = gtEstimator;
		} else { // default
			this.lnEstimator = new BayesNetEstimator(headerPath, netPath, ess, sourceid);
//			this.nNode = lnEstimator.getNNode();
			this.estimator = lnEstimator;
		}
		this.ess = ess;
	}
	
	
	/**
	 * @param headerPath 
	 * 		descriptions about nodes
	 * @param lnPath     
	 * 		learned net
	 * @param isAlist    
	 * 		type of the net
	 * @param sourceid   
	 * 		type of the net
	 * 
	 */
	public void updateLearnednet(String headerPath, String lnPath, int sourceid) throws Exception {
		this.lnEstimator = new BayesNetEstimator(headerPath, lnPath, ess, sourceid);
//		this.nNode = lnEstimator.getNNode();
		this.estimator = lnEstimator;
	}
	
	
	public void evaluate(Instances instances) {	
		estimator.evaluate(instances);
	}
	
	
	public void estimateParameters(Instances instances) {
		estimator.estimateParameters(instances);
	}
	
	
	public void predict(Instances instances) {
		estimator.predict(instances);
	}
	
	
	public void query(Instances instances) {
		estimator.query(instances);
	}
	
	
	/**
	 * Did not consider directions of the arcs.
	 * 
	 * @return 
	 * 		number of missing arc, say, exists in GroundTruth, not in learned net
	 * 
	 */
	public int getNMissingArc() {
		int nMissing = 0;
		for ( int i = 0; i < gtEstimator.getNEdge(); i++ ) {
			if ( !lnEstimator.isContained(gtEstimator.BNEdges[i]) ) {
				nMissing++;
			}
		}
		return nMissing;
	}

	
	/**
	 * Did not consider directions of the arcs.
	 * 
	 * @return 
	 * 		number of extra arc, compared to GroundTruth
	 * 
	 */
	public int getNExtraArc() {
		int nExtra = 0;
		for ( int i = 0; i < lnEstimator.nEdge; i++ ) {
			if ( !gtEstimator.isContained(lnEstimator.BNEdges[i]) ) {
				nExtra++;
			}
		}
		return nExtra;
	}
	
	
	/**
	 * @param isDAG 
	 * 		dag (true) or cpdag (false)
	 * @return 
	 * 		number of reversed arcs
	 * 
	 */
	public int getNReversedArc(boolean isDAG) {
		int nReverse = 0;
		for ( int i = 0; i < lnEstimator.nEdge; i++ ) {
			if ( gtEstimator.isReversed(lnEstimator.BNEdges[i], isDAG) ) {
				nReverse++;
			}
		}
		return nReverse;
	}
	
	
	/**
	 * Get the Markov blanket of the <tt>iNode</tt>.
	 * 
	 * @param  iNode 
	 * 		of which the Markov blanket will be returned.
	 * @return 
	 * 		Markov blanket of <tt>iNode</tt>
	 * 
	 */
	protected ArrayList<ArrayList<Integer>> getMarkovBlanket(int iNode) {
		ArrayList<ArrayList<Integer>> mb = new ArrayList<ArrayList<Integer>>();
		BayesNetNode[] nodes = estimator.getBNNodes();
		int nNode = estimator.getNNode();
		
		ArrayList<Integer> list = nodes[iNode].getParInfo();
		mb.add(list);

		for ( int i = 0; i < nNode; i++ ) {
			if ( i != iNode && nodes[i].hasParent(iNode) ) {
				list = nodes[i].getParInfo();
				mb.add(list);
			}
		}
		return mb;
	}
	
	
	protected void writeMBOfClass(String mbFile) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(mbFile));
			int iNode = estimator.getNNode() - 1;
			ArrayList<ArrayList<Integer>> mb = getMarkovBlanket(iNode);
			StringBuffer sb = new StringBuffer();
			
			for ( int i = 0; i < mb.size(); i++ ) {
				ArrayList<Integer> list = mb.get(i);
				for ( Integer item : list ) {
					sb.append(item + " ");
				}
				sb.append("\n");
			}
			writer.write(sb.toString());
			writer.close();
			System.out.println("Markov blanket: " + mbFile);
		} catch (Exception e) {
			System.out.println("Read or write file error: " + mbFile);
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Generate the naive Bayesian Network skeleton.
	 * 
	 * @param naiveBNPath 
	 * 		storing naive Bayesian network
	 * 
	 */
	protected void addClassAttr2NaiveBN(String naiveBNPath) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(naiveBNPath));
			StringBuffer sb = new StringBuffer();
			int nNode = estimator.getNNode();
			
			sb.append((nNode + 1) + "\n");
			for ( int i = 0; i < nNode; i++ ) {
				sb.append(i + " " + 1 + " " + nNode + "\n");
			}
			sb.append(nNode + " " + 0 + "\n");
			writer.write(sb.toString());
			writer.close();
			System.out.println("Naive net: " + naiveBNPath);
		} catch (Exception e) {
			System.out.println("Read or write file error: " + naiveBNPath);
			e.printStackTrace();
		}
	}
	
	
	/**
	 * After learning a Bayesian network among the attributes set without class attribute contained, we try 
	 * to build a Bayesian network classifier in which every non-class attribute gets an additional parent, 
	 * say, the class attribute. That is, there is an directed edge emanating from the class attribute to 
	 * each of the non-class attributes. 
	 * 
	 * @param netPath    
	 * 		bayesian network structure without class attribute contained
	 * @param newNetPath 
	 * 		new Bayesian network structure with class attribute added into Bayesian network structure
	 * 
	 */
	protected void addClassAttr2Net(String newNetPath) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(newNetPath));
			BayesNetNode[] nodes = estimator.getBNNodes();
			StringBuffer sb = new StringBuffer();
			int nNode = estimator.getNNode();
			int nNewNode = nNode + 1;
			
			sb.append(nNewNode + "\n");
			for ( int i = 0; i < nNode; i++ ) {
				int[] parentSet = nodes[i].getParSet();
				sb.append(i + " " + (parentSet.length + 1) + " ");
				for (int iParent = 0; iParent < parentSet.length; iParent++) {
		        	int iNode = parentSet[iParent];
		        	sb.append(iNode + " ");
		        }
				sb.append(nNode + "\n");
			}
			sb.append(nNode + " " + 0 + "\n");
			writer.write(sb.toString());
			writer.close();
			System.out.println("New net: " + newNetPath);
		} catch (Exception e) {
			System.out.println("Read or write file error: " + newNetPath);
			e.printStackTrace();
		}
	}
	
	
	/**
	 * For MMHC, this procedure just makes an adjacent-list-presentation of bayesian network structure from the
	 * adjacent-matrix-presentation.<p>
	 * 
	 * For CL, this procedure just make a copy of the adjacent-list-presentation of bayesian network structure.<p>
	 * 
	 * @param newNetPath 
	 * 		new Bayesian network structure with class attribute added into Bayesian network structure
	 * 
	 */
	protected void makeCopyOfBN(String newNetPath) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(newNetPath));
			BayesNetNode[] nodes = estimator.getBNNodes();
			StringBuffer sb = new StringBuffer();
			int nNode = estimator.getNNode();
			
			sb.append(nNode + "\n");
			for ( int i = 0; i < nNode; i++ ) {
				int[] parentSet = nodes[i].getParSet();
				sb.append(i + " " + (parentSet.length) + " ");
				for (int iParent = 0; iParent < parentSet.length; iParent++) {
		        	int iNode = parentSet[iParent];
		        	sb.append(iNode + " ");
		        }
				sb.append("\n");
			}
			writer.write(sb.toString());
			writer.close();
			System.out.println("New net: " + newNetPath);
		} catch (Exception e) {
			System.out.println("Read or write file error: " + newNetPath);
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Generate naive bayesian network skeleton.
	 * 
	 * @param naiveBNPath 
	 * 		storing naive bayesian network
	 * 
	 */
	protected void generateNaiveBN(String naiveBNPath) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(naiveBNPath));
			StringBuffer sb = new StringBuffer();
			int nNode = estimator.getNNode();
			
			sb.append((nNode) + "\n");
			for ( int i = 0; i < nNode - 1; i++ ) {
				sb.append(i + " " + 1 + " " + (nNode - 1) + "\n");
			}
			sb.append((nNode - 1) + " " + 0 + "\n");
			writer.write(sb.toString());
			writer.close();
			System.out.println("Naive net: " + naiveBNPath);
		} catch (Exception e) {
			System.out.println("Read or write file error: " + naiveBNPath);
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Write log likelihood.
	 * 
	 * @param llFile
	 * 
	 */
	protected void toSummaryQuery(String llFile) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(llFile));
			double llscore = estimator.getLLOfInsts();

			String formatScore = String.format("%.3f", llscore) + "\n";
			writer.write(formatScore);
			writer.close();
			System.out.println("Log likelihood: " + formatScore);
		} catch (NullPointerException e) {
			System.err.print("Log likelihood might not be initialized yet.");
		} catch (Exception e) {
			System.err.print("File write error: " + llFile);
			e.printStackTrace();
		}
	}

	/**
	 * Write confusion matrix to the file for the further analysis.
	 * 
	 */
	protected void toSummary(String confusionMatrixFile) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(confusionMatrixFile));
			int[][] confusionMatrix = estimator.getConfusionMatrix();
			StringBuffer sb = new StringBuffer();
			int tp = 0, n = 0;
			
			int nClass = confusionMatrix[0].length;
			for ( int i = 0; i < nClass; i++ ) {
				for ( int j = 0; j < nClass; j++ ) {
					if ( i == j ) { tp += confusionMatrix[i][j]; }
					n += confusionMatrix[i][j];
					sb.append(confusionMatrix[i][j] + "\t");
				}
				sb.append("\n");
			}
			String precision = String.format("%.3f", (double) tp / n) + "\n";
			sb.insert(0, precision);
			writer.write(sb.toString());
			writer.close();
			System.out.println("Classification precision: " + precision);
			System.out.println("Confusion matrix: " + confusionMatrixFile + "\n\n");
		} catch (NullPointerException e) {
			System.err.print("Confusion Matrix might not be initialized yet.");
		} catch (Exception e) {
			System.err.print("File write error: " + confusionMatrixFile);
			e.printStackTrace();
		}
	}
	
	
	public int getNNode() {
		return estimator.getNNode();
	}
	
	
	public int getNEdge() {
		return estimator.getNEdge();
	}
	
	
	public int getMaxIndegree() {
		return estimator.getMaxIndegree();
	}
	
	
	public int getMaxOutdegree() {
		return estimator.getMaxOutdegree();
	}
	
	
	public double getBDeuScore() {
		return estimator.getBDeuScore();
	}
	
	
	public double getBICScore() {
		return estimator.getBICScore();
	}
	
	
	public double getLLScore() {
		return estimator.getLLScore();
	}
	
	
	public double getKLScore() {
		return estimator.getKLScore();
	}
	
	
	protected  void printCPDAG() {
		estimator.printCPDAG();
	}
	
	
	protected static Instances getTestSet(String testSetPath) {
		try {
			ArffLoader arffLoader = new ArffLoader();
			File file = new File(testSetPath);
			arffLoader.setFile(file);
			return arffLoader.getDataSet();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}
	

	/**
	 * This is for the condition when the procures terminate due to certain exceptions, 
	 * it allows you to get estimations without restarting and reproducing results.
	 * 
	 * @param iStep       
	 * 		the starting step
	 * @param nStep       
	 * 		number of the steps
	 * @param testSetPath 
	 * 		test set
	 * @param basePath    
	 * 		root
	 * @param path        
	 * 		relative sub-path
	 * @param nameOfDS    
	 * 		name of data set
	 * @param IA          
	 * 		id of the algorithm
	 * 
	 */
	protected static void evaluation(
			int iStep,
    		int nStep, 
    		String testSetPath,
    		String basePath,
    		String path,
    		String nameOfDS, 
    		String IA) throws Exception {
		Instances testSet = getTestSet(testSetPath);
		
    	String pathOfGT  = basePath + "res/" + nameOfDS + "/" + 
    			nameOfDS + "_" + AUtils.FSUFFIX_GT;
    	String pathOfHD  = basePath + "net/" + nameOfDS + "/" +
    			nameOfDS + "_" + AUtils.FSUFFIX_HEADER;
    	String pathOfSHD = path + nameOfDS + IA + AUtils.FSUFFIX_SHD;
    	String pathOfEva = path + nameOfDS + IA + AUtils.FSUFFIX_EVA;
    	
    	// initialize ground truth
    	CLBayesNetEstimator ec = new CLBayesNetEstimator(
    			pathOfHD, pathOfGT, 1.0, true, AUtils.STRUCT_FROM_ALIST); 
    	
    	FileWriter outfile = new FileWriter(pathOfSHD);
    	FileWriter evaluation = new FileWriter(pathOfEva);
    	
		ec.evaluate(testSet); // evaluate ground truth
		evaluation.write(ec.getBDeuScore() + " " + ec.getBICScore() + " " + 
				ec.getLLScore() + " " + ec.getKLScore() + "\n");
    	
    	for ( int i = iStep; i < nStep; i++ ) {
    		int stepLength = i;
    		String learntNet = path + nameOfDS + "_step_" + 
	     			String.valueOf(stepLength) + IA + AUtils.FSUFFIX_LN;
    		// initialize learned net
    		ec.updateLearnednet(pathOfHD, learntNet, AUtils.STRUCT_FROM_ALIST); 
    		ec.evaluate(testSet); // evaluate learned net
    		
        	String shd = "M: " + ec.getNMissingArc() + " " +
        				"E: " + ec.getNExtraArc() + " " +
        				"R: " + ec.getNReversedArc(true) + " " +
        				"RC: " + ec.getNReversedArc(false) + "\n";
        	String score = ec.getBDeuScore() + " " + ec.getBICScore() + " " + 
        				ec.getLLScore() + " " + ec.getKLScore() + "\n";

        	outfile.write(shd);
        	evaluation.write(score);
    	}
    	outfile.close();
    	evaluation.close();
    }
	
	
	protected static void evaSHD(
			int iStep,
    		int nStep, 
    		String basePath,
    		String path,
    		String nameOfDS, 
    		String IA,
    		Instances testSet) throws Exception {
		String gtHeader = basePath + "res/" + nameOfDS + "/" + nameOfDS + "_" + AUtils.FSUFFIX_HEADER;
		String gtStruct = basePath + "res/" + nameOfDS + "/" + nameOfDS + "_" + AUtils.FSUFFIX_GT;
    	
    	// initialize ground truth
    	CLBayesNetEstimator ec = new CLBayesNetEstimator(
    			gtHeader, gtStruct, AUtils.ESS, true, AUtils.STRUCT_FROM_ALIST); 
    	
    	for ( int i = iStep; i < nStep; i++ ) {
    		int stepLength = i;
    		String learntNet = path + nameOfDS + "_step_" + String.valueOf(stepLength) + IA + AUtils.FSUFFIX_LN;
    		// initialize learned net
    		ec.updateLearnednet(gtHeader, learntNet, AUtils.STRUCT_FROM_ALIST); 
    		ec.evaluate(testSet);
    	}
    }
	
	
	protected static void getEvaSHD() throws Exception {
		int iStep = 1;
		int nStep = 2;
		int seed = 0;
		String scale = "500";
		String nameOfDS = "barley";
		String basePath = "E:/NewData/";

		String path = basePath + "res/" + nameOfDS + "/seed" + String.valueOf(seed) + "/" + scale + "/";
		String testSetPath = basePath + "res/" + nameOfDS + "/seed" + String.valueOf(9) + "/" + String.valueOf(5000) + 
				"/" + nameOfDS + String.valueOf(5000) + AUtils.FSUFFIX_ARFF;
		String IA = "_CL0_" + String.valueOf((int) AUtils.ESS_CL) + "_";
		
		ArffLoader arffLoader = new ArffLoader();
		File file = new File(testSetPath);
		arffLoader.setFile(file);
		Instances testSet =  arffLoader.getDataSet();
		evaSHD(iStep, nStep, basePath, path, nameOfDS, IA, testSet);
	}
	
	
	protected static void getSHDAtEachStage() throws Exception {
		int nStage = 11;
		int seed = 3;
		int step = 1;
		
		String scale = "5000";
		String nameOfDS = "hailfinder";
		String basePath = "E:/NewData/";
		
		String gtHeader = basePath + "res/" + nameOfDS + "/" + nameOfDS + "_" + AUtils.FSUFFIX_HEADER;
		String gtStruct = basePath + "res/" + nameOfDS + "/" + nameOfDS + "_" + AUtils.FSUFFIX_GT;

		String path = basePath + "res/" + nameOfDS + "/seed" + String.valueOf(seed) + "/" + scale + "/";
		
		// initialize ground truth
    	CLBayesNetEstimator ec = new CLBayesNetEstimator(
    			gtHeader, gtStruct, AUtils.ESS, true, AUtils.STRUCT_FROM_ALIST); 
    	
    	int M = 0, E = 0, R = 0, RC = 0;
    	StringBuffer sb = new StringBuffer();
    	for ( int i = 0; i < nStage; i++ ) {
    		int stage = i;
    		String learntNet = path + nameOfDS + "_step_" + String.valueOf(step) + "_" + String.valueOf(stage) + "_" + AUtils.FSUFFIX_FNDAG;
    		
    		// initialize learned net
    		ec.updateLearnednet(gtHeader, learntNet, AUtils.STRUCT_FROM_ALIST);
    		M = ec.getNMissingArc();
			E = ec.getNExtraArc();
			R = ec.getNReversedArc(true);
			RC = ec.getNReversedArc(false);	
    		sb.append(M + "\t" + E + "\t" + R + "\t" +RC + "\t" + (M + E + R + RC) + "\n");
    	}
    	System.out.println(sb.toString());
	}
	
	
	public static void main(String[] args) throws Exception {
//		getEvaSHD();
//		getSHDAtEachStage();
	}
}
