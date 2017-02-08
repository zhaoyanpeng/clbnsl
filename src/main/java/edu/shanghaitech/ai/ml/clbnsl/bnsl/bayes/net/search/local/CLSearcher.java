package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.search.local;

import java.io.FileWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.BayesNet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.ParentSet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instance;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * @author Yetian Chen
 * @author Yanpeng Zhao
 * 3/25/2015
 */
@SuppressWarnings("unused")
public class CLSearcher extends LocalScoreSearchAlgorithm {
	
	protected Cache cache = null;
	
	private int nFirstAdded = 3;
	protected int stepLength = 0;
	protected int nNode = 0;
	
	// curriculum
	protected int[] listOfCL;
	private int[][] candidateSet;
	private int nCandidate = 0;
	
	// save details
	private String basePath = null;
	private FileWriter writer = null;
	
	private boolean bDebug = false;
	private boolean bUseRandom = false;
	private boolean bUseArcReverse = true;
	private boolean bUsePartition = false;
	protected boolean bUseCandidate = false;
	private boolean bUseLimitadd = true;
	
	// parameters for self-adapted step
	protected int end = 0;
	private Vector<Vector<Integer>> pcSets;
	
	private int cacheTime = 0;
	private int searchTime = 0;
	private boolean printCacheTime = false;
	private boolean printSearchTime = false;
	
	private long startTime = 0; 
	private long endTime = 0; 
	
	public CLSearcher() {}

	
	public CLSearcher(
			String basePath,
			Instances instances, 
			int stepLength, 
			int nFirstAdded,
			int[][] candidateSet,
			boolean bDebug,
			boolean bUseRandom,
			boolean bUsePartition,
			boolean bUseCandidate,
			boolean bUseLimitadd) {
		this.basePath = basePath;
		this.stepLength = stepLength;
		this.nFirstAdded = nFirstAdded;
		this.nNode = instances.numAttributes();
		this.bDebug = bDebug;
		this.bUseRandom = bUseRandom;
		this.bUsePartition = bUsePartition;
		this.listOfCL = new int[nNode];
		
		this.bUseCandidate = bUseCandidate;
		this.candidateSet = candidateSet;
		
		this.bUseLimitadd = bUseLimitadd;
		
		if ( candidateSet != null ) {
			this.nCandidate = candidateSet[0].length;
		}
		this.indexPartitions = new ArrayList<ArrayList<Integer>>();
	}
	
	
	public void initCacheStatistic() {
		this.cacheTime = 0;
		this.printCacheTime = true;
	}
	
	
	public void initSearchStatistic() {
		this.searchTime = 0;
		this.printSearchTime = true;
	}
	
	
	public void searchStatistic() {
		this.searchTime++;
		if ( searchTime >= 1e5 && printSearchTime ) {
			this.printSearchTime = false;
			System.out.println("I've searched over " + searchTime + " times.");
		}
	}
	
	
	public void cacheStatistic() {
		this.cacheTime++;
		if ( cacheTime >= 1e5 && printCacheTime ) {
			this.printCacheTime = false;
			System.out.println("I've cached over " + cacheTime + " times.");
		}
	}
	
	
	public void timeStatistic(String description) {
		System.out.println((this.endTime - this.startTime) + "ms.");
		if ( (this.endTime - this.startTime) > 500 ) {
			System.out.println(description + " consume more than 500ms.");
		}
	}
	
	
	protected void pintParentSet() {
		
		for ( int i = 0; i < end; i++ ) {
			int iNode = listOfCL[i];
			ParentSet parentSet = m_BayesNet.getParentSet(iNode);
			System.out.print("iNode: " + iNode + " | ");
			for ( int j = 0; j < parentSet.getNrOfParents(); j++ ) {
				System.out.print(parentSet.getParent(j) + "\t");
			}
			System.out.println();
		}
	}
	
  	/**
  	 * If you need to use this function to record the intermediate networks learned by our algorithm,
  	 * just rename it to <tt>search</tt>, and comment the existing <tt>search</tt> function. Could we
  	 * make it easier? Yes, but lazy Author didn't declare it in the parent class {@code SearchAlgorithm}.
  	 * 
  	 */
	protected void searchbiubiubiu(BayesNet bayesNet, Instances instances) throws Exception {
		
		if ( stepLength >= 0 ) {
			
			int stage = 0, count = 0;
			String detailPath = null;
			boolean needPartition = true;
			instances.setNumPattern(0);
			
			if ( stepLength > 0 ) {			
				
				if ( nNode < stepLength + nFirstAdded || nNode < nFirstAdded || nNode < 1 ) {
					System.out.println("nNode: " + nNode + "; " + "stepLength: " + stepLength + "; " + "nFirstAdded: " + nFirstAdded);
					throw new IllegalArgumentException("the step size is too large for the problem size or the problem is too small.");
				}

				// curriculum learning except the final step
				this.end = nFirstAdded + stage * stepLength;
				int lastNPartition = instances.numInstances();
				while ( this.end < nNode ) {
					
					if ( needPartition ) { indexInstances(instances, false); }

					// @debug
					System.out.println("->Current stage: " + stage);
					System.out.print("end: " + end + "\t" + lastNPartition + "\t" + instances.getNumPattern() + "\t");
					
					this.stage = stage;
					stage++;
					
					// data partition at the current stage little differs that at the last stage 
					if ( Math.abs(instances.getNumPattern() - lastNPartition) < (int) Math.ceil(0.02 * lastNPartition) ) {
						System.out.println("skip");
						
						this.indexPartitions.clear();
						if ( !needPartition ) { stage--; }
						this.end = nFirstAdded + stage * stepLength; // important
						needPartition = true;
						
						continue;
					} else { System.out.println(); }
					
					lastNPartition = instances.getNumPattern();
					needPartition = false;
					
					initCacheStatistic();
					initCache(bayesNet, instances);

					// do search
					initSearchStatistic(); // @debug
					Operation operation = getOptimalOperation(bayesNet, instances);
					while ( operation != null && operation.fDeltaScore > 0 ) {
						
						startTime =System.currentTimeMillis(); 
						
						performOperation(bayesNet, instances, operation);
						
						endTime =System.currentTimeMillis(); 
						
						initSearchStatistic(); // @debug
						
						operation = getOptimalOperation(bayesNet, instances);

						System.out.print(String.format("%.6f", operation.fDeltaScore) + "\tsearchTime: " + searchTime + "\tcacheTime: " + cacheTime + "\t");
						
						this.timeStatistic("Perform Operation");
					}
					
					// write details at each learning stage this.m_BayesNet.toString();
					detailPath = basePath + "_step_" + String.valueOf(stepLength) + "_" + String.valueOf(count) + "_" + AUtils.FSUFFIX_FNDAG;
					writer = new FileWriter(detailPath);
					writer.write(m_BayesNet.toString(listOfCL, end, fullConnectedDAG));
					writer.close();
					count++;
				}
				
				// final stage or no curricula
				this.end = this.nNode;
				instances.setNumPattern(0);
				initCache(bayesNet, instances);
				
				// do search
				Operation operation = getOptimalOperation(bayesNet, instances);
				while ( operation != null && operation.fDeltaScore > 0 ) {
					performOperation(bayesNet, instances, operation);
					operation = getOptimalOperation(bayesNet, instances);
				}
			} else {
				// no curricula
				for ( int i = 0; i < nNode; i++ ) {
					listOfCL[i] = i;
				}
				this.end = this.nNode;
				initCache(bayesNet, instances);
	
				// do search
				Operation operation = getOptimalOperation(bayesNet, instances);
				while ( operation != null && operation.fDeltaScore > 0 ) {
					performOperation(bayesNet, instances, operation);
					operation = getOptimalOperation(bayesNet, instances);
				}
			}
		}
	}
	
	
	protected void search(BayesNet bayesNet, Instances instances) throws Exception {
		
		if ( stepLength >= 0 ) {
			
			int stage = 0;
			String detailPath = null;
			boolean needPartition = true;
			instances.setNumPattern(0);
			
			if ( stepLength > 0 ) {			
				
				if ( nNode < stepLength + nFirstAdded || nNode < nFirstAdded || nNode < 1 ) {
					System.out.println("nNode: " + nNode + "; " + "stepLength: " + stepLength + "; " + "nFirstAdded: " + nFirstAdded);
					throw new IllegalArgumentException("the step size is too large for the problem size or the problem is too small.");
				}

				// curriculum learning except the final step
				this.end = nFirstAdded + stage * stepLength;
				int lastNPartition = instances.numInstances();
				while ( this.end < nNode ) {
					
					// it's the only difference between algorithm 0 & 1
					// comment it to get algorithm1, or get algorithm 0
					if ( bUsePartition ) { // use algorithm 0
						if ( bUseRandom ) {
							if ( Math.random() < 0.5 ) { 
								indexInstances(instances, false);
							} else {
								// no partition
								instances.setNumPattern(0);
							}
						} else {
							if ( needPartition ) {
								indexInstances(instances, false);
							}
						}
					}

					// @debug
					System.out.println("->Current stage: " + stage);
					System.out.print("end: " + end + "\t" + lastNPartition + "\t" + instances.getNumPattern() + "\t");
					
					this.stage = stage;
					stage++;
					
					// data partition at the current stage little differs that at the last stage 
					if ( Math.abs(instances.getNumPattern() - lastNPartition) < (int) Math.ceil(0.02 * lastNPartition) ) {
						System.out.println("skip");
						
						this.indexPartitions.clear();
						this.end = nFirstAdded + stage * stepLength; // important
						needPartition = true;
						
						continue;
					} else { System.out.println(); }
					
					lastNPartition = instances.getNumPattern();
					needPartition = false;
					
					initCacheStatistic();
					initCache(bayesNet, instances);

					// do search
					initSearchStatistic(); // @debug
					Operation operation = getOptimalOperation(bayesNet, instances);
					while ( operation != null && operation.fDeltaScore > 0 ) {
						
						startTime =System.currentTimeMillis(); 
						
						performOperation(bayesNet, instances, operation);
						
						endTime =System.currentTimeMillis(); 
						
						initSearchStatistic(); // @debug
						
						operation = getOptimalOperation(bayesNet, instances);

						System.out.print(String.format("%.6f", operation.fDeltaScore) + "\tsearchTime: " + searchTime + "\tcacheTime: " + cacheTime + "\t");
						
						this.timeStatistic("Perform Operation");
					}
				}
				
				// final stage or no curricula
				this.end = this.nNode;
				instances.setNumPattern(0);
				initCache(bayesNet, instances);
				
				// do search
				Operation operation = getOptimalOperation(bayesNet, instances);
				while ( operation != null && operation.fDeltaScore > 0 ) {
					performOperation(bayesNet, instances, operation);
					operation = getOptimalOperation(bayesNet, instances);
				}
			} else {
				// no curricula
				for ( int i = 0; i < nNode; i++ ) {
					listOfCL[i] = i;
				}
				this.end = this.nNode;
				initCache(bayesNet, instances);
	
				// do search
				Operation operation = getOptimalOperation(bayesNet, instances);
				while ( operation != null && operation.fDeltaScore > 0 ) {
					performOperation(bayesNet, instances, operation);
					operation = getOptimalOperation(bayesNet, instances);
				}
			}
		}
	}
	
	
	public void resetPartition(Instances instances) {
		if ( instances.getNumPattern() <= 2 ||
			 instances.getNumPattern() > (instances.size() / 2) ) {
			instances.setNumPattern(0);
			System.out.println("No Partition.");
		}
	}

	
	public Operation getOptimalOperation(BayesNet bayesNet, Instances instances) 
			throws Exception {
		
		Operation bestOperation = new Operation();
		bestOperation = findBestArcToAdd(bayesNet, instances, bestOperation); // add
		bestOperation = findBestArcToDelete(bayesNet, instances, bestOperation); // delete
		if ( getUseArcReversal() ) { // reverse
			bestOperation = findBestArcToReverse(bayesNet, instances, bestOperation);
		}
		// double check
		if ( bestOperation.fDeltaScore == -1E100 ) { return null; }
		return bestOperation;
	}
	
	
	public Operation findBestArcToAdd(BayesNet bayesNet, 
			Instances instances, 
			Operation bestOperation) {
		
		boolean canadd = true;
		for ( int i = 0; i < end; i++ ) {
			
			int iHead = listOfCL[i];
			if ( bUseLimitadd ) {
				canadd = bayesNet.getParentSet(iHead).getNrOfParents() <= m_nMaxNrOfParents;
				// System.err.println("Use limited number of parents."); // @debug
			}
			if ( canadd ) {						
				for ( int j = 0; j < end; j++ ) {
					int iTail = listOfCL[j];
					if ( shallBeArc(iHead, iTail, bUseCandidate) && addArcMakesSense(bayesNet, instances, iHead, iTail) ) {
						
						Operation operation = new Operation(iTail, iHead, Operation.OPERATION_ADD);
						if ( cache.get(operation) > bestOperation.fDeltaScore && isNotTabu(operation) ) {
							bestOperation = operation;
							bestOperation.fDeltaScore = cache.get(operation);
						}	
					}	
					this.searchStatistic(); // @debug
				}
			} else {
				// randomly delete a parent and add the new one.
			}
		}
		return bestOperation;
	}
	
	
	public boolean shallBeArc(int head, int tail, boolean useCandidate) {
		// System.err.println("Use candidate parents."); // @debug
		if ( !useCandidate ) { return true; }
		Vector<Integer> pcSet = pcSets.get(head);
		if ( pcSet.contains(new Integer(tail)) ) {
			return true;
		}
		return false;
	}
	
	
	public Operation findBestArcToDelete(BayesNet bayesNet,
			Instances instances,
			Operation bestOperation) {
		
		for ( int i = 0; i < end; i++ ) {
			
			int iHead = listOfCL[i];
			ParentSet parentSet = bayesNet.getParentSet(iHead);
			for ( int j = 0; j < parentSet.getNrOfParents(); j++ ) {
				
				Operation operation = new Operation(parentSet.getParent(j), iHead, Operation.OPERATION_DEL);
				if ( cache.get(operation) > bestOperation.fDeltaScore && isNotTabu(operation) ) {
					bestOperation = operation;
					bestOperation.fDeltaScore = cache.get(operation);
				}	
				this.searchStatistic(); // @debug
			}
		}
		return bestOperation;
	}
	
	
	public Operation findBestArcToReverse(BayesNet bayesNet,
			Instances instances,
			Operation bestOperation) {
		
		boolean canadd = true;
		for ( int i = 0; i < end; i++ ) {
			
			int iHead = listOfCL[i];
			ParentSet parentSet = bayesNet.getParentSet(iHead);
			for ( int j = 0; j < parentSet.getNrOfParents(); j++ ) {
				
				int iTail = parentSet.getParent(j);
				if ( bUseLimitadd ) {
					canadd = bayesNet.getParentSet(iTail).getNrOfParents() <= m_nMaxNrOfParents;
				}
				
				if ( canadd && shallBeArc(iTail, iHead, bUseCandidate) && reverseArcMakesSense(bayesNet, instances, iHead, iTail) ) {
					
					Operation operation = new Operation(parentSet.getParent(j), iHead, Operation.OPERATION_REVERSE);
					if ( cache.get(operation) > bestOperation.fDeltaScore && isNotTabu(operation) ) {
						bestOperation = operation;
						bestOperation.fDeltaScore = cache.get(operation);
					}	
				}	
				this.searchStatistic(); // @debug
			}
		}
		return bestOperation;
	}
	
	
	public void initCache(BayesNet bayesNet, Instances instances) {

		cache = Cache.getInstance(nNode);

		for ( int i = 0; i < end; i++ ) {
			int iHead = listOfCL[i];
			updateCache(iHead, bayesNet.getParentSet(iHead));
		}
	}
	
	
	public void updateCache(int iHead,  ParentSet parentSet) {
		
		this.method = "BaseScore";
		double fBaseScore = calcNodeScoreCurricula(iHead);
		
		int nNrOfParents = parentSet.getNrOfParents();
		for ( int i = 0; i < end; i++ ) {
			
			int iTail = listOfCL[i];
			if ( iTail != iHead ) {
				if ( !parentSet.contains(iTail) ) {
					if ( !parentSet.contains(iTail) && shallBeArc(iHead, iTail, bUseCandidate) ) {
						
						Operation operation = new Operation(iTail, iHead, Operation.OPERATION_ADD);
						
						if ( beWithExtraParent(iHead, iTail) ) {
							cache.put(operation, calcScoreWithExtraParentCurricula(iHead, iTail) - fBaseScore);
							this.cacheStatistic();
						} else {
							// directly set it to 0
							cache.put(operation, 0);
							this.cacheStatistic();
						}

					}
				} else {
					Operation operation = new Operation(iTail, iHead, Operation.OPERATION_DEL);

					cache.put(operation, calcScoreWithMissingParentCurricula(iHead, iTail) - fBaseScore);
					this.cacheStatistic();
				}
			}
		}
	}
	
	
	public int getIPattern(String pattern, ArrayList<String> patternList) {
		for ( int i = 0; i < patternList.size(); i++ ) {
			if ( pattern.equals(patternList.get(i)) ) {
				return i;
			}
		}
		return -1;
	}
	
	
	/**
     * Index the data for each stage of curriculum learning.
     * 
     */
	public void indexInstances(Instances instances, boolean newMethod){
    	
    	if( listOfCL == null ){
    		throw new NullPointerException("curriculum does not exist");
    	}
    	if( this.end >= instances.numAttributes() ) {
    		throw new IllegalArgumentException("Final step of Curriculum, No need to index data");
    	}	

		// construct the pattern list
		String pattern = "";
		Instance instance = null;
		int index = 0, count = 0, ipattern;
		
		// store all the patterns
    	ArrayList<String> patternList = new ArrayList<String>();

		Enumeration<Instance> enumInsts = instances.enumerateInstances();
		while ( enumInsts.hasMoreElements() ) {
			pattern = "";
			index = this.end;
			instance = (Instance) enumInsts.nextElement();
			while ( index < instance.numAttributes() ){
				pattern += instance.stringValue(listOfCL[index]);
				index++;
			}
			
			if ( (ipattern = getIPattern(pattern, patternList)) < 0 ) {
				ArrayList<Integer> indexPartition = new ArrayList<Integer>();
				
				indexPartition.add(new Integer(count));
				indexPartitions.add(indexPartition);
				
				patternList.add(pattern);
			} else {
				indexPartitions.get(ipattern).add(new Integer(count));
			}
			count++;
		}
		// set the number of patterns in instances
		instances.setNumPattern(patternList.size());
		
		patternList.clear();
    }
    
    
    public void updateCL(int stepLength, int[] listOfCL) {
		this.stepLength = stepLength;
		
		if ( listOfCL == null ) { return; } // when step size is 0
		// avoid shadow copy - junk reference in Java
		// since we first run the algorithm without curriculum, we need to
		// reinitialize listOfCl when we use curriculum.
		for ( int i = 0; i < nNode; i++ ) {
			this.listOfCL[i] = listOfCL[i];
		}
	}
    
    
	public void setPCSets(Vector<Vector<Integer>> pcSets) {
		this.pcSets = pcSets;
	}
	
	
	public void setFullConnectedDAG(ArrayList<ArrayList<Integer>> fcDAG) {
		this.fullConnectedDAG = fcDAG;
	}
    
    
    public void setStepSize(int stepSize) {
    	this.stepLength = stepSize;
    }
    
    
	boolean isNotTabu(Operation oOperation) {
		return true;
	} // overwrite
	
	
	/** 
	 * Apply an operation on the Bayes network and update the cache.
	 * Modified for Curriculum Learning
	 * 
	 * @param bayesNet 
	 * 		Bayesian network to apply operation on
	 * @param instances 
	 * 		data set to learn from
	 * @param oOperation 
	 * 		operation to perform
	 * @throws Exception 
	 * 		if something goes wrong
	 * 
	 */
	void performOperation(BayesNet bayesNet, Instances instances, Operation oOperation) throws Exception {
		// perform operation
		switch (oOperation.nOperation) {
			case Operation.OPERATION_ADD:
				applyArcAddition(bayesNet, oOperation.nHead, oOperation.nTail, instances);
				if ( this.bDebug ) {
					System.out.println("Add " + oOperation.nHead + " -> " + oOperation.nTail);
				}
				break;
			case Operation.OPERATION_DEL:
				applyArcDeletion(bayesNet, oOperation.nHead, oOperation.nTail, instances);
				if ( this.bDebug ) {
					System.out.println("Del " + oOperation.nHead + " -> " + oOperation.nTail);
				}
				break;
			case Operation.OPERATION_REVERSE:
				applyArcDeletion(bayesNet, oOperation.nHead, oOperation.nTail, instances);
				applyArcAddition(bayesNet, oOperation.nTail, oOperation.nHead, instances);
				if ( this.bDebug ) {
					System.out.println("Rev " + oOperation.nHead+ " -> " + oOperation.nTail);
				}
				break;
		}
	} // performOperation	
	
	
	void applyArcAddition(BayesNet bayesNet, int iHead, int iTail, Instances instances) {
		ParentSet bestParentSet = bayesNet.getParentSet(iHead);
		bestParentSet.addParent(iTail, instances);
		updateCache(iHead, bestParentSet);
	}
	
	
	void applyArcDeletion(BayesNet bayesNet, int iHead, int iTail, Instances instances) {
		ParentSet bestParentSet = bayesNet.getParentSet(iHead);
		bestParentSet.deleteParent(iTail, instances);
		updateCache(iHead, bestParentSet);
	}


	/** 
	 * @return 
	 * 		whether the arc reversal operation should be used
	 * 
	 */
	public boolean getUseArcReversal() {
		return bUseArcReverse;
	}
	

	/** set use the arc reversal operation
	 * @param bUseArcReversal whether the arc reversal operation should be used
	 */
	public void setUseArcReversal(boolean bUseArcReversal) {
		this.bUseArcReverse = bUseArcReversal;
	}

	
	/**
	 * @author Yanpeng Zhao
	 * 
	 */
	protected class Operation {
		
		// type of an operation
		final static int OPERATION_ADD = 0;
		final static int OPERATION_DEL = 1;
		final static int OPERATION_REVERSE = 2;
		
		// description of an arc
		public int nTail;
		public int nHead;
		public int nOperation;
		
		// change of score due to this operation
		public double fDeltaScore = -1E100;
		
		// constructor
		public Operation() { }
		
		// constructor for initialization
		public Operation(int nTail, int nHead, int nOperation) {
			this.nTail = nTail;
			this.nHead = nHead;
			this.nOperation = nOperation;
		}
		
		// compare this operation with another one
		public boolean equals(Operation other) {
			if ( other == null) { return false; }
			
			return ( (nOperation == other.nOperation) &&
					 (nHead == other.nHead) &&
					 (nTail == other.nTail) );
		}
		
		public String toString() {
			return nTail + "->" + nHead + " | " + nOperation;
		}
	}
	
	
	/**
	 * Singleton, record changes in the score for different steps.
	 * 
	 * @author Yanpeng Zhao
	 * 
	 */
	protected static class Cache {
		
		// change of score due to adding/deleting an arc
		private static int nNode = 0;
		private static double[][] fDeltaScoreAdd = null;
		private static double[][] fDeltaScoreDel = null;
		
		private static Cache instance = null;
		
		
		// constructor
		private Cache(int nNrOfNode, boolean flag) {
			nNode = nNrOfNode;
			fDeltaScoreAdd = new double[nNrOfNode][nNrOfNode];
			fDeltaScoreDel = new double[nNrOfNode][nNrOfNode];
			
			updateCache();
			
			// @debug
			System.out.println("Different cache with nNode: " + nNode);
		}	
		
		public static Cache getInstance(int nNrOfNode) {
			if ( instance == null || nNode != nNrOfNode ) {
				instance = new Cache(nNrOfNode, true);
			} else {
				updateCache();
			}
			return instance;
		}
		
		public static void updateCache() {
			for ( int i = 0; i < nNode; i++ ) {
				for ( int j = 0; j < nNode; j++ ) {
					fDeltaScoreAdd[i][j] = 0;
					fDeltaScoreDel[i][j] = 0;
				}
			}
		}
		
		// set caches
		public void put(Operation operation, double fValue) {
			if ( operation.nOperation == Operation.OPERATION_ADD ) {
				fDeltaScoreAdd[operation.nTail][operation.nHead] = fValue;
			} else {
				fDeltaScoreDel[operation.nTail][operation.nHead] = fValue;
			}
		}
		
		// get caches
		public double get(Operation operation) {
			switch( operation.nOperation ) {
			case Operation.OPERATION_ADD:
				return fDeltaScoreAdd[operation.nTail][operation.nHead];
			case Operation.OPERATION_DEL:
				return fDeltaScoreDel[operation.nTail][operation.nHead];
			case Operation.OPERATION_REVERSE:
				return fDeltaScoreDel[operation.nTail][operation.nHead] +
					   fDeltaScoreAdd[operation.nHead][operation.nTail];
			}
			return 0;
		}
	}

}
