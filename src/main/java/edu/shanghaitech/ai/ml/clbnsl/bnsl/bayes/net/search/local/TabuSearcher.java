package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.search.local;

import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.BayesNet;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

public class TabuSearcher extends CLSearcher {
	
	private int nTabu = 100;
	private Operation[] tabuList = null;
	
	
	public TabuSearcher(String basePath,
			Instances instances, 
			int stepLength, 
			int nFirstAdded,
			int[][] candidateSet,
			boolean bDebug,
			boolean bUseRandom,
			boolean bUsePartition,
			boolean bUseCandidate,
			boolean bUseLimitadd) {
		super(basePath, instances, stepLength, nFirstAdded, candidateSet, bDebug, bUseRandom, bUsePartition, bUseCandidate, bUseLimitadd);
	}
	
	
	protected void search(BayesNet bayesNet, Instances instances) throws Exception {

		int iTabu = 0, count = 0;
		boolean change = false;
		double bestScore = 0;
		double currentScore = 0.0, previousScore = 0.0;
		
		for ( int i = 0; i < instances.numAttributes(); i++ ) {
			currentScore += calcNodeScoreCurricula(i);
		}
		
		bestScore = currentScore;
		previousScore = currentScore;
		
		BayesNet bestBayesNet = new BayesNet();
		bestBayesNet.m_Instances = instances;
		bestBayesNet.initStructure();
		copyParentSets(bestBayesNet, bayesNet);
		
		tabuList = new Operation[nTabu];
		instances.setNumPattern(0);
		// no curricula
		for ( int i = 0; i < nNode; i++ ) {
			listOfCL[i] = i;
		}
		this.end = this.nNode;
		initCache(bayesNet, instances);
		
		// training and timing
		double buildTime, timeBoundary = bayesNet.getTimeBoundary();
     	long endTime, startTime = System.currentTimeMillis();
     	
		while ( true ) {
			Operation operation = getOptimalOperation(bayesNet, instances);
			
			if ( operation != null ) {
				performOperation(bayesNet, instances, operation);
				
				tabuList[iTabu] = operation;
				iTabu = (iTabu + 1) % nTabu;
				
				currentScore += operation.fDeltaScore;
				if ( currentScore > bestScore ) {
					change = true;
					bestScore = currentScore;
					copyParentSets(bestBayesNet, bayesNet);
				} 
			} // sanity check
			
//			System.out.println(change + " " + currentScore + " " + previousScore + " " + bestScore);
	
//			if ( !change ) {
//				if ( ++count >= 15 ) { break; }
//			} else {
//				count = 0;
//				change = false;
//			}
			
			previousScore = currentScore;
			
			endTime = System.currentTimeMillis();
			buildTime = (double)(endTime - startTime) / 1000;
			if ( buildTime > timeBoundary ) { break; }
		}
		copyParentSets(bayesNet, bestBayesNet);
		System.out.println("Time on Search: " + buildTime + "s");
	}
	
	
	protected void searchb(BayesNet bayesNet, Instances instances) throws Exception {
		instances.setNumPattern(0);
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
	
	
	protected void copyParentSets(BayesNet des, BayesNet src) {
		int nNode = src.getNNode();
		for ( int i = 0; i < nNode; i++ ) {
			des.getParentSet(i).copy(src.getParentSet(i));
		}
	}
	
	
	protected boolean isNotTabu(Operation operation) {
		for ( int i = 0; i < nTabu; i++ ) {
			if ( operation.equals(tabuList[i]) ) {
				return false;
			}
		}
		return true;
	}
	
	
    public void setStepSize(int stepSize) {

    }
}
