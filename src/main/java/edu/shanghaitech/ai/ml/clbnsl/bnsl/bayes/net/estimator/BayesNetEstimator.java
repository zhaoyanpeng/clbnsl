package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.estimator;

import java.util.Enumeration;
import java.util.HashMap;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.BayesNetNode;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.BayesNetStruct;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instance;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Statistics;

/**
 * Bayesian Network structure estimator.
 * 
 * @author Yanpeng Zhao
 * 5/5/2015
 */
public class BayesNetEstimator extends BayesNetStruct{
	
	private Instances instances;
	private Instances testInsts;
	
	private int nClass;
	private int[][] confusionMatrix;
	
	// equivalent sample size(ESS), 1 by default
	private double ess = 1.0; 
	private double BDeu;
	private double BIC;
	private double LL;
	private double KL;
	
	// log likelihood
	private double llOfInsts;
	
	
	/**
	 * See details in the class BayesNetStruct.
	 * 
	 * @param headerPath 
	 * 		descriptions about nodes
	 * @param structPath 
	 * 		structure file
	 * @param ess		 
	 * 		equivalent sample size
	 * @param sourceid   
	 * 		type of the structure
	 * 
	 */
	public BayesNetEstimator(
			String headerPath, 
			String structPath, 
			double ess, 
			int sourceid) throws Exception {
		super(headerPath, structPath, sourceid);
		this.ess = ess;
		this.BDeu= 0;
		this.BIC = 0;
		this.LL  = 0;
		this.KL  = 0;
	}
	
	
	/**
	 * Verify the test set is valid.
	 * 
	 * @param instances 
	 * 		test set
	 * 
	 */
	protected boolean isValid(Instances instances) {
		if ( instances == null || instances.numInstances() < 1 ) {
			return false;
		}
		return true;
	}
	
	
	/**
	 * For the purpose of debug.
	 * 
	 * @param node 
	 * 		of which the cpt will be printed
	 * 
	 */
	protected void printNodeCPD(BayesNetNode node) {
		Integer number = null;
		int nValue = node.getValSet().length;
		int cardinality = node.getCardinalityOfParSet();
		HashMap<Integer, Integer> counts = node.getSparseCPD();
		
		System.out.println("---------> Node: " + node.iName + " : " + node.getName());
		for ( int j = 0; j < cardinality; j++ ) {
			double Nij = 0;
			for ( int k = 0; k < nValue; k++ ) {
				number = counts.containsKey(new Integer(j * nValue + k)) ? counts.get(new Integer(j * nValue + k)).intValue() : 0;
				Nij += number;
				System.out.print(number + " ");
			}
			System.out.println(" = " + Nij);
		}
	}
	
	
	/**
	 * An entrance, traverse every node, see {@link #printNodeCPD(BayesNetNode)}.
	 * 
	 */
	protected void printCPD() {
		for ( int i = 0; i < nNode; i++ ) {
			printNodeCPD(BNNodes[i]);
		}
	}
	
	
	/**
	 * Compute the log likelihood of the given instance. We set the prior of the frequency to 1. 
	 * 
	 * @param instance 
	 * 		on which the estimation is conducted
	 * @return         
	 * 		log likelihood of the instance
	 * 
	 */
	private double instanceLL(Instance instance) {
		double likelihood = 0, iCPT = 0;
		Integer key = null, Nij = null, Nijk = null;
		int nValue, frequency;
		int[] parentSet = null;
		BayesNetNode bnNode = null;
		HashMap<Integer, Integer> counts = null;
		
		for ( int i = 0; i < nNode; i++ ) {
			bnNode = BNNodes[i]; 
			nValue = bnNode.getValSet().length;
			counts = bnNode.getSparseCPD();
			parentSet = bnNode.getParSet();
			
			iCPT = 0;
	        for (int iParent = 0; iParent < parentSet.length; iParent++) {
	        	int iNode = parentSet[iParent];
	        	iCPT = iCPT * instances.attribute(iNode).numValues() + instance.value(iNode);
	        }
	        
	        // System.out.print("-> " + i + "\t" + iCPT + "\t"); // @debug
	        Nij = new Integer(0);
	        for ( int iValue = 0; iValue < nValue; iValue++ ) {
	        	key = new Integer(nValue * ((int)iCPT) + iValue);
	        	Nijk = counts.containsKey(key) ? counts.get(key).intValue() : 0;
	        	Nij += Nijk;
	        	// System.out.print(key + ":" + Nijk + " "); // @debug
	        }
	        
	        key = new Integer(nValue * ((int)iCPT) + (int)instance.value(i));
	        frequency = counts.containsKey(key) ? counts.get(key).intValue() + 1 : 1; // prior
	        Nij += nValue; // prior
	        
	        // System.out.print(frequency + "/" + Nij + ";\n"); // @debug
	        likelihood += Math.log((double) frequency / Nij);
    	}
		return likelihood;
	}
	
	
	/**
	 * Compute local conditional probability of <tt>nodeIndex</tt>.
	 * 
	 * @param nodeIndex 
	 * 		of which the local conditional probability will be returned.
	 * @param instance  
	 * 		a single instance.
	 * @return
	 * 
	 */
	private double localNodeLL(int nodeIndex, Instance instance) {
		double likelihood = 0, iCPT = 0;
		Integer key = null, Nij = null, Nijk = null;
		BayesNetNode bnNode = BNNodes[nodeIndex]; 
		
		int frequency;
		int[] parentSet = bnNode.getParSet();
		int nValue = bnNode.getValSet().length;
		
		HashMap<Integer, Integer> counts = bnNode.getSparseCPD();
		
        for (int iParent = 0; iParent < parentSet.length; iParent++) {
        	int iNode = parentSet[iParent];
        	iCPT = iCPT * instances.attribute(iNode).numValues() + instance.value(iNode);
        }
        
        // System.out.print("-> " + nodeIndex + "\t" + iCPT + "\t"); // @debug
        
        Nij = new Integer(0);
        for ( int iValue = 0; iValue < nValue; iValue++ ) {
        	key = new Integer(nValue * ((int)iCPT) + iValue);
        	Nijk = counts.containsKey(key) ? counts.get(key).intValue() : 0;
        	Nij += Nijk;
        	// System.out.print(key + ":" + Nijk + " "); // @debug
        }
        
        key = new Integer(nValue * ((int)iCPT) + (int)instance.value(nodeIndex));
        frequency = counts.containsKey(key) ? counts.get(key).intValue() + 1 : 1; // prior
        Nij += nValue; // prior
        
        // System.out.print(frequency + "/" + Nij + "; "); // @debug
        likelihood = Math.log((double) frequency / Nij);
        return likelihood;
	}
	
	
	/**
	 * Compute local conditional probabilities involving class attribute and its children.
	 * 
	 * @param instance a single instance
	 * @return
	 * 
	 */
	private double localClassLL(Instance instance) {
		double likelihood = 0;
		int classIndex = instance.classIndex();
		
		for ( int i = 0; i < nNode; i++ ) {
			
			if ( BNNodes[i].hasParent(classIndex) ) {
				likelihood += localNodeLL(i, instance);
				// System.out.println(); // @debug
			}
    	}
		likelihood += localNodeLL(classIndex, instance);
		// System.out.println(); // @debug
		return likelihood;
	}
	
	
	/**
	 * According to the equation described below<p>.
	 * 
	 * max_{c} P\left(C = c | d[X/C]\right) 
	 * = max_{c} \frac{P\left(c, d[X/C]\right)}{P\left(d[X/C]\right)}
	 * = max_{c } P\left(c, d[X/C]\right)
	 * 
	 * @param testSet 
	 * 		on which the classifier will be evaluated.
	 * 
	 */
	protected void predict(Instances testSet) {
		if ( !isValid(testSet) ) {
			System.out.println("Prediction Err: |samples| < 1");
			System.exit(0);
		}
		this.testInsts = testSet;
		// reserve memory for predictions
		initStatisticMatrix();
		
		Instance instance = null;
		
		double probability, ltmp;
		int label, predictedLabel = -1, count = 0;
		
		Enumeration<Instance> enumInsts = testInsts.enumerateInstances();
	    while (enumInsts.hasMoreElements()) {
	    	
	    	instance = enumInsts.nextElement();
	    	label    = (int) instance.classValue();
	    	
	    	probability = -Integer.MAX_VALUE;
	    	for ( int i = 0; i < nClass; i++ ) {
	    		
	    		instance.setClassValue(String.valueOf(i));
	    		
//	    		ltmp = instanceLL(instance);   // using whole instance
	    		ltmp = localClassLL(instance); // using local (markov blanket) info
	    		
	    		// System.out.print(Math.exp(ltmp) + "\t" + ltmp + ";\n"); // @debug
	    		
	    		if ( probability < ltmp ) {
	    			probability = ltmp;
	    			predictedLabel = i;
	    		}
	    	}
	    	// System.out.println(label + "\tvs\t" + predictedLabel); // @debug
	    	confusionMatrix[label][predictedLabel] += 1;
	    	
	    	// if ( ++count >= 3) { break; } // @debug
	    }
	}
	
	
	/**
	 * According to the equation described below<p>.
	 * 
	 * \prod_{i = 1}^{m} p(C = c^{\star} | d[X/C]) = 
	 * \prod_{i = 1}^{m} p(d[C], d[X/C]) / p(d[X/C]) = 
	 * \prod_{i = 1}^{m} p(c^{\star}, d[X/C]) / p(d[X/C]), where C = c^{\star} is the truth label.
	 * 
	 * @param testSet 
	 * 		on which the classifier will be evaluated.
	 * 
	 */
	protected void query(Instances testSet) {
		if ( !isValid(testSet) ) {
			System.out.println("Prediction Err: |samples| < 1");
			System.exit(0);
		}
		this.testInsts = testSet;
		this.llOfInsts = 0;
		
		int label;
		Instance instance = null;
		
		double ltmp, evidence = 0, probability, conditionalProb;

		Enumeration<Instance> enumInsts = testInsts.enumerateInstances();
	    while (enumInsts.hasMoreElements()) {
	    	instance = enumInsts.nextElement();
	    	label    = (int) instance.classValue();
	    	evidence = 0;
	    	probability = -Integer.MAX_VALUE;
	    	for ( int i = 0; i < nClass; i++ ) {
	    		instance.setClassValue(String.valueOf(i));
	    		ltmp = instanceLL(instance);
	    		evidence += Math.exp(ltmp);
	    		
	    		// System.out.print(Math.exp(ltmp) + "\t" + ltmp + ";\t"); // @debug
	    		
	    		if ( i == label ) {
	    			probability = Math.exp(ltmp);
	    		}
	    	}
	    	conditionalProb = probability / evidence;
	    	llOfInsts += Math.log(conditionalProb);
	    	// System.out.println(Math.log(conditionalProb) + "\tvs\t" + conditionalProb); // @debug
	    }
	}

	
	/**
	 * Estimate Bayesian network parameters on training set. To ensure of sufficient precision,
	 * instead of storing the real conditional probabilities (parameters), we store the corresponding 
	 * counter (occurrence frequency). So when we use the parameters, we need to divide the corresponding 
	 * counter by the size of population(number of instances in the training set).<p>
	 * 
	 * Counter is stored in the sparse format. i.e., a variable <tt>A</tt>, can take two states, has three 
	 * parents <tt>P1, P2, P3</tt> which can take <tt>2, 3, 2</tt> states respectively. We have:<p>
	 * <pre>
	 *  i  P3  P2  P1 a0 a1
	 *  0   0   0   0  .  .
	 *  2   1   0   0  .  .
	 *  4   0   1   0  .  .
	 *  6   1   1   0  .  .
	 *  8   0   2   0  .  .
	 * 10   1   2   0  .  .
	 * 12   0   0   1  .  .
	 * 14   1   0   1  .  .
	 * 16   0   1   1  .  .
	 * 18   1   1   1  .  .
	 * 20   0   2   1  .  .
	 * 22   1   2   1  .  .
	 * <pre>
	 * 
	 * @param trainSet 
	 * 		on which the parameters are estimated.
	 * 
	 */
	protected void estimateParameters(Instances trainSet) {
		if ( !isValid(trainSet) ) {
			System.err.println("Estimate Parameters Err: |samples| < 1");
			System.exit(0);
		}
		this.instances = trainSet;
		this.nClass    = instances.numClasses();
		
		Integer key = null;
		Instance instance = null;
		
		int nValue;
		double iCPT;
		int[] parentSet = null;
		HashMap<Integer, Integer> counts = null;
		
		for ( int i = 0; i < nNode; i++ ) {
			BayesNetNode bnNode = BNNodes[i];
			nValue = bnNode.getValSet().length;
			
			parentSet = bnNode.getParSet();
			
			// each of the nodes keep a counter to memory the frequencies
			counts = new HashMap<Integer, Integer>();
			counts.clear();
		    
			Enumeration<Instance> enumInsts = instances.enumerateInstances();
		    while (enumInsts.hasMoreElements()) {
		    	iCPT = 0;
		    	instance = enumInsts.nextElement();
		        for (int iParent = 0; iParent < parentSet.length; iParent++) {
		        	int iNode = parentSet[iParent];
		        	iCPT = iCPT * instances.attribute(iNode).numValues() + instance.value(iNode);
		        } // how to compute? see example above.
		
		        key = new Integer(nValue * ((int)iCPT) + (int)instance.value(i));
	    		if ( counts.containsKey(key) ) {
	    			counts.put(key, new Integer(counts.get(key).intValue() + 1));
	    		} else {
	    			counts.put(key, new Integer(1));
	    		}
		    }
		    // associate with the corresponding node
		    bnNode.setSparseCPD(counts); // set sparse cpt
		}
		// printCPD();
	}

	
	/**
	 * Currently only support BDeu, BIC, LL and KL metric.
	 * 
	 */
	public void evaluate(Instances instances) {
		isValid(instances);
		this.instances = instances;
		Integer key = null;
		Instance instance = null;

		double iCPT;
		int cardinality, nValue;
		int[] parentSet = null;
		
		HashMap<Integer, Integer> counts = new HashMap<Integer, Integer>();
		
		for ( int i = 0; i < nNode; i++ ) {
			BayesNetNode bnNode = BNNodes[i];
			cardinality = bnNode.getCardinalityOfParSet();
			nValue = bnNode.getValSet().length;
			parentSet = bnNode.getParSet();
			
			counts.clear();
		    Enumeration<Instance> enumInsts = instances.enumerateInstances();
		    while (enumInsts.hasMoreElements()) {
		    	iCPT = 0;
		    	instance = enumInsts.nextElement();
		        for (int iParent = 0; iParent < parentSet.length; iParent++) {
		        	int iNode = parentSet[iParent];
		        	iCPT = iCPT * instances.attribute(iNode).numValues() + instance.value(iNode);
		        }
		
		        key = new Integer(nValue * ((int)iCPT) + (int)instance.value(i));
	    		if ( counts.containsKey(key) ) {
	    			counts.put(key, new Integer(counts.get(key).intValue() + 1));
	    		} else {
	    			counts.put(key, new Integer(1));
	    		}
		    }
		    
			this.BDeu += calcScoreBDeu(counts, cardinality, nValue, false);
			this.LL   += calcScoreLL(counts, cardinality, nValue, false);
			this.BIC   = this.LL - 0.5 * cardinality * (nValue - 1) * 
							Math.log(instances.numInstances());
			if ( cardinality > 1 ) {
				this.KL   += calcScoreKL(counts, cardinality, nValue, false);
			}
		}
		this.KL /= instances.numInstances();
		counts.clear();
	}
	
	
	protected double calcScoreBDeu(HashMap<Integer, Integer> counts, int cardinality, int nValue, boolean flag) {
		int number;
		double score = 0.0, Nij;
		for ( int j = 0; j < cardinality; j++ ) {
			Nij = 0;
			for ( int k = 0; k < nValue; k++ ) {
				number = counts.containsKey(new Integer(j * nValue + k)) ? counts.get(new Integer(j * nValue + k)).intValue() : 0;
				if ( AUtils.A_SMALL_FLOAT_CONSTANT + number > 0 ) {
					score += Statistics.lnGamma(1.0 / (nValue * cardinality) + number);
					Nij += ess / (nValue * cardinality) + number;
				}
			}
			score -= Statistics.lnGamma(Nij);
			score -= nValue * Statistics.lnGamma(1.0 / (nValue * cardinality));
			score += Statistics.lnGamma(ess / cardinality);
		}
		return score;
	}

	
	protected double calcScoreLL(HashMap<Integer, Integer> counts, int cardinality, int nValue, boolean flag) {
		int number;
		double score = 0.0, Nij;
		for ( int j = 0; j < cardinality; j++ ) {
			Nij = 0;
			for ( int k = 0; k < nValue; k++ ) {
				number = counts.containsKey(new Integer(j * nValue + k)) ? counts.get(new Integer(j * nValue + k)).intValue() : 0;
				Nij += number;
			}
			
			if ( Nij <= 0 ) { continue; }
			
			for ( int k = 0; k < nValue; k++ ) {
				number = counts.containsKey(new Integer(j * nValue + k)) ? counts.get(new Integer(j * nValue + k)).intValue() : 0;
				if ( number > 0 ) {
					score += number * Math.log(number / Nij);
				}
			}
		}
		return score;
	}
	
	
	protected double calcScoreKL(HashMap<Integer, Integer> counts, int cardinality, int nValue, boolean flag) {
		int number;
		double score = 0.0;
		int[] Nik = new int[nValue];
		for ( int k = 0; k < nValue; k++ ) {
			for ( int j = 0; j < cardinality; j++ ) {
				number = counts.containsKey(new Integer(j * nValue + k)) ? counts.get(new Integer(j * nValue + k)).intValue() : 0;
				Nik[k] += number;
			}
		}
		
		for ( int j = 0; j < cardinality; j++ ) {
			double Nij = 0;
			for ( int k = 0; k < nValue; k++ ) {
				number = counts.containsKey(new Integer(j * nValue + k)) ? counts.get(new Integer(j * nValue + k)).intValue() : 0;
				Nij += number;
			}
			
			if ( Nij <= 0 ) { continue; }
			
			for ( int k = 0; k < nValue; k++ ) {
				number = counts.containsKey(new Integer(j * nValue + k)) ? counts.get(new Integer(j * nValue + k)).intValue() : 0;
				if ( number > 0 ) {
					score += number * (Math.log(number) + Math.log(instances.numInstances()) - Math.log(Nij * Nik[k]));
				}
			}
		}
		return score;
	}
	
	
	/**
	 * Initialize confusion and likelihood matrix.
	 * 
	 */
	private void initStatisticMatrix() {
		this.confusionMatrix  = new int[nClass][nClass]; 
		if ( confusionMatrix != null ) {
			for ( int i = 0; i < nClass; i++ ) {
				for ( int j = 0; j < nClass; j++ ) {
					confusionMatrix[i][j] = 0;
				}
			}
		}
	}
	
	
	protected int[][] getConfusionMatrix() {
		return this.confusionMatrix;
	}
	
	
	protected double getLLOfInsts() {
		return this.llOfInsts;
	}
	
	public double getBDeuScore() {
		return this.BDeu;
	}
	
	
	public double getBICScore() {
		return this.BIC;
	}
	
	
	public double getLLScore() {
		return this.LL;
	}
	
	
	public double getKLScore() {
		return this.KL;
	}
}
