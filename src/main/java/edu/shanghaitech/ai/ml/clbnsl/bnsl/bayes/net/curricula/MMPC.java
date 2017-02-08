package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.curricula;

import java.io.FileWriter;
import java.util.LinkedList;
import java.util.Vector;

import net.sourceforge.jdistlib.ChiSquare;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instance;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * Implementation of MMPC. (<a href="http://link.springer.com/article/10.1007/s10994-006-6889-7">Reference</a>).<p>
 * 
 * @author Yanpeng Zhao
 * 5/18/2015
 */
public class MMPC {
	
	private int nNode;
	private int samplesize;
	private int nMaxNeighbor = -1;
	private double[][] mutualInfo = null;
	
	private Vector<Integer> candidates;
	private Vector<Integer> intermediate;
	private Vector<Vector<Integer>> neighborset;
	
	private Instances instances;

	public MMPC(Instances instances, double[][] mutualInfo) {
		this.instances = instances;
		this.nNode = instances.numAttributes();
		this.samplesize = instances.numInstances();
		this.mutualInfo = mutualInfo;
		this.candidates = new Vector<Integer>();
		this.intermediate = new Vector<Integer>();
		this.neighborset = new Vector<Vector<Integer>>();
	}
	
	
	public MMPC(Instances instances, double[][] mutualInfo, int nMaxNeighbor) {
		this.instances = instances;
		this.nNode = instances.numAttributes();
		this.samplesize = instances.numInstances();
		this.mutualInfo = mutualInfo;
		this.nMaxNeighbor = nMaxNeighbor;
		this.candidates = new Vector<Integer>();
		this.intermediate = new Vector<Integer>();
		this.neighborset = new Vector<Vector<Integer>>();
	}
	
	
	public void generatePCsets(String pcsetFilePath) throws Exception {
		Vector<Integer> neighbors;
		StringBuffer sb = new StringBuffer();
//		sb.append(nNode + "\n");
		for ( int i = 0; i< nNode; i++ ) {
			neighbors = getPCset(i);
			neighborset.add(neighbors);
			
			for ( Integer neighbor : neighbors ) {
				sb.append(neighbor.intValue() + "\t");
			}
			sb.append("\n");
		}
		FileWriter pcset = new FileWriter(pcsetFilePath);
		pcset.write(sb.toString());
		pcset.close();
		System.out.println("Save to: " + pcsetFilePath);
	}
	
	
	public Vector<Vector<Integer>> getPCsets(String pcsetFilePath) throws Exception {
		Vector<Integer> neighbors;
		for ( int i = 0; i< nNode; i++ ) {
			neighbors = getPCset(i);
			neighborset.add(neighbors);
		}
		
		print2DVector(neighborset);
		
		if ( pcsetFilePath != null ) {
			write2Dvector(neighborset, pcsetFilePath);
		}
		return neighborset;
	}
	
	
	public Vector<Integer> getPCset(int target) {
		updateOrderedCandidate(target);
		// @debug
		System.out.print("Init: cand-> | ");
		printVector(candidates);
		// no filter
//		Vector<Integer> neighbors = filterOrderedCandidate(target);
		Vector<Integer> neighbors = new Vector<Integer>();
		// @debug
		System.out.print("Init: cand-> | ");
		printVector(candidates);
		System.out.print("Init: neig-> | ");
		printVector(neighbors);
		
		double tmp;
		boolean goon = true;
		int iStart, iNode, iMaxAssoc, nzero, count, lastElement = -1;
		// cache technique
		double[] assocs = new double[nNode - 1];
		double[] cachedMinAssoc = new double[nNode - 1];
		Vector<Vector<Integer>> cachedSubSets = new Vector<Vector<Integer>>();
		for ( int i = 0; i < nNode - 1; i++ ) {
			cachedMinAssoc[i] = Double.MAX_VALUE;
		}
		// 0st phase
		while ( goon ) {
			iStart = cachedSubSets.size();
			updateCachedSubSets(lastElement, cachedSubSets);
			// @debug
//			System.out.println("0size: " + iStart + "\t1size: " + cachedSubSets.size());
//			print2DVector(cachedSubSets);
			for ( int i = 0; i < candidates.size(); i++ ) {
				iNode = candidates.get(i).intValue();
//				assocs[i] = getMinAssoc(iNode, target, neighbors);
				assocs[i] = getMinAssocCached(iStart, iNode, target, cachedMinAssoc[i], neighbors, cachedSubSets);
			}
			
			// @debug
			printVector(1, candidates.size(), assocs);
			System.out.print("1Target: " + target + " | ");
			printVector(candidates);
			
			nzero = 0;
			tmp = -0.1;
			iMaxAssoc = -1;
			for ( int i = 0; i < candidates.size(); i++ ) {
				if ( tmp < assocs[i] ) {
					tmp = assocs[i];
					iMaxAssoc = i;
				}
				if ( assocs[i] == 0 ) { nzero++; }
			}
			
			// there must exist a p-value larger than -0.1
			// so iMaxAssoc won't be -1
			if ( assocs[iMaxAssoc] != 0 ) {
				neighbors.add(candidates.get(iMaxAssoc));
				lastElement = candidates.get(iMaxAssoc).intValue();
			} else {   // termination condition 0
				System.out.println("->Break by condition 0");
				break; // no more dependences
			}
			
			// termination condition 1 & 2
			// only exist one node dependent on target
			if ( nzero == candidates.size() - 1 ) { 
				System.out.println("->Break by condition 1");
				break; 
			} 
			// if have set nMaxNeighbor and the number of the current neighbors
			// is larger than the max setting
			if ( nMaxNeighbor > 0 && neighbors.size() >= nMaxNeighbor ) { 
				System.out.println("->Break by condition 2");
				break; 
			}
			
			count = 0;
			intermediate.clear();
			for ( int i = 0; i < candidates.size(); i++ ) {
				if ( assocs[i] != 0 && i != iMaxAssoc ) {
					intermediate.add(candidates.get(i));
					// backup minimum association
					cachedMinAssoc[count] = assocs[i]; 
					count++;
				}
			}

			candidates.clear();
			for ( int i = 0; i < intermediate.size(); i++ ) {
				candidates.add(intermediate.get(i));
			}
			intermediate.clear();
			
			// @debug
			System.out.print("2Target: " + target + " | ");
			printVector(neighbors);
			
//			System.exit(0);
		}
		
		// @debug
		System.out.print("3Target: " + target + " | ");
		printVector(neighbors);

		// 1st phase
		Vector<Integer> subneighbors = new Vector<Integer>();
		for ( int i = 0; i < neighbors.size(); i++ ) {
			
			subneighbors.clear();
			for ( int j = 0; j < neighbors.size(); j++ ) {
				if ( i != j ) {
					subneighbors.add(neighbors.get(j));
				}
			}
			assocs[i] = getMinAssoc(neighbors.get(i).intValue(), target, subneighbors);
		}
		
		// @debug
		System.out.print("4Target: " + target + " | ");
		printVector(1, neighbors.size(), assocs);
		
		intermediate.clear();
		for ( int i = 0; i < neighbors.size(); i++ ) {
			if ( assocs[i] != 0 ) {
				intermediate.add(neighbors.get(i));
			}
		}
		
		neighbors.clear();
		for ( int i = 0; i < intermediate.size(); i++ ) {
			neighbors.add(intermediate.get(i));
		}
		intermediate.clear();
		
		// @debug
		System.out.print("5Target: " + target + " | ");
		printVector(neighbors);
		System.out.println("\nTarget: " + target + " over\n");
		
		return neighbors;
	}
	
	
	protected Vector<Integer> filterOrderedCandidate(int target) {
		
		Vector<Integer> neighbors = new Vector<Integer>();
		for ( int i = 0; i < neighborset.size(); i++ ) {
			Vector<Integer> preNeighbors = neighborset.get(i);
			
			// @debug
			System.out.print("Filter: " + i + " | ");
			printVector(preNeighbors);
			
			if ( isContained(target, preNeighbors) ) {
				neighbors.add(new Integer(i));
				candidates.remove(new Integer(i));
			} 

		}
		return neighbors;
	}
	
	
	protected boolean isContained(int target, Vector<Integer> neighbor) {
		
		for ( int j = 0; j < neighbor.size(); j++ ) {
			if ( target == neighbor.get(j).intValue() ) {
				return true;
			}
		}
		return false;
	}
	
	
	protected int getCardinal(Vector<Integer> condition) {
		int cardinal = 1;
		for ( int i = 0; i < condition.size(); i++ ) {
			cardinal *= instances.attribute(condition.get(i).intValue()).numValues();
		}
		return cardinal;
	}
	
	
	protected int getNParam(int iNode, int target, Vector<Integer> condition, boolean isDegreeOfFreedom) {
		int df;
		if ( isDegreeOfFreedom ) { // actual degree of freedom
			df = (instances.attribute(iNode).numValues() - 1) * (instances.attribute(target).numValues() - 1);
		} else {
			df = instances.attribute(iNode).numValues() * instances.attribute(target).numValues();
		}
		df *= getCardinal(condition);
		return df;
	}
	
	
	protected void copyIntVector(Vector<Integer> des, Vector<Integer> ori) {
		des.clear();
		for ( Integer element : ori ) { des.add(element); }
	}
	
	
	protected void updateCachedSubSets(
			int lastElement, 
			Vector<Vector<Integer>> cachedSubSets) {
		int cardinal = 1, iStart;
		
		// make cached conditional set
		if ( lastElement < 0 ) {
			Vector<Integer> subSet = new Vector<Integer>();
			cachedSubSets.add(subSet);
		} else {
			iStart = cachedSubSets.size();
			for ( int i = 0; i < iStart; i++ ) {
				Vector<Integer> subSet = new Vector<Integer>();
				copyIntVector(subSet, cachedSubSets.get(i));
				
				subSet.add(new Integer(lastElement));
				// roughly filter some meaningless conditional set
				cardinal = getCardinal(subSet);
				if ( samplesize >= 5 * cardinal) {  // avoid small sample size
					cachedSubSets.add(subSet);
				}
			}
		}
	}
	
	
	protected double getMinAssocCached(
			int iStart,
			int iNode, 
			int target, 
			double cachedMinAssoc,
			Vector<Integer> condition, 
			Vector<Vector<Integer>> cachedSubSets) {
		
		int df;
		double assoc = .0, minAssoc = cachedMinAssoc;

		for ( int i = iStart; i < cachedSubSets.size(); i++ ) {
			df = getNParam(iNode, target, cachedSubSets.get(i), true);
			// avoid small sample size, safe check
//			if ( samplesize < 5 * df) { return 0; } 
			if ( samplesize < 5 * df) { continue; }  
			
			// getPValue returns a p-value corresponds to the probability of 
			// falsely rejecting the null hypothesis given that it is true
			assoc = 1 - getPValue(iNode, target, cachedSubSets.get(i));
//			System.out.println("(->" + assoc);
			if ( assoc < 1 - AUtils.ALPHA ) { // 0.05 strong independence
				return 0;
			} else {
				if ( assoc < minAssoc) {
					minAssoc = assoc;
				}
			}
		}
//		return minAssoc;
		return minAssoc == Double.MAX_VALUE ? 0 : minAssoc;
	}
	
	
	protected double getMinAssoc(int iNode, int target, Vector<Integer> condition) {
		
		int initSizeOfSet = 0, df = 1;
		double assoc, minAssoc = Double.MAX_VALUE;
		for ( int i = initSizeOfSet; i <= condition.size(); i++ ) {
			Vector<Vector<Integer>> subSets = getSubSet(condition, i);
			// conditioned on every subset of condition
			for ( int j = 0; j < subSets.size(); j++ ) { 
				// do not perform an independence test (i.e., we assume independence)
				// unless there are at least five training instances on average per 
				// parameter (count) to be estimated.
				df = getNParam(iNode, target, subSets.get(j), true);
				// avoid small sample size
//				if ( samplesize < 5 * df) { return 0; }
				if ( samplesize < 5 * df) { continue; }
				
				// p-value corresponds to the probability of falsely rejecting 
				// the null hypothesis given that it is true
				// here 1 - pvalue is used for convenience
				assoc = 1 - getPValue(iNode, target, subSets.get(j));
				if ( assoc < AUtils.ALPHA ) { // 0.95 weak independence
					return 0;
				} else {
					if ( assoc < minAssoc) {
						minAssoc = assoc;
					}
				}
			}
		}
//		return minAssoc;
		return minAssoc == Double.MAX_VALUE ? 0 : minAssoc;
	}
	
	
	protected double getPValue(int iNode, int target, Vector<Integer> condition) {
		if ( condition.isEmpty() ) {
//			System.out.println("->Empty condition");
			return chiSquare(iNode, target);
		} else {
//			System.out.println("->Not Empty condition");
			return conditionalChiSquare(iNode, target, condition);
		}
	}
	
	
	/**
	 * @param iNode  
	 * 		the test node
	 * @param target 
	 * 		the target node
	 * @return 
	 * 		p-value of ChiSquare
	 */
	protected double chiSquare(int iNode, int target) {
		int nValue0 = instances.attribute(iNode).numValues();
		int nValue1 = instances.attribute(target).numValues();
		int[][] Nij = new int[nValue0][nValue1];
		for ( int i = 0; i < nValue0; i++ ) {
			for ( int j = 0; j < nValue1; j++ ) {
				Nij[i][j] = 0;
			}
		}
		
		int vi, vj;
		Instance obs;
		for ( int i = 0; i < samplesize; i++ ) {
			obs = instances.instance(i);
			vi = Integer.parseInt(obs.stringValue(iNode).trim());
			vj = Integer.parseInt(obs.stringValue(target).trim());
			
			Nij[vi][vj] += 1;
		}
		
		int[] Ni = new int[nValue0];
		int[] Nj = new int[nValue1];
		
		for ( int i = 0; i < nValue0; i++ ) {
			Ni[i] = 0;
			for ( int j = 0; j < nValue1; j++ ) {
				Ni[i] += Nij[i][j];
			}
		}
		for ( int j = 0; j < nValue1; j++ ) {
			Nj[j] = 0;
			for ( int i = 0; i < nValue0; i++ ) {
				Nj[j] += Nij[i][j];
			}
		}
		
		double x = 0, Eij, Dij;
		for ( int i = 0; i < nValue0; i++ ) {
			for ( int j = 0; j < nValue1; j++ ) {
				Eij = (double) Ni[i] * Nj[j] / samplesize; // 0? int overflow
				Dij = Nij[i][j] - Eij;
//				System.out.print(x + " " + Eij + " " + Ni[i] + " " + Nj[j] + "; ");
				if ( Eij != 0 ) { // avoid 0
					if ( Eij < 10 ) { Dij = Math.abs(Dij) - 0.5; }
//					if ( Eij < 5 ) { continue; }
					x += Math.pow(Dij, 2) / Eij;
				}
			}
		}
//		System.out.println();
		
		int df = (nValue0 - 1) * (nValue1 - 1);

//		System.out.println("-------------------------------------" + iNode + " + |" + x + "| + " + df);

		return ChiSquare.cumulative(x, df, false, false);
	}
	
	
	/**
	 * @param iNode     
	 * 		the test node
	 * @param target   
	 * 		the target node
	 * @param condition 
	 * 		the condition set
	 * @return 
	 * 		p-value of conditional ChiSquare
	 */
	protected double conditionalChiSquare(int iNode, int target, Vector<Integer> condition) {
		
		int nValue0 = instances.attribute(iNode).numValues();
		int nValue1 = instances.attribute(target).numValues();
		LinkedList<String> vconditions = new LinkedList<String>();
		
		@SuppressWarnings("unchecked")
		Vector<Integer>[][] Nijk = new Vector[nValue0][nValue1];
		for ( int i = 0; i < nValue0; i++ ) {
			for ( int j = 0; j < nValue1; j++ ) {
				Nijk[i][j] = new Vector<Integer>();
			}
		}
		
		Instance obs;
		int vi, vj, index;
		int size = condition.size();
		StringBuffer sb = new StringBuffer();
		for ( int i = 0; i < samplesize; i++ ) {
			
			obs = instances.instance(i);
			vi = Integer.parseInt(obs.stringValue(iNode).trim());
			vj = Integer.parseInt(obs.stringValue(target).trim());
			
			sb.delete(0, sb.length());
			for ( int j = 0; j < size - 1; j++ ) {
				sb.append(obs.stringValue(condition.get(j).intValue()) + "_");
			}
			sb.append(obs.stringValue(condition.get(size - 1).intValue()));
			
			index = -1;
			for ( int j = 0; j < vconditions.size(); j++ ) {
				if ( sb.toString().trim().equals(vconditions.get(j).trim()) ) {
					index = j;
					break;
				}
			}
			
			if ( index == -1 ) {
				index = vconditions.size();
				vconditions.add(sb.toString().trim());
				updateNijk(nValue0, nValue1, Nijk, vconditions);
				Nijk[vi][vj].set(index, 1);
			} else {
				Nijk[vi][vj].set(index, Nijk[vi][vj].get(index) + 1);
			}
		}
		
		int nvCondition = vconditions.size();
		int[] Nk    = new int[nvCondition];
		int[][] Nik = new int[nValue0][nvCondition];
		int[][] Njk = new int[nValue1][nvCondition];
		
		for ( int i = 0; i < nValue0; i++ ) {
			for ( int k = 0; k < nvCondition; k++ ) {
				Nik[i][k] = 0;
				for ( int j = 0; j < nValue1; j++ ) {
					Nik[i][k] += Nijk[i][j].get(k).intValue();
				}
			}
		}
		for ( int j = 0; j < nValue1; j++ ) {
			for ( int k = 0; k < nvCondition; k++ ) {
				Njk[j][k] = 0;
				for ( int i = 0; i < nValue0; i++ ) {
					Njk[j][k] += Nijk[i][j].get(k).intValue();
				}
			}
		}
		for ( int k = 0; k < nvCondition; k++ ) {
			Nk[k] = 0;
			for ( int i = 0; i < nValue0; i++ ) {
				Nk[k] += Nik[i][k];
			}
		}
		
		double x = 0, Eij, Dij;
		for ( int k = 0; k < nvCondition; k++ ) {
			for ( int i = 0; i < nValue0; i++ ) {
				for ( int j = 0; j < nValue1; j++ ) {
					Eij = (double) Nik[i][k] * Njk[j][k] / Nk[k]; // 0?
					Dij = Nijk[i][j].get(k) - Eij;
					if ( Eij != 0 ) { // avoid 0
						if ( Eij < 10 ) { Dij = Math.abs(Dij) - 0.5; }
//						if ( Eij < 5 ) { continue; }
						x += Math.pow(Dij, 2) / Eij;
					}
				}
			}
		}
		int df = this.getNParam(iNode, target, condition, true);
		return ChiSquare.cumulative(x, df, false, false);
	}
	
	
	protected void updateNijk(int nValue0, int nValue1, Vector<Integer>[][] Nijk, LinkedList<String> vconditions) {
		for ( int i = 0; i < nValue0; i++ ) {
			for ( int j = 0; j < nValue1; j++ ) {
				if ( Nijk[i][j].size() < vconditions.size() ) {
					for ( int k = Nijk[i][j].size(); k < vconditions.size(); k++ ) {
						Nijk[i][j].add(new Integer(0));
					}
				}
			}
		}
	}
	
	
	/**
	 * Sort candidate neighbor nodes of the target by descending mutual information.
	 */
	protected void updateOrderedCandidate(int target) {
		int tmpInd;
		int[] inds = new int[nNode - 1];
		double tmpMI;
		double[] mi = new double[nNode - 1];
		
		// initializing
		tmpInd = 0;
		for ( int i = 0; i < nNode; i++ ) {
			if ( i != target ) {
				mi[tmpInd] = mutualInfo[target][i];
				inds[tmpInd] = i;
				tmpInd++;
			}
		}
		
		// @debug
//		printVector(1, nNode - 1, mi);
		
		// bubble sort
		for ( int i = 0; i < nNode - 1; i++ ) {
			for ( int j = 0; j < nNode - i - 2; j++ ) {
				if ( mi[j] < mi[j + 1] ) {
					// exchange mutual info
					tmpMI = mi[j + 1];
					mi[j + 1] = mi[j];
					mi[j] = tmpMI;
					// exchange index
					tmpInd = inds[j + 1];
					inds[j + 1] = inds[j];
					inds[j] = tmpInd;
				}
			}
		}
		// @debug
//		printVector(1, nNode - 1, mi);

		candidates.clear(); 
		for ( int i = 0; i < nNode - 1; i++ ) {
			candidates.add(new Integer(inds[i]));
		}
		// @debug
//		printVector(1, nNode - 1, inds);
	}
	

	/**
	 * Get the subset of size n from the source recursively.
	 * 
	 * @param source 
	 * 		input set
	 * @param n      
	 * 		cardinality
	 * @return       
	 * 		set of the subsets of the source
	 * 
	 */
	protected Vector<Vector<Integer>> getSubSet(Vector<Integer> source, int n) {
		Vector<Vector<Integer>> subSets = new Vector<Vector<Integer>>();
		if ( n < 0 ) {
			System.out.println("Err: subset size must be positive.");
			System.exit(0);
		} else if ( n == 0 ) {
			Vector<Integer> subSet = new Vector<Integer>();
			subSets.add(subSet);
		} else {
			int sizeSet = source.size();
			if ( sizeSet >= n ) {
				if ( n == 1 ) {
					for ( int i = 0; i < sizeSet; i++ ) {
						Vector<Integer> subSet = new Vector<Integer>();
						subSet.add(source.get(i));
						subSets.add(subSet);
					}
				} else {
					for ( int i = 0; i < sizeSet; i++ ) {
						
						Vector<Integer> partialSet = new Vector<Integer>();
						for ( int j = i + 1; j < sizeSet; j++ ) {
							partialSet.add(source.get(j));
						}
						
						Vector<Vector<Integer>> thisSets = getSubSet(partialSet, n - 1);
						for ( int j = 0; j < thisSets.size(); j++ ) {
							Vector<Integer> thisSet = thisSets.get(j);
							Vector<Integer> subSet  = new Vector<Integer>();
							
							subSet.add(source.get(i));
							for ( int k = 0; k < thisSet.size(); k++ ) {
								subSet.add(thisSet.get(k));
							}
							subSets.add(subSet);
						}
					}
				}
			}
		}
		return subSets;
	}
	
	
	public void testGetPCset() throws Exception {
		this.getPCsets(null);
	}
	
	
	public void testChiSquare(int iNode, int target, Vector<Integer> condition) {
//		System.out.println("chi = " + chiSquare(iNode, target));
		System.out.println("chi = " + conditionalChiSquare(iNode, target, condition));
	}
	
	
	public void testBubbleSort(int target) {
		updateOrderedCandidate(target);
	}
	
	
	public void testSubSet(Vector<Integer> source, int n) {
		Vector<Vector<Integer>> subSets = getSubSet(source, n);
		for ( int i = 0; i < subSets.size(); i++ ) {
			Vector<Integer> subSet = subSets.get(i);
			for ( int j = 0; j < subSet.size(); j++ ) {
				System.out.print(subSet.get(j) + " ");
			}
			System.out.println("|");
		}
	}
	
	
	protected void write2Dvector(Vector<Vector<Integer>> neighborset, String file) throws Exception {
		Vector<Integer> neighbors;
		StringBuffer sb = new StringBuffer();
		for ( int i = 0; i < neighborset.size(); i++ ) {
			neighbors = neighborset.get(i);
			for ( Integer neighbor : neighbors ) {
				sb.append(neighbor.intValue() + "\t");
			}
			sb.append("\n");
		}
		FileWriter pcset = new FileWriter(file);
		pcset.write(sb.toString());
		pcset.close();
		System.out.println("PCSet Saved to: " + file);
	}
	
	
	protected void print2DVector(Vector<Vector<Integer>> neighborset) {
		for ( int i = 0; i < neighborset.size(); i++ ) {
			printVector(neighborset.get(i));
		}
	}
	
	
	protected void printVector(Vector<Integer> neighbors) {
		System.out.println(neighbors);
	}
	
	
	protected void printVector(int r, int c, double[] Nk) {
		for ( int i = 0; i < r; i++ ) {
			for ( int j = 0; j < c; j++ ) {
				System.out.print(Nk[j] + " ");
			}
			System.out.println();
		}
	}
	
	
	protected void printVector(int r, int c, int[] Nk) {
		for ( int i = 0; i < r; i++ ) {
			for ( int j = 0; j < c; j++ ) {
				System.out.print(Nk[j] + " ");
			}
			System.out.println();
		}
	}
	
	
	protected void printMatrix(int r, int c, int[][] Nij) {
		System.out.println();
		for ( int i = 0 ; i < r; i++ ) {
			for ( int j = 0; j < c; j++ ) {
				System.out.print(Nij[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	
	protected void print3DMatrix(int r, int c, int h, Vector<Integer>[][] Nijk) {
		for ( int k = 0; k < h; k++ ) {
			for ( int i = 0; i < r; i++ ) {
				for ( int j = 0; j < c; j++ ) {
					System.out.print(Nijk[i][j].get(k) + " ");
				}
				System.out.println();
			}
			System.out.println();
		}
	}
	
}
