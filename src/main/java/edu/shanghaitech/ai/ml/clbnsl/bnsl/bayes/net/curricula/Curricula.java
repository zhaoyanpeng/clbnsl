package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.curricula;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instance;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * @author Yetian Chen
 * @author Yanpeng Zhao
 * 4/7/2015
 */
public class Curricula {

	private int nNode;
	private int nCandidate = 0;
	private int[] listOfCL = null;
	private int[][] candidateSet = null;
	private double[][] mutualInfo = null;
	private double[][] chiSquare = null;
	
	private boolean bCLReverse = false;
	private Instances instances;
	
	
	public Curricula() {}
	
	
	public Curricula(boolean bCLReverse, Instances instances, int nCandidate) {
		this.instances = instances;
		this.bCLReverse = bCLReverse;
		this.nCandidate = nCandidate;
		this.nNode = instances.numAttributes();
		this.listOfCL = new int[nNode];
		// trivial, in fact we can traverse all the networks when the cardinality of 
		// all the nodes is small
		if ( nCandidate > nNode ) {
			this.nCandidate = nNode * 4 / 5;
		}
//		this.initChiquare();
		this.initMutualInfo();
	}
	
	
	public double[][] getMutualInfo() {
		if ( mutualInfo != null ) {
			return this.mutualInfo;
		} else {
			return null;
		}
	}
	
	
	public int[] getCList(int iMethod) {
		if ( this.mutualInfo == null ) {
			System.out.println("mutual info is null");
		}
		designCurricula(iMethod);
		if ( listOfCL == null ) { /*something wrong*/ }
		return this.listOfCL;
	}
	
	
	public int[][] getCandidateSet(int iMethod) {
		if ( iMethod == 0 ) {
			makeCandidateSet(0);
		} else if ( iMethod == 1 ) {
			makeCandidateSet(1);
		} else {
			System.out.println("->Undefined method for making candidate set.");
			return null;
		}
		return this.candidateSet;
	}
	
	
	public void initChiquare() {
		this.chiSquare = new double[nNode][nNode];
		HashMap<Integer, String[]> nodeAndValue = 
				new HashMap<Integer, String[]>();
		for ( int i = 0; i < nNode; i++ ) {
			int numValue = instances.attribute(i).numValues();
			String[] values = new String[numValue];
			for ( int j = 0; j < numValue; j++ ) {
				values[j] = instances.attribute(i).value(j);
			}
			// add into...
			nodeAndValue.put(i, values);
		}

		for ( int i = 0; i < nNode; i++ ) {
			for ( int j = i; j < nNode; j++ ) { 
				chiSquare[i][j] = computeChiquare(
						i, 
						nodeAndValue.get(i),
						j, 
						nodeAndValue.get(j),
						instances);
				chiSquare[j][i] = chiSquare[i][j];
			}
		}
	}

	
	public void initMutualInfo() {
		this.mutualInfo = new double[nNode][nNode];
		HashMap<Integer, String[]> nodeAndValue = 
				new HashMap<Integer, String[]>();
		for ( int i = 0; i < nNode; i++ ) {
			int numValue = instances.attribute(i).numValues();
			String[] values = new String[numValue];
			for ( int j = 0; j < numValue; j++ ) {
				values[j] = instances.attribute(i).value(j);
			}
			// add into...
			nodeAndValue.put(i, values);
		}

		// compute mutual information
		for ( int i = 0; i < nNode; i++ ) {
			for ( int j = i; j < nNode; j++ ) {
				mutualInfo[i][j] = computeMutualInfo(
						i, 
						nodeAndValue.get(i),
						j, 
						nodeAndValue.get(j),
						instances);
				mutualInfo[j][i] = mutualInfo[i][j];
			}
		}
	}
	
	
	public void makeCandidateSet(int iMethod) {
//		if ( this.mutualInfo == null ||
//			 this.chiSquare == null ) {
//			System.out.println("mutual or chi-square info is null");
//			System.exit(0);
//		}
		double[][] info = null;
		if ( iMethod == 0 ) {
			info = this.mutualInfo;
		} else if ( iMethod == 1 ) {
			info = this.chiSquare;
		} else {
			System.out.println("->Undefined method for making candidate set.");
			System.exit(0);
		}
		
		this.candidateSet = new int[nNode][nCandidate];
		double[] tempMIs = new double[nCandidate];
		for ( int i = 0; i < nNode; i++ ) {
			for ( int j = 0; j < nCandidate; j++ ) {
				candidateSet[i][j] = -1;
			}
		}
		for ( int i = 0; i < nNode; i++ ) {
			for ( int kk = 0; kk < nCandidate; kk++ ) { tempMIs[kk] = -0.001; }
			for ( int j = 0; j < nNode; j++ ) {
				if ( i != j ) {
					double tempMI = info[i][j];
					int iFlag = -1;
					for ( int mm = 0; mm < nCandidate; mm++ ) {
						if ( tempMI > tempMIs[mm] ) { iFlag = mm; break; }
					}
					if ( iFlag >= 0 ) {
						for ( int nn = nCandidate - 1; nn > iFlag; nn-- ) {
							candidateSet[i][nn] = candidateSet[i][nn - 1];
							tempMIs[nn] = tempMIs[nn - 1];
						}
						candidateSet[i][iFlag] = j;
						tempMIs[iFlag] = tempMI;
					}
				}
			}
		}
	}
	
	
	public void designCurricula(int iMethod) {
		double initMaxSumMI = 0;
		if ( !bCLReverse ) {
			initMaxSumMI = -0.1;
		} else {
			initMaxSumMI = Double.MAX_VALUE;
		}
		
		double[][] info = null;
		if ( iMethod == 0 ) {
			info = this.mutualInfo;
		} else if ( iMethod == 1 ) { // it shouldn't be used in this way
			info = this.chiSquare;
		} else {
			System.out.println("->Undefined method for making candidate set.");
			System.exit(0);
		}
		// determine the first node according to the maximum mutualInfo
		int iNodeOfMaxMI = 0;
		double tempSumMI = 0, maxSumMI = initMaxSumMI;	
		ArrayList<Integer> remainList = new ArrayList<Integer>();
		ArrayList<Integer> curricula = new ArrayList<Integer>();
		// determine the first node according to the maximum mutualInfo
		for ( int i = 0; i < nNode; i++ ) {
			tempSumMI = 0;
			for ( int j = 0; j < nNode; j++ ) {
				if ( i != j ) { tempSumMI += info[i][j]; }
			}
			if ( AUtils.compareAandB( tempSumMI, maxSumMI, !bCLReverse) ) {
				maxSumMI = tempSumMI;
				iNodeOfMaxMI = i;
			}
			
		}
		// initialize the remainList & curricula
		curricula.add(new Integer(iNodeOfMaxMI));
		for ( int i = 0; i < nNode; i++ ) {
			if ( i != iNodeOfMaxMI ) { 
				remainList.add(new Integer(i)); 
			}
		}
		
		int count = 0;
		Integer[] inds = new Integer[nNode - 1];
		double[] mis = new double[nNode - 1];
		// add nodes into CL list
		while ( !remainList.isEmpty() ) {
			count = 0;
			iNodeOfMaxMI = 0;
			maxSumMI = initMaxSumMI;
			for ( Integer node : remainList ) {
				tempSumMI = 0;
				for ( Integer nodeInc : curricula ) {
					tempSumMI += info[node][nodeInc.intValue()];
				}
				inds[count] = node.intValue();
				mis[count] = tempSumMI;
				count++;
				if ( AUtils.compareAandB( tempSumMI, maxSumMI, !bCLReverse) ) {
					maxSumMI = tempSumMI;
					iNodeOfMaxMI = node;
				}	
			}
			// should figure out differences between int & Integer on the numerical precision
			curricula.add(new Integer(iNodeOfMaxMI));
			remainList.remove(new Integer(iNodeOfMaxMI));
		}
		int index = 0;
		for ( Integer node : curricula ) {
			this.listOfCL[index] = node.intValue();
			index++;
		}
		printCurricula(curricula);
	}
	
	
	public double computeChiquare(
			int node1, 
			String[] node1Values,
			int node2, 
			String[] node2Values,
			Instances instances) {
		Map<String,Integer> node1ValueCount = new HashMap<String,Integer>();
		Map<String,Integer> node2ValueCount = new HashMap<String,Integer>();
		Map<String,Integer> jointValueCount = new HashMap<String,Integer>();
		
		for ( int i = 0; i < node1Values.length; i++ ) {
			node1ValueCount.put(node1Values[i], 0);
		}	
		for ( int i = 0; i < node2Values.length; i++) {
			node2ValueCount.put(node2Values[i], 0);
		}	
		for ( int i = 0; i < node1Values.length; i++ ) {
			for ( int j = 0; j < node2Values.length; j++ ) {
				String pattern = node1Values[i] + "_" + node2Values[j];
				jointValueCount.put(pattern, 0);
			}
		}
		
		Instance obs;
		String x, y, xy;
		int samplesize = instances.numInstances();
		// go through the data set to compute frequencies for X, Y, XY
		for ( int i = 0; i < samplesize; i++ ) {
			obs = instances.instance(i);
			x = obs.stringValue(node1);
			y = obs.stringValue(node2);
			xy = x + "_" + y;
			
			if ( node1ValueCount.containsKey(x) ) {
				node1ValueCount.put(x, node1ValueCount.get(x) + 1);
			}	
			if ( node2ValueCount.containsKey(y) ) {
				node2ValueCount.put(y, node2ValueCount.get(y) + 1);
			}	
			if( jointValueCount.containsKey(xy) ) {
				jointValueCount.put(xy, jointValueCount.get(xy) + 1);
			}	
		}
		
		double chiSquare = 0;
		for ( int i = 0; i < node1Values.length; i++ ) {
			for ( int j = 0; j < node2Values.length; j++ ) {
				x = node1Values[i];
				y = node2Values[j];
				xy = x + "_" + y;
				long Oi = node1ValueCount.get(x);
				long Oj = node2ValueCount.get(y);
				long Oij = jointValueCount.get(xy);
				double Eij = Oi * Oj / (double)samplesize;
				// why? not sufficient statistic
				if ( Oi == 0 || Oj == 0 ) {
					chiSquare = chiSquare + 0;
				} else {
					chiSquare = chiSquare + Math.pow((Oij - Eij), 2) / Eij;
				}
			}
		}
		return chiSquare;
	}
	

	public double computeMutualInfo(
			int node1, 
			String[] node1Values,
			int node2, 
			String[] node2Values,
			Instances instances) {
		Map<String,Integer> node1ValueCount = new HashMap<String,Integer>();
		Map<String,Integer> node2ValueCount = new HashMap<String,Integer>();
		Map<String,Integer> jointValueCount = new HashMap<String,Integer>();
		for ( int i = 0; i < node1Values.length; i++ ) {
			node1ValueCount.put(node1Values[i], 0);
		}	
		for ( int i = 0; i < node2Values.length; i++) {
			node2ValueCount.put(node2Values[i], 0);
		}	
		for ( int i = 0; i < node1Values.length; i++ ) {
			for ( int j = 0; j < node2Values.length; j++ ) {
				String pattern = node1Values[i] + "_" + node2Values[j];
				jointValueCount.put(pattern, 0);
			}
		}
		Instance obs;
		String x, y, xy;
		double H_X = 0, H_Y = 0, H_XY = 0;
		int samplesize = instances.numInstances();
		// go through the data set to compute frequencies for X, Y, XY
		for ( int i = 0; i < samplesize; i++ ) {
			obs = instances.instance(i);
			x = obs.stringValue(node1);
			y = obs.stringValue(node2);
			xy = x + "_" + y;
			if ( node1ValueCount.containsKey(x) ) {
				node1ValueCount.put(x, node1ValueCount.get(x) + 1);
			}	
			if ( node2ValueCount.containsKey(y) ) {
				node2ValueCount.put(y, node2ValueCount.get(y) + 1);
			}	
			if( jointValueCount.containsKey(xy) ) {
				jointValueCount.put(xy, jointValueCount.get(xy) + 1);
			}	
		}
		// compute entropy for X
		Integer[] xValueCount = 
				node1ValueCount.values().toArray(new Integer[0]);
		for ( int i = 0; i < xValueCount.length; i++ ) {
			int freq = xValueCount[i];
			if ( freq == 0 ) {
				H_X = H_X - 0;
			} else {
				H_X = H_X - ( ((double)freq) / samplesize ) * 
					  Math.log( ((double)freq) / samplesize ); 
			}
		}
		// reference: https://en.wikipedia.org/wiki/Mutual_information
		if ( node1 == node2 ) { return H_X; }
		// compute entropy for Y
		Integer[] yValueCount = 
				node2ValueCount.values().toArray(new Integer[0]);
		for ( int i = 0; i < yValueCount.length; i++ ) {
			int freq = yValueCount[i];
			if ( freq == 0 ){
				H_Y = H_Y - 0;
			} else {
				H_Y = H_Y - ( ((double)freq) / samplesize ) * 
					  Math.log( ((double)freq) / samplesize ); 
			}
		}
		// compute entropy for (X,Y)
		Integer[] xyValueCount = 
				jointValueCount.values().toArray(new Integer[0]);
		for ( int i = 0; i < xyValueCount.length; i++ ) {
			int freq = xyValueCount[i];
			if ( freq == 0 ) {
				continue;
			} else {
				H_XY = H_XY - ( ((double)freq) / samplesize ) *
					   Math.log( ((double)freq) / samplesize ); 
			}
		}
		return (H_X + H_Y - H_XY);
	} //computeMutualInformation
	
	
	protected void printCurricula(ArrayList<Integer> curricula) {
		System.out.println("\nCurriculum List: ");
		for ( int i = 0; i < curricula.size(); i++ ) {
			System.out.print(curricula.get(i) + " ");
		}
		System.out.println();
	}
	
	
	protected void print2DMatrix(int[][] matrix, int dim) {
		System.out.println("CandidateSet: ");
		for ( int i = 0; i < nNode; i++ ) {
			for ( int j = 0; j < dim; j++ ) {
				System.out.print(candidateSet[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	
	/**
	 * @param matrix 
	 * 		a 2-D square matrix
	 * @param dim 
	 * 		number of the row or column
	 */
	protected void print2DMatrix(double[][] matrix, int dim) {
		System.out.println("Info Matrix: ");
		for ( int i = 0; i < dim; i++ ) {
			for ( int j = 0; j < dim; j++ ) {
				System.out.print(String.format("%1$.2f ", matrix[i][j]));
			}
			System.out.println();
		}
		System.out.println();
	}

}
