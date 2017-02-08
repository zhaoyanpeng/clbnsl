package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * @author Yanpeng Zhao
 * 5/1/2015
 */
public class BayesNetNode {
	// description of the node
	public String name = null;
	// index of the node
	public int iName = -1;
	// values the node can take
	public String[] valSet = null;
	// containing indexes of the parents
	public int[] parSet = null;
	// containing indexes of the children
	public int[] chiSet = null;
	// number of different configurations of the parent set
	private int cardinalityOfParSet = 1;
	
	// sparse conditional probability table for memory saving
	private HashMap<Integer, Integer> sparsecpt = null;
	// 
	public double[][] cpt = null;
	
	
	public BayesNetNode() {}
	
	
	public void setName(String name) {
		this.name = name;
	}
	
	
	public String getName() {
		return this.name;
	}
	
	
	public void setIName(int iName) {
		this.iName = iName;
	}
	
	
	public void setValSet(String[] valSet) {
		this.valSet = valSet;
	}
	
	
	public String[] getValSet() {
		return this.valSet;
	}
	
	
	public void setParSet(int[] parSet) {
		this.parSet = parSet;
	}
	
	
	public int[] getParSet() {
		return this.parSet;
	}
	
	
	public boolean hasParent(int iNode) {
		for ( int i = 0; i < parSet.length; i++ ) {
			if ( parSet[i] == iNode ) {
				return true;
			}
		}
		return false;
	}
	
	
	public ArrayList<Integer> getParInfo() {
		ArrayList<Integer> list = new ArrayList<Integer>();
		
		list.add(iName);         // node id
		list.add(parSet.length); // number of parents
		
		for ( int i = 0; i < parSet.length; i++ ) {
			list.add(parSet[i]); // parents id
		}
		return list;
	}
	
	
	public void setCardinalityOfParSet(int cardinalityOfParSet) {
		this.cardinalityOfParSet = cardinalityOfParSet;
	}
	
	
	public int getCardinalityOfParSet() {
		return this.cardinalityOfParSet;
	}
	
	
	public void setSparseCPD(HashMap<Integer, Integer> statistic) {
		this.sparsecpt = statistic;
	}
	
	
	public HashMap<Integer, Integer> getSparseCPD() {
		return this.sparsecpt;
	}
	
	
	public void setCPD(int irow, int icol, double cpd) {
		if ( cpt == null ) {
			if ( valSet == null || parSet == null ) {
				System.out.println("Err: Please initialize valSet & parSet first.");
				System.exit(0);
			}
			cpt = new double[cardinalityOfParSet][valSet.length]; // initializing
		}
		if ( irow >= cardinalityOfParSet || icol >= valSet.length ) {
			System.out.println("Err: irow or icol exceeds boundary.\n" + 
					"irow: " + irow + " icol: " + icol + 
					"cardinal: " + cardinalityOfParSet + " valsize: " + valSet.length);
			System.exit(0);
		}
		
		this.cpt[irow][icol] = cpd;
	}
	
	
	public double getCPD(int irow, int icol) {
		return this.cpt[irow][icol];
	}
	
	
	public double[][] getCPDs() {
		return this.cpt;
	}
	
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("->" + name + "|" + valSet.length + "|" + cardinalityOfParSet + "|" );
		if ( parSet != null ) {
			for ( int i = 0; i < parSet.length; i++ ) {
				sb.append(parSet[i] + " ");
			}
			sb.append("\n");
		}
		
		if ( cpt != null ) {
			for ( int i = 0; i < cardinalityOfParSet; i++ ) {
				for ( int j = 0; j < valSet.length; j++ ) {
					sb.append(cpt[i][j] + " ");
				}
				sb.append("\n");
			}
		}
		return sb.toString();
	}
	
	
	/**
	 * @param bnNode
	 * 		the old node
	 * 
	 */
	public void copy(BayesNetNode bnNode) {
		this.name = bnNode.name;
		this.iName = bnNode.iName;
		this.cardinalityOfParSet = bnNode.cardinalityOfParSet;
		
		if ( bnNode.valSet != null ) {
			this.valSet = new String[bnNode.valSet.length];
			for ( int i = 0; i < valSet.length; i++ ) {
				this.valSet[i] = bnNode.valSet[i];
			}
		}
		
		if ( bnNode.parSet != null ) {
			this.parSet = new int[bnNode.parSet.length];
			for ( int i = 0; i < parSet.length; i++ ) {
				this.parSet[i] = bnNode.parSet[i];
			}
		}
		
		if ( bnNode.chiSet != null ) {
			this.chiSet = new int[bnNode.chiSet.length];
			for ( int i = 0; i < chiSet.length; i++ ) {
				this.chiSet[i] = bnNode.chiSet[i];
			}
		}
		
		if ( bnNode.cpt != null ) {
			this.cpt = new double[cardinalityOfParSet][valSet.length];
			for ( int i = 0; i < cardinalityOfParSet; i++ ) {
				for ( int j = 0; j < valSet.length; j++ ) {
					this.cpt[i][j] = bnNode.cpt[i][j];
				}
			}
		}
	}
	
}
