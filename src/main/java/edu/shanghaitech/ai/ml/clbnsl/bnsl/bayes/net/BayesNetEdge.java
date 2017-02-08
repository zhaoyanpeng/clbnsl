package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net;

/**
 * @author Yanpeng Zhao
 * 5/1/2015
 */
public class BayesNetEdge {
	
	public int x;  // starting vertex (parent)
	public int y;  // end vertex (child)
	
	public char status;  // compelled (1) or reversible (0)
	
	public BayesNetEdge() {
		this.x = -1;
		this.y = -1;
		this.status = 256;
	}
	
	public BayesNetEdge(int x, int y, char status) {
		this.x = x;
		this.y = y;
		this.status = status;
	}
	
	public void setX(int x) {
		this.x = x;
	}
	
	public void setY(int y) {
		this.y = y;
	}
	
	public void setStatus(char status) {
		this.status = status;
	}
}
