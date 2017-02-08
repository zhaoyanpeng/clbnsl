package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data;

/**
 * @author Yanpeng Zhao
 * 7/5/2015
 */
public class Instance {
	
	int nNode = -1;
	char[] data = null;
	
	public Instance(int nNode) {
		this.nNode = nNode;
		this.data = new char[nNode];
	}
	
	
	public void setData(int pos, int value) {
		this.data[pos] = (char)value;
	}
	
	
	public int getData(int pos) {
		return this.data[pos];
	}
	
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("( ");
		if ( data != null ) {
			for ( int i = 0; i < nNode; i++ ) {
				sb.append((int) data[i] + " ");
			}
		}
		sb.append(")");
		return sb.toString();
	}

}
