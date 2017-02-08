package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data;

import java.util.Vector;

/**
 * @author Yanpeng Zhao
 * 7/5/2015
 */
public class Instances {
	Vector<Instance> dataset = null;
	
	public Instances() {
		dataset = new Vector<Instance>();
	}
	
	
	public void add(Instance instance) {
		dataset.add(instance);
	}
	
	
	public Instance get(int index) {
		return dataset.get(index);
	}
	
	
	public void clear() {
		this.dataset.clear();
	}
	
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		for ( int i = 0; i < dataset.size(); i++ ) {
			sb.append(dataset.get(i) + "\n");
		}
		return sb.toString();
	}

}
