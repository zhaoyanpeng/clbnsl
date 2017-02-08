package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data;

import java.io.File;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.curricula.Curricula;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.curricula.MMPC;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.converters.ArffLoader;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * @author Yanpeng Zhao
 * 5/18/2015
 */
public class PCSetGenerator {

	private String arrfFilePath;
	private Instances instances;
	private ArffLoader arffLoader;
	
	
	public PCSetGenerator() {
		this.arffLoader = new ArffLoader();
	}
	

	public PCSetGenerator(String arrfFilePath) {
		this.arrfFilePath = arrfFilePath;
		this.arffLoader = new ArffLoader();
		// make data ready
		initInstances();
	}
	
	
	protected void updateInstances(String arrfFilePath) {
		this.arrfFilePath = arrfFilePath;
		initInstances();
	}
	
	
	protected void initInstances() {
		try {
			File file = new File(this.arrfFilePath);
			arffLoader.setFile(file);
			this.instances = arffLoader.getDataSet();
			// @debug
			System.out.println("Instances are ready..Good luck!");
			// sets must be done before everything 
			this.instances.setClassIndex(this.instances.numAttributes() - 1);
		} catch ( Exception e ) {
			e.printStackTrace();
		}
	}
	

    public void generatePCsets(
			int seed, 
			int[] scales, 
			boolean bDebug,
			String basePath) throws Exception {
    	int[] iScales = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    	int[] nScales = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    	String[] names = AUtils.nameOfDS;

    	int iTest = 7;
    	int nTest = 8;
    	
    	MMPC pcset = null;
    	Curricula curricula = null;
    	double[][] mutualInfo = null;
    	String pcsetid = "_BA_";
    	
    	for ( int ii = iTest; ii < nTest; ii++ ) {
    		int iScale = iScales[ii];
    		int nScale = nScales[ii];
    		String nameOfDS = names[ii];
			for ( int kk = iScale; kk < nScale; kk++ ) {
				String scale = String.valueOf((scales[kk]));
				String arrfFilePath = basePath + "res/" + nameOfDS + "/seed" + String.valueOf(seed) + "/" + scale + 
						"/" + nameOfDS + scale + AUtils.FSUFFIX_ARFF;
				String pcsetFilePath = basePath + "res/" + nameOfDS + "/seed" + String.valueOf(seed) + "/" + scale + 
						"/" + nameOfDS + pcsetid + AUtils.FSUFFIX_PCSET;

				this.updateInstances(arrfFilePath);
				
				// @debug
				System.out.println("->" + nameOfDS + " scale " + scale);
				
				curricula = new Curricula(false, instances, 0);
				mutualInfo = curricula.getMutualInfo();
				
				if ( mutualInfo != null ) {
					pcset = new MMPC(instances, mutualInfo);
					pcset.generatePCsets(pcsetFilePath);
//					pcset.getPCset(25);
				} else {
					System.out.println("MutualInfo is NULL.");
					System.exit(0);
				}
			}
    	}
    }


	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		int seed = 1;
		int nScale = 8;
		boolean bDebug = false;
		int[] scales = AUtils.getScaleSequence(0, nScale);
		String basePath = "E:/NewData/";
		PCSetGenerator WLSet = new PCSetGenerator();
		long startTime = System.currentTimeMillis();
		
		for ( int i = 0; i < 5; i++ ) {
			seed = i;
			WLSet.generatePCsets(seed, scales, bDebug, basePath);
		}
		
 		long endTime = System.currentTimeMillis();
 		double totalTime = (double)(endTime - startTime) / 1000;
 		System.out.println("Time on PC generating: " + totalTime + "s");
	}

}
