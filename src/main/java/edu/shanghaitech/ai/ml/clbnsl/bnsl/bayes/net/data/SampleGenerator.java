package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.BayesNetNode;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.BayesNetStruct;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;

/**
 * Generate samples sampled according to the distribution associated with
 * the BN structure stored in the BIF file.
 * 
 * @author Yanpeng Zhao
 * 7/5/2015
 */
public class SampleGenerator extends Generator {
	private String bifPath = null;
	private String logPath = null;
	private String name = null;
	
	private Random random = null;
	private BayesNetStruct bns = null;
	private Instances instances = null;

	private int iScale2Write = 0; 
	
	public SampleGenerator() {}

	
	public SampleGenerator(
			int seed, 
			int iScale, 
			int nScale, 
			int iTestDS, 
			int nTestDS, 
			String basePath) {
		this.random = new Random(seed);
		this.instances = new Instances();
		initConfig(seed, iScale, nScale, iTestDS, nTestDS, basePath);
	}
	
	
	public void setIScale2Write(int iScale2Write) {
		this.iScale2Write = iScale2Write;
	}
	
	
	public boolean isContinue(int iScale) {
		return iScale < iScale2Write ? true : false;
	}
	
	protected void updateConfig(String name) throws Exception {
		instances.clear();
		
		this.name = name;
		this.logPath = netPath + name + "/" + name;
		this.bifPath = netPath + name + "/" + name + AUtils.FSUFFIX_BIF;
		this.bns = new BayesNetStruct(logPath, bifPath, AUtils.STRUCT_FROM_BIF);
	}
	
	
	/**************************************************************************
	 * 0st Phase: generate header of .arrf file
	 **************************************************************************/
	public boolean genArrFileHeader() {
		try {
			StringBuffer header = new StringBuffer();
			StringBuffer count = new StringBuffer();
	    	String arrHeaderFile = null;
	    	String countFilePath = null;
	    	
	    	BayesNetNode[] nodes = bns.BNNodes;

			header.append("@relation " + name + "\n\n");
			for ( int j = 0; j < nodes.length; j++ ) {
				header.append("@attribute " + nodes[j].name + " {");
/************************************************************************************/		
				count.append(nodes[j].getValSet().length + " ");
/************************************************************************************/		
				int k = 0;
				for ( ; k < nodes[j].getValSet().length - 1; k++ ) {
					header.append(k + ",");
				}
				header.append(k + "}\n");
			}
			header.append("\n@data");

			arrHeaderFile = netPath + name + "/" + name + "_" + AUtils.FSUFFIX_HEADER;
	   		writer = new FileWriter(arrHeaderFile);
			writer.write(header.toString());
			writer.close();
/************************************************************************************/					
			countFilePath = resPath + name + "/";
			if ( !AUtils.initDirs(countFilePath) ) {
    			System.out.println("Dir Err: " + countFilePath);
    		}
			writer = new FileWriter(countFilePath + name + "_" + AUtils.FSUFFIX_COUNT);
			writer.write(count.toString());
			writer.close();
/************************************************************************************/		
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	/**************************************************************************
	 * 1st Phase: generate GT (Ground Truth) structure as adjacent list
	 **************************************************************************/
	public boolean genGTStructure() {
		try {
			StringBuffer gt = new StringBuffer();
	    	String gtFile = null;
	    	
	    	int[] parentSet = null;
	    	BayesNetNode[] nodes = bns.BNNodes;

			gt.append(nodes.length + "\n");
			for ( int j = 0; j < nodes.length; j++ ) {
				
				int k = 0;
				parentSet = nodes[j].getParSet();
				gt.append(j + " " + parentSet.length + " ");
				for ( ; k < parentSet.length; k++ ) {
					gt.append(parentSet[k] + " ");
				}
				gt.append("\n");
			}

			gtFile = netPath + name + "/" + name + "_" + AUtils.FSUFFIX_GT;
	   		writer = new FileWriter(gtFile);
			writer.write(gt.toString());
			writer.close();

			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	/**************************************************************************
	 * 2st Phase: sample from the distribution associated with Bayesian Network
	 **************************************************************************/
	public boolean genSample() {
		try {
			String basePath = null;
			String dataFile = null;
			basePath = netPath + name + "/seed" + String.valueOf(seed) + "/";
    		// make directory
    		if ( !AUtils.initDirs(basePath) ) {
    			System.out.println("Dir Err: " + basePath);
    		}
    		
    		int preScale = 0;
    		for ( int j = 0; j < scales.length; j++ ) {
    			if ( isContinue(j) ) { continue; }
        		sample((scales[j] - preScale));
        		preScale = scales[j];
    			dataFile = basePath + String.valueOf(scales[j]) + ".txt";
        		writer = new FileWriter(dataFile);
				writer.write(instances.toString());
				writer.close();
        	}
	    	return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	public void sample(int nSample) {
		int[] parSet = null;
		int iCPD = 0, iNode, iPar, iValue;
		int[] orderedList = bns.topologicalSort();
		// prinTopological(orderedList); // @debug
		double prandom;
		double[][] cpd = null;
		BayesNetNode[] nodes = bns.BNNodes;
		for ( int i = 0; i < nSample; i++ ) {
			Instance instance = new Instance(bns.nNode);
			for ( int j = 0; j < bns.nNode; j++ ) {
				iCPD = 0;
				iNode = orderedList[j];
				parSet = nodes[iNode].getParSet();
				for ( int k = 0; k < parSet.length; k++ ) {
					iPar = parSet[k];
					iCPD = iCPD * nodes[iPar].getValSet().length + 
							instance.getData(iPar);
				}

				iValue = 0;
				cpd = nodes[iNode].getCPDs();
				prandom = random.nextInt(1000) / 1000.0f;
				while ( prandom > cpd[iCPD][iValue] && 
						iValue < nodes[iNode].getValSet().length ) {
					prandom -= cpd[iCPD][iValue];
					iValue++;
				}
				instance.setData(iNode, iValue);
			}
			instances.add(instance);
		}
	}
	
	
	/**************************************************************************
	 * 3st Phase: combine generated header and samples into .arff file
	 **************************************************************************/
    public boolean genArfFile() {
    	try {
    		String pathOfHeader = netPath + name + "/" + name + 
    				"_" + AUtils.FSUFFIX_HEADER;
    		StringBuffer header = new StringBuffer();
    		reader = new BufferedReader(
    				new FileReader(new File(pathOfHeader)));
    		
    		// @debug
    		System.out.println("->Processing " + name + "...");
    		String line = "";
    		while ( (line = reader.readLine()) != null ) {
    			header.append(line + "\n");
    		}
    		reader.close();
    		
    		for ( int j = 0; j < scales.length; j++ ) {
    			if ( isContinue(j) ) { continue; }
        		String scale = String.valueOf(scales[j]);
        		String pathOfData = netPath + 
        				name + "/seed" + String.valueOf(seed) + "/" + 
        				scale + ".txt";
        		String pathOfArff = netPath + 
        				name + "/seed" + String.valueOf(seed) + "/";
        		String pathOfMMHC = resPath + name + "/seed" + 
        				String.valueOf(seed) + "/" + scale + "/";
        		
        		// sensible operation, make sure that directory exists
        		if ( !AUtils.initDirs(pathOfArff) ) { 
        			System.out.println("Dir Err: " + pathOfArff);
        		}
        		if ( !AUtils.initDirs(pathOfMMHC) ) {
        			System.out.println("Dir Err: " + pathOfMMHC);
        		}
        		
        		StringBuffer data = new StringBuffer();
        		StringBuffer arff = new StringBuffer();
        		StringBuffer mmhc = new StringBuffer();
        		
        		writer = new FileWriter(
        				(pathOfArff + name + scale + ".arff"));
        		reader = new BufferedReader
        				(new FileReader(new File(pathOfData)));
        		
        		while ( (line = reader.readLine()) != null ) {
        			// remove '(' & ')'
        			line = line.trim().substring(1, line.length() - 1);
/************************************************************************************/		
        			mmhc.append(line + "\n");
/************************************************************************************/		
        			// use ',' instead of ' '
        			line = line.trim().replace(" ", ",").trim();
        			data.append(line + "\n");
        		}
        		
        		arff.append(header);
        		arff.append(data);
        		writer.write(arff.toString());
        		
        		reader.close();
        		writer.close();
/************************************************************************************/		        		
        		writer = new FileWriter(
        				(pathOfMMHC + name + scale + AUtils.FSUFFIX_MMHC));
        		writer.write(mmhc.toString());
        		writer.close();
/************************************************************************************/		
        	}
        	// @debug
        	System.out.println("->Processing end, good luck!");
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
    }
	
	
	/**************************************************************************
	 * 4st Phase: migrate generated samples or backup
	 **************************************************************************/
    /**
     * Move .arff files to 'res' directory for the use of  experiments.
     * 
     * @param isCopy 
     * 		true: copy; false: move
     * @throws Exception 
     * 
     */
	public boolean migrateArfFile(boolean isCopy) {
		try {
    		// @debug, migrate .arff files
    		System.out.println("->Migrating " + name + "...");
    		for ( int j = 0; j < scales.length; j++ ) {
    			
    			if ( isContinue(j) ) { continue; }
    			
    			String scale = String.valueOf(scales[j]);
    			String orgPath = netPath + 
        				name + "/seed" + String.valueOf(seed) + 
        				"/" + name + scale + ".arff";
    			String desPath = resPath + 
        				name + "/seed" + String.valueOf(seed) + 
        				"/" + scale + "/";
    			// make directory
    			if ( !AUtils.initDirs(desPath) ) {
    				System.out.println("Dir Err: " + desPath);
    			}
    			
    			// migrate
    			desPath = desPath + name + scale + ".arff";
    			AUtils.migrateFile(orgPath, desPath, isCopy);
    		}
    		
    		// migrate ground truth files
    		String orgPath = netPath + name + "/" + name + "_" + AUtils.FSUFFIX_GT;
			String desPath = resPath + name + "/" + name + "_" + AUtils.FSUFFIX_GT;
			AUtils.migrateFile(orgPath, desPath, isCopy);

			// migrate header
			orgPath = netPath + name + "/" + name + "_" + AUtils.FSUFFIX_HEADER;
			desPath = resPath + name + "/" + name + "_" + AUtils.FSUFFIX_HEADER;
			AUtils.migrateFile(orgPath, desPath, isCopy);

	    	// @debug
	    	System.out.println("->Migrating end, ready for experiments!");
	    	return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
    
	
	/**************************************************************************
	 * entrance of the routine
	 **************************************************************************/
	protected void makeTrainingSetReady(boolean isCopy) {
		
		System.out.println("->Generate headers...");
		if ( !genArrFileHeader() ) {
			System.out.println("Err: Generate headers");
			System.exit(0);
		}
		
		System.out.println("->Generate GT structures...");
		if ( !genGTStructure() ) {
			System.out.println("Err: Generate GT structures");
			System.exit(0);
		}
		
		System.out.println("->Generate samples...");
		if ( !genSample() ) {
			System.out.println("Err: Generate samples");
			System.exit(0);
		}
		
		System.out.println("->Generate arffiles...");
		if ( !genArfFile() ) {
			System.out.println("Err: Generate arffiles");
			System.exit(0);
		}
		
		System.out.println("->Migrate arffiles...");
		if ( !migrateArfFile(isCopy) ) {
			System.out.println("Err: Migrate arffiles");
			System.exit(0);
		}

		System.out.println("->Training set " + name.toUpperCase() + " is ready");
	}
	
	
	protected void prinTopological(int[] order) {
		System.out.println("Topological Order:");
		for ( int i = 0; i < order.length; i++ ) {
			System.out.print(order[i] + " ");
			if ( i % 10 == 0 ) { System.out.println(); }
		}
		System.out.println();
	}

	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		int seed = 0;
    	int iScale = 0;
    	int nScale = 8;
    	int iTestDS = 9;
    	int nTestDS = 12;
    	String basePath = "E:/NewData/";
    	
    	SampleGenerator sg = new SampleGenerator
    			(seed, iScale, nScale, iTestDS, nTestDS, basePath);
    	sg.setIScale2Write(0);
    	
    	for ( int i = iTestDS; i < nTestDS; i++ ) {
//    		sg.updateConfig(AUtils.nameOfDS[i]);
//    		sg.makeTrainingSetReady(true);
//    		sg.genArfFile();
    	}
	}

}
