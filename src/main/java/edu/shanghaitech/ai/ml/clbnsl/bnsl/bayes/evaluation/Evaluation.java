package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.estimator.CLBayesNetEstimator;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.converters.ArffLoader;

/**
 * @author Yanpeng Zhao
 * 3/25/2015
 */
public class Evaluation {
	
	private final static String[] names = AUtils.nameOfDS;
	protected int seed = 0;
	
	public void evaMMHC() throws Exception {
		int ess = (int) AUtils.ESS;
		int iScale = 1;
		int nScale = 7;
		int iTest = 0;
    	int nTest = 10;
//    	int seed = 4;
		int[] scales = AUtils.getScaleSequence(0, nScale);
		
		String name = "";
		String gtPath = "";
		String lnPath = "";
		String hdPath = "";
		String tsPath = "";
		
		String basePath = "E:/NewData/";
		
		StringBuffer sbeva = new StringBuffer();
		StringBuffer sbshd = new StringBuffer();
		
		String evafile = "";
		String shdfile = "";
		
		String algid = "_MMHC_";
		
		for ( int i = iTest; i < nTest; i++ ) {
			name = names[i];
			gtPath = basePath + "res/" + name + "/" + name + "_" + AUtils.FSUFFIX_GT;
			hdPath = basePath + "net/" + name + "/" + name + "_" + AUtils.FSUFFIX_HEADER;
			tsPath = basePath + "res/" + name + "/seed" + String.valueOf(9) + "/" + String.valueOf(5000) + "/" + 
					name + String.valueOf(5000) + AUtils.FSUFFIX_ARFF;
			// @debug
			System.out.println("GTPath: " + gtPath + "\nHDPath: " + hdPath + "\nTSPath: " + tsPath);
			
			evafile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + name + algid + String.valueOf(ess) + 
					"_" + AUtils.FSUFFIX_EVA;
			shdfile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" +name + algid + String.valueOf(ess) + 
					"_" + AUtils.FSUFFIX_SHD;
			FileWriter eva = new FileWriter(evafile);
			FileWriter shd = new FileWriter(shdfile);
			Instances testSet = getTestSet(tsPath);
			
			sbeva.delete(0, sbeva.length());
			sbshd.delete(0, sbshd.length());
			
			// initialize ground truth
			CLBayesNetEstimator ec = new CLBayesNetEstimator(
	    			hdPath, gtPath, ess, true, AUtils.STRUCT_FROM_ALIST); 
	    	
	    	ec.evaluate(testSet); // evaluate ground truth
	    	sbeva.append(ec.getBDeuScore() + " " + ec.getBICScore() + " " + 
					ec.getLLScore() + " " + ec.getKLScore() + "\n");
	    	
			for ( int j = iScale; j < nScale; j++ ) {
				lnPath = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" +
						String.valueOf(scales[j]) + "/" + name + algid + String.valueOf(ess) + 
						"_" + AUtils.FSUFFIX_DAG;
				// @debug
				System.out.println("LearnNet: " + lnPath);
				// initialize learned net
	    		ec.updateLearnednet(hdPath, lnPath, AUtils.STRUCT_FROM_AMATRIX); 
	    		ec.evaluate(testSet); // evaluate learned net
	    		
	    		sbeva.append(ec.getBDeuScore() + " " + ec.getBICScore() + " " + 
        				ec.getLLScore() + " " + ec.getKLScore() + "\n");
	    		
	    		sbshd.append("M: " + ec.getNMissingArc() + " " + 
        				"E: " + ec.getNExtraArc() + " " + 
        				"R: " + ec.getNReversedArc(true) + " " +
        				"RC: " + ec.getNReversedArc(false) + "\n");
	    		
			}
			sbeva.append("\n");
			sbshd.append("\n");
			eva.write(sbeva.toString());
			shd.write(sbshd.toString());
			eva.close();
			shd.close();
		}
	}
	
	
	/**
	 * Collect the optimal score and time from <tt>../name/seed#/< scale >/name< * >ST.TXT</tt> into 
	 * <tt>../name/seed#/name< * >EVA.TXT</tt>, and collect all the steps to which the optimal structures(BDeu)
	 * correspond.
 	 * 
	 * @throws Exception
	 */
	public void evaCLDistribution() throws Exception {
		int ess = (int) AUtils.ESS_CL;
		int iScale = 1;
		int nScale = 7;
		int iTest = 0;
    	int nTest = 10;
    	int iSeed = 0;
    	int nSeed = 5;
		int[] scales = AUtils.getScaleSequence(0, nScale);
		
		String name = "";
		String gtPath = "";
		String lnPath = "";
		String hdPath = "";
		String tsPath = "";
		
		String basePath = "E:/NewData/"; // root 
		
		StringBuffer sbeva = new StringBuffer();
		StringBuffer step = new StringBuffer();
		StringBuffer time = new StringBuffer();
		
		String evafile = "";
		String shdfile = "";
		String stfile  = "";
		String timefile = "";
		String stepfile = "";
		String line    = "";
		String iStep   = "";
		String iTime   = "";
		
		String algid = "_20160311_";     // flag
		String prefix = "";
		
		FileWriter writer = null;
		BufferedReader reader = null;
		
		Pattern pattern = Pattern.compile(AUtils.ST_PATTERN);
		Matcher matcher = null;
		
		for ( int i = iTest; i < nTest; i++ ) {
			name = names[i];
			gtPath = basePath + "res/" + name + "/" + name + "_" + AUtils.FSUFFIX_GT;
			hdPath = basePath + "net/" + name + "/" + name + "_" + AUtils.FSUFFIX_HEADER;
			tsPath = basePath + "res/" + name + "/seed" + String.valueOf(9) + "/" + String.valueOf(5000) + "/" + 
					name + String.valueOf(5000) + AUtils.FSUFFIX_ARFF;

			System.out.println("GTPath: " + gtPath + "\nHDPath: " + hdPath + "\nTSPath: " + tsPath); // @debug
			
			Instances testSet = getTestSet(tsPath);

			step.delete(0, step.length());
			time.delete(0, time.length());
			
			// initialize ground truth
			CLBayesNetEstimator ec = new CLBayesNetEstimator(
	    			hdPath, gtPath, AUtils.ESS, true, AUtils.STRUCT_FROM_ALIST); 
	    	
	    	for ( int k = iSeed; k < nSeed; k++ ) {
	    		
		    	seed = k;
		    	sbeva.delete(0, sbeva.length());
				for ( int j = iScale; j < nScale; j++ ) {

					stfile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" +
							String.valueOf(scales[j]) + "/" + name + algid + String.valueOf(ess) + 
							"_" + AUtils.FSUFFIX_ST;
					reader = new BufferedReader(
							new FileReader(new File(stfile)));
					matcher = pattern.matcher(reader.readLine());
					
					if ( matcher.find() ) { // find the step to which the optimal BDeu corresponds
						iTime = matcher.group(2).trim();
						iStep = matcher.group(3).trim();
						System.out.println(iStep + " finds best results consuming " + iTime + "s");
						
						time.append(iTime + " ");
						step.append(iStep + " ");
					} else { System.out.println("Err: ST file " + line); System.exit(0); }	
					
					lnPath = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" +
							String.valueOf(scales[j]) + "/" + name + "_step_" + iStep + algid + 
							String.valueOf(ess) + "_" + AUtils.FSUFFIX_LN; 
					
					System.out.println("LearnNet: " + lnPath); // @debug
					
		    		ec.updateLearnednet(hdPath, lnPath, AUtils.STRUCT_FROM_ALIST); // initialize learned net
		    		ec.evaluate(testSet); // evaluate learned net
		    		
		    		sbeva.append(ec.getBDeuScore() + " " + ec.getBICScore() + " " + 
	        				ec.getLLScore() + " " + ec.getKLScore() + "\n");       // record statistics 
				}
				sbeva.append("\n");
				time.append("\n");
				step.append("\n");
				
				evafile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + name + prefix + algid + 
						String.valueOf(ess) + "_" + AUtils.FSUFFIX_EVA;
				writer = new FileWriter(evafile);
				writer.write(sbeva.toString());
				writer.close();
	    	}
	    	// and record the steps to which the optimal structures(BDeu) correspond
			stepfile = basePath + "res/" + name + "/" + name + prefix + algid + String.valueOf(ess) + "_" + AUtils.FSUFFIX_BESTEP;
			writer = new FileWriter(stepfile);
			writer.write(step.toString());
			writer.close();
			// running time consumed to obtain the optimal structure
			timefile = basePath + "res/" + name + "/" + name + prefix + algid + String.valueOf(ess) + "_" + AUtils.FSUFFIX_RUNTIME;
			writer = new FileWriter(timefile);
			writer.write(time.toString());
			writer.close();
			System.out.println("Saving eva: " + evafile + "\ntime: " + timefile + "\nstep: " + stepfile); // @debug
		}
	}
	
	
	
	/**
	 * Just make a copy of results.
	 */
	public void backupCL() throws Exception {
		int itsESS = (int) AUtils.ESS_CL;
		int iScale = 1;
		int nScale = 7;
		int iTest = 0;
    	int nTest = 1;
		int[] scales = AUtils.getScaleSequence(0, nScale);
		
		String name = "";
		String evaPath = "";
		String shdPath = "";

		StringBuffer sbeva = new StringBuffer();
		StringBuffer sbshd = new StringBuffer();
		
		String evafile = "", devafile = "";
		String shdfile = "", dshdfile = "";
		String line    = "", dalgid;
		
		String basePath = "E:/NewData/";
		
		String algid = "_CL0_";
		
		for ( int i = iTest; i < nTest; i++ ) {
			name = names[i];
			for ( int j = iScale; j < nScale; j++ ) { 
				evafile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + String.valueOf(scales[j]) + "/" + name + algid + 
						String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_ST;
				shdfile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + String.valueOf(scales[j]) + "/" + name + algid + 
						String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_SHD;
				
				dalgid = algid + "COPY_";;
				
				devafile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + String.valueOf(scales[j]) + "/" + name + dalgid + 
						String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_ST;
				dshdfile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + String.valueOf(scales[j]) + "/" + name + dalgid + 
						String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_SHD;
				
				AUtils.migrateFile(shdfile, dshdfile, true);
				AUtils.migrateFile(evafile, devafile, true);
				
				System.out.println("O->" + shdfile + "\n" + "D->" + dshdfile);
				System.out.println("O->" + evafile + "\n" + "D->" + devafile);
			}
		}
	}
	
	
	/**
	 * Collect the optimal SHD from <tt>../name/seed#/< scale >/name< * >SHD.TXT</tt> into 
	 * <tt>../name/seed#/name< * >SHD.TXT</tt>.
	 */
	public void evaCLSHD() throws Exception {
		int itsESS = (int) AUtils.ESS_CL;
		int iScale = 1;
		int nScale = 7;
		int iTest = 0;
    	int nTest = 10;
    	int iSeed = 0;
    	int nSeed = 5;
		int[] scales = AUtils.getScaleSequence(0, nScale);
		
		String name = "";
		String evaPath = "";
		String shdPath = "";

		StringBuffer sbeva = new StringBuffer();
		StringBuffer sbshd = new StringBuffer();
		
		String evafile = "";
		String shdfile = "";
		String line    = "";
		
		String basePath = "E:/NewData/"; // root
		
		String algid = "_20160311_";     // flag
		String dalgid = "";
		String suffix = "";
		String prefix = "";
		
		FileWriter writer = null;
		BufferedReader reader;
		
		// for every dataset
		for ( int i = iTest; i < nTest; i++ ) {
			name = names[i];
			// for every seed
			for ( int k = iSeed; k < nSeed; k++ ) {
		    	seed = k;
		    	sbshd.delete(0, sbshd.length());
		    	// for every scale
				for ( int j = iScale; j < nScale; j++ ) {
					
					shdPath = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + String.valueOf(scales[j]) + "/" + name + algid + 
							String.valueOf(itsESS) + "_" + suffix + AUtils.FSUFFIX_SHD;
					
					System.out.println("EVAPath: " + evaPath + "\nSHDPath: " + shdPath); // @debug
					
					reader = new BufferedReader(new FileReader(new File(shdPath)));
					while ( (line = reader.readLine()) != null ) {
						sbshd.append(line + "\n");
					} // record every single SHD from every data scale
					
					reader.close();
				} // end scale
				shdfile = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" + name + prefix + algid + 
				String.valueOf(itsESS) + "_" + suffix + AUtils.FSUFFIX_SHD;
				writer = new FileWriter(shdfile);
				writer.write(sbshd.toString());
				writer.close();
				System.out.println("Saving shd: " + shdfile);
			} // end seed
		} // end data-set
	}
	
	
	public void extractSHD() throws Exception {	
		int itsESS = (int) AUtils.ESS_CL;
		int iScale = 1;
		int nScale = 5;
		int iTest = 0;
    	int nTest = 0;
		int[] scales = AUtils.getScaleSequence(0, nScale);
    	
    	int m, e, r, rc;
    	
    	String algid = "_CL0_";
		String suffix = "";
		String prefix = "_CANDI_UA_SKIP_0.02";
		
		algid = "_MMHC_";
		prefix = "";
		itsESS = 10;
		
		String basePath = "E:/NewData/";
    	
    	String line = "";
    	String name = "";
    	String rpath = "";
    	String spath = "";
    	BufferedReader reader;
    	FileWriter writer; 
    	
    	Pattern pattern = Pattern.compile(AUtils.SHD_PATTERN);
		Matcher matcher;
		
		StringBuffer sb = new StringBuffer();
		
		for ( int i = iTest; i < nTest; i++ ) {
			name = names[i];
			rpath = basePath + "/res/" + name + "/seed" + String.valueOf(seed) + "/" + name + prefix + algid + String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_SHD;
			spath = basePath + "/res/" + name + "/seed" + String.valueOf(seed) + "/" + name + prefix + algid + String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_SHD_EXT;

			// @debug
			System.out.println("ReadPath: " + rpath + "\nSavePath: " + spath);
			
			sb.delete(0, sb.length());
			reader = new BufferedReader(new FileReader(new File(rpath)));
			while ( (line = reader.readLine()) != null ) {
				matcher = pattern.matcher(line);
				if ( matcher.find() ) {
					m = Integer.valueOf(matcher.group(1).trim());
					e = Integer.valueOf(matcher.group(2).trim());
					r = Integer.valueOf(matcher.group(3).trim());
					rc = Integer.valueOf(matcher.group(4).trim());
					sb.append(m + " " + e + " " +
							  r + " " + rc + " " + (m + e + rc));
				}
				sb.append("\n");
			}
			writer = new FileWriter(spath);
			writer.write(sb.toString());
			writer.close();
		}
	}
	
	
	public void evaMMHCTime() throws Exception {
		int ess = (int) AUtils.ESS;
		int iScale = 1;
		int nScale = 7;
		int iSeed = 0;
    	int nSeed = 5;
		int iTest = 0;
    	int nTest = 10;
		int[] scales = AUtils.getScaleSequence(0, nScale);
		
		String name = "";
		String lnPath = "";
		String savePath = "";
		String line   = "";
		
		String basePath = "E:/NewData/";
		
		StringBuffer sbtime = new StringBuffer();
		
		String evafile = "";
		String shdfile = "";
		
		String algid = "_MMHC_";
		String prefix = "_ACML2015_A";
		
		FileWriter writer = null;
		BufferedReader reader = null;
		
		for ( int i = iTest; i < nTest; i++ ) {
			name = names[i];
			sbtime.delete(0, sbtime.length());
	    	for ( int k = iSeed; k < nSeed; k++ ) {
	    		seed = k;
				for ( int j = iScale; j < nScale; j++ ) {
					lnPath = basePath + "res/" + name + "/seed" + String.valueOf(seed) + "/" +
							String.valueOf(scales[j]) + "/" + name + algid + String.valueOf(ess) + 
							"_" + AUtils.FSUFFIX_DAG;
					// @debug
					System.out.println("LearnNet: " + lnPath);

					reader = new BufferedReader(new FileReader(new File(lnPath)));
		    		line = reader.readLine();
		    		String[] res = line.split("@");
		    		if ( res.length == 2 ) {
		    			sbtime.append(res[1].trim() + " ");
		    		}
				}
				sbtime.append("\n");
	    	}
	    	savePath = basePath + "res/" + name + "/" + name + prefix + algid + String.valueOf(ess) + "_" + AUtils.FSUFFIX_RUNTIME;
	    	writer = new FileWriter(savePath);
	    	writer.write(sbtime.toString());
	    	writer.close();
	    	System.out.println("Saving: " + savePath);
		}
	}
	
	
	protected Instances getTestSet(String testSetPath) {
		try {
			ArffLoader arffLoader = new ArffLoader();
			File file = new File(testSetPath);
			arffLoader.setFile(file);
			return arffLoader.getDataSet();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}
	}
	
	
	public static void main(String[] args) throws Exception {
		Evaluation eva = new Evaluation();
//		eva.evaMMHCTime();
//		eva.evaMMHC();
		
/*		
		eva.evaCLSHD();
		eva.evaCLDistribution();
*/		
		
//		eva.backupCL();
//		eva.extractSHD();
	}

}
