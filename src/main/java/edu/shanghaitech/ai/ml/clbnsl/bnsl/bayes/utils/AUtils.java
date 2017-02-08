package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Properties;

import edu.shanghaitech.ai.ml.clbnsl.weka.core.Attribute;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * Constant variable set.
 * 
 * @author Yanpeng Zhao
 * 4/7/2015
 */
public final class AUtils {
	// suffix of files
	public final static String FSUFFIX_BIF      = ".bif";         // bayesian net
	public final static String FSUFFIX_NET      = ".net";         // bayesian net
	public final static String FSUFFIX_NET_SC   = "sc.net";       // bayesian net generated from LibB using sparse candidate
	public final static String FSUFFIX_ARFF     = ".arff";        // training or test data
	public final static String FSUFFIX_NAME     = ".name";        // used in LibB for SC algorithm
	public final static String FSUFFIX_MMHC     = "mmhc.txt";     // data used in MMHC algorithm
	public final static String FSUFFIX_GT       = "GT.TXT";       // ground truth
	public final static String FSUFFIX_LN       = "LN.TXT";       // learned net
	public final static String FSUFFIX_LN_PC    = "LN_PC.TXT";    // learned net using PC
	public final static String FSUFFIX_LN_SC    = "LN_SC.TXT";    // learned net using sparse candidate
	public final static String FSUFFIX_LN_NEW   = "LN_NEW.TXT";   // with class attribute added in to build up the BN classifier
	public final static String FSUFFIX_LN_MMHC  = "LN_MMHC.TXT";  // learned net using MMHC
	public final static String FSUFFIX_DAG      = "DAG.TXT";      // directed acyclic graph
	public final static String FSUFFIX_CPDAG    = "CPDAG.TXT";    // completed partial 
	public final static String FSUFFIX_COUNT    = "COUNT.TXT";    // used by MMHC
	public final static String FSUFFIX_HEADER   = "HEADER.TXT";   // descriptions about nodes
	public final static String FSUFFIX_DETAIL   = "DETAIL.TXT";   // details at every stage
	public final static String FSUFFIX_ST       = "ST.TXT";       // bdeu score and build time 
	public final static String FSUFFIX_DC       = "DC.TXT";       // domain counts 
	public final static String FSUFFIX_EVA      = "EVA.TXT";      // evaluation 
	public final static String FSUFFIX_EVA_EXT  = "EVAEXT.TXT";   // modified files from EVA.TXT 
	public final static String FSUFFIX_EVA_SUM  = "EVASUM.TXT";   // sum of eva of scales varying from 100-5000
	public final static String FSUFFIX_EVA_AVE  = "EVAAVE.TXT";   // ave of eva of scales varying from 100-5000
	public final static String FSUFFIX_EVA_PC   = "EVA_PC.TXT";   // evaluation PC
	public final static String FSUFFIX_EVA_SC   = "EVA_SC.TXT";   // evaluation SC
	public final static String FSUFFIX_EVA_MMHC = "EVA_MMHC.TXT"; // evaluation MMHC
	public final static String FSUFFIX_SHD      = "SHD.TXT";      // shd
	public final static String FSUFFIX_SHD_EXT  = "SHDEXT.TXT";   // modified files from SHD.TXT
	public final static String FSUFFIX_SHD_SUM  = "SHDSUM.TXT";   // sum of shd of scales varying from 100-5000
	public final static String FSUFFIX_SHD_AVE  = "SHDAVE.TXT";   // ave of shd of scales varying from 100-5000
	public final static String FSUFFIX_SHD_PC   = "SHD_PC.TXT";   // shd PC
	public final static String FSUFFIX_SHD_SC   = "SHD_SC.TXT";   // shd SC
	public final static String FSUFFIX_SHD_MMHC = "SHD_MMHC.TXT"; // shd MMHC
	public final static String FSUFFIX_PCSET    = "PCSET.TXT";    // parents and children set from MMHC
	public final static String FSUFFIX_FNDAG    = "FNDAG.TXT";    // fully connected directed acyclic grapgh
	public final static String FSUFFIX_SUMMARY  = "SUMMARY.TXT";  // fully connected directed acyclic grapgh
	public final static String FSUFFIX_PREWHOLE = "PREWHOLE.TXT"; // predict using likelihood of the whole instance
	public final static String FSUFFIX_PRELOCAL = "PRELOCAL.TXT"; // predict using info of the local (Markov blanket) of the class attribute
	public final static String FSUFFIX_PREDICT  = "PREDICT.TXT";  // common predict file
	public final static String FSUFFIX_PRELL    = "PRELL.TXT";    // positive likelihood 
	public final static String FSUFFIX_NAIVEBN  = "NB.TXT";       // prediction precision of naive bayesian network classifier
	public final static String FSUFFIX_MB       = "MB.TXT";       // markov blanket file
	public final static String FSUFFIX_BESTEP   = "BESTEP.TXT";
	public final static String FSUFFIX_RUNTIME  = "RUNTIME.TXT";
	
	// pattern for the net name & the details of the nodes
	public static final String ARFF_NAME_PATTERN = "[@]relation(.*)";
	public static final String ARFF_VAR_PATTERN  = "[@]attribute(.*)[{](.*)[}]";
	
	// used to parse normal dot net file
	public static final String NET_VAR_PATTERN = "[(]var(.*)[(](.*)[)].*[)]";
	public static final String NET_PAR_PATTERN = "[(]parents(.*?)[(](.*?)[)]";
	
	// used to parse normal dot net file generated from LibB
	// ref: http://compbio.cs.huji.ac.il/LibB/programs.html
	public static final String NET_VAR_PATTERN_LIB = "[(]var.*?[\'](.*?)[\'][(](.*)[)].*[)]";
	public static final String NET_PAR_PATTERN_LIB = "[(]parents.*?[\'](.*?)[\'][(](.*?)[)]";
	
	// pattern for BIF files
	public static final String BIF_VAR_PATTERN		= "variable(.*)[{]";
	public static final String BIF_VAL_PATTERN		= "type.*[{](.*)[}]";
	public static final String BIF_PAR_PATTERN  	    = "probability.*[(](.*?)[|](.*?)[)]";
	public static final String BIF_PAR_SINGLE_PATTERN	= "probability.*[(](.*?)[)]";
	public static final String BIF_PRO_PATTERN 		= "[(](.*?)[)](.*)[;]";
	public static final String BIF_PRO_SINGLE_PATTERN = "table(.*)[;]";
	
	// pattern for data post process
	public static final String SHD_PATTERN  = "M:(.*?)E:(.*?)R:(.*?)RC:(.*)";
	public static final String EVA_PATTERN  = "(.*?)\\s(.*?)\\s(.*?)\\s(.*)";
	public static final String ST_PATTERN   = "(.*?)&(.*?)@(.*)";
	
	// two different kind of net file, net file generated from LibB included some boring "'"
	public static final int NET_FILE     = 0;
	public static final int NET_FILE_LIB = 1;
	
	// structure initialized from which source
	public static final int STRUCT_FROM_ALIST     = 0;
	public static final int STRUCT_FROM_AMATRIX   = 1;
	public static final int STRUCT_FROM_CPDAG     = 2;
	public static final int STRUCT_FROM_BIF       = 3;
	
	// how to combine two scores
	public static final int SCORE_DEFAULT     = 0;
	public static final int SCORE_ADD         = 1;
	public static final int SCORE_NORM_1      = 2;
	public static final int SCORE_NORM_2      = 3;
	public static final int SCORE_EMPTY       = 4;
	
	// used for equivalent sample size
	public final static double ESS = 10.0;
	public final static double ESS_CL = 10.0;
	
	// 
	public final static double ALPHA = 0.05;
	
	// used in BDeu score based metric
	public final static float A_SMALL_FLOAT_CONSTANT = 0.01f;
	public final static double A_SMALL_DELTA_SCORE = 1e-5;
	
	// logger configuration
	public final static String LOGGER_XML_CONF = "config/log4j.xml";
	
	// datasets
	public final static String[] nameOfDS = {
		"alarm", "asia", "insurance", "child", "sachs", "water", "hepar2", "win95pts", "hailfinder", "andes"
	};
	public final static String[] moreDS = {
		"pathfinder", "munin1", "link", "pigs", "munin2", "munin3", "munin4", "munin", "mildew", "barley", "diabetes"
	};
	
	public final static int[] N_VARIABLE = {37, 8, 27, 20, 11, 32, 70, 76, 56, 223, 35, 48, 441};
	
	// sample scales, practically, multiplied by a factor of 100
	private final static double[] SCALES = {0.2, 1, 5, 10, 50, 100, 500, 1000};
	
	
    /**
     * Perhaps Map is a better choice.
     * 
     * @param iAlgorithm
     * 		index of the algorithm
     * @return 
     * 		algorithm flag
     * 
     */
    public static String getAlgorithmFlag(int iAlgorithm) {
    	String IA = "";
    	
    	switch ( iAlgorithm ) {
	 	case 0:
	 		IA = "_CL0_"; // CL0
	 		break;
	 	case 1:
	 		IA = "_TABU_LIMITEDTIME_"; // Tabu
	 		break;
	 	case 2:
	 		IA = "_20160717_";
	 		break;
	 	case 3:
	 		IA = "_CL_";
	 		break;
 		default:
 			System.err.println("->Undefined Algorithm ID: " + iAlgorithm);
 			System.exit(0);
 			break;
	 	}
    	return IA;
    }
    
    
    @SuppressWarnings("rawtypes")
    public static void printSystemProperty(String key, boolean printAll) {
    	if ( printAll ) {
    		Properties properties = System.getProperties();  
			Iterator it =  properties.entrySet().iterator();  
    		while( it.hasNext() ) { 
    		    Entry entry = (Entry)it.next();  
    		    System.out.print(entry.getKey() + " = ");  
    		    System.out.println(entry.getValue()); 
    		}  
    	} else if ( !key.equals("") ){
    		System.out.println("key = " + key + "; value = " + 
    				System.getProperty(key));
    	}
    }
    
	
	/**
	 * Get the larger or smaller one.
	 * 
	 */
	public static boolean compareAandB(
			double a, 
			double b, 
			boolean getLarger) {	
		
		if ( getLarger ) { 
			return a > b; 
		} else {
			return a < b; 
		}
	}
	
	
    /**
     * Copy or move.
     * 
     */
    public static void migrateFile(
    		String orgPath, 
    		String desPath, 
    		boolean isCopy) {
    	
		try {
			// migrate
	    	File orgFile = new File(orgPath);  			
			File newFile = new File(desPath);
			
	    	// copy or move
			if ( !isCopy ) {
				orgFile.renameTo(newFile);
			} else {
				InputStream in = new FileInputStream(orgFile);
				OutputStream out = new FileOutputStream(newFile);
				
				byte[] buffer = new byte[1024];
				int ins;
				while ( (ins = in.read(buffer)) > 0 ) {
					out.write(buffer, 0, ins);
				}
				in.close();
				out.close();
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("->Migrate File Error: Org = " + orgPath + 
					"\nDes = " + desPath + "\nIsCopy: " + isCopy);
			e.printStackTrace();
		}
    	
    }
    
    
    /**
     * Format a float or double
     * 
     * @param number    
     * 		to be formated
     * @param precision 
     * 		number of significants
     * 
     */
    public static String format(Double number, int precision) {
    	String expression = "%." + String.valueOf(precision) + "f";
    	return String.format(expression , number);
    }
	
	
    /**
     * Please make sure the path is valid, since there are no further checks.
     * 
     * @param content 
     * 		in string
     * @param path 
     * 		path & name of the file
     * 
     */
    public static void writeFile(String content, String path) {
    	FileWriter outfile;
		try {
			outfile = new FileWriter(path);
			outfile.write(content);
	    	outfile.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("->Write File Error: Path = " + path);
			e.printStackTrace();
		}
    }
    
    
    /**
     * Get a customized scale sequence
     * 
     * @return scale list
     * 
     */
    public static int[] getScaleSequence(int iScale, int nScale) {
    	if ( (iScale + nScale) > SCALES.length ) {
    		System.out.println("->Far too large scale: " + 
    				(SCALES[SCALES.length - 1] * 100));
    		System.exit(0);
    	}
    	
    	int factor = 100, count = 0;
    	int[] scales = new int[nScale];
    	for ( int i = iScale; i < (iScale + nScale); i++ ) {
    		scales[count] = (int)(SCALES[i] * factor);
    		count++;
    	}
    	return scales;
    }
      
    
	/**
	 * Load input arff file into instance object
	 * 
	 * @param fileName   
	 * 		input arff file
	 * @return 
	 * 		arff file as instances
	 * 
	 */
	public static Instances loadInsts(String fileName) {
		Instances insts = null;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			insts = new Instances(reader);
			// we must specific an attribute to be used as class variable (x -> y)
			insts.setClassIndex(insts.numAttributes() - 1);
			reader.close();
			
			System.out.println("Reading from: " + fileName); // @debug
		} catch (Exception e) {
			System.out.println("Read file error: " + fileName);
			e.printStackTrace();
			return null;
		}
		return insts;
	}
	
	
	/**
	 * Generate arff file header according to the current <tt>instances</tt>.
	 * 
	 * @return {@code StringBuffer} containing the header info of arff file
	 */
	public static void writeArffHeader(Instances instances, String headFile) {
	    int nAttr = instances.numAttributes();
	    StringBuffer sb = new StringBuffer();
	    
	    sb.append("@relation " + instances.relationName() + "\n\n");
	    for ( int i = 0; i < nAttr; i++ ) {
	    	Attribute attr = instances.attribute(i);
	    	sb.append(attr.toString() + "\n");
	    	// System.out.println(attr.toString()); // @debug
	    }
	    sb.append("\n@data\n");
		
	    writeFile(sb.toString(), headFile);
	}
    
    
	/**
	 * @param  directory 
	 * 		under which the stuff will be returned.
	 * @return 
	 * 		array containing all the stuff under <tt>dir</tt>.
	 * 
	 */
	public static File[] getFileList(String directory) {
		File dir = new File(directory);
		return dir.listFiles();
	}
	
	
    /**
     * Generate the Faboncci sequence.
     * 
     * @param N 
     * 		scale list containing N numbers
     * 
     */
    public static int[] generateFaboncci(int N) {
     	// generate Fibonacci list
    	int nMaxTest = N;
     	int[] scales = new int[nMaxTest];
     	int current = 1, before = 1;
     	for ( int i = 0; i < nMaxTest; i++ ) {
     		scales[i] = current;
     		current = current + before;
     		before = scales[i];
     	}
     	return scales;
    }
    
    
	/**
	 * Make directories ready.
	 * 
	 * @param dirs 
	 * 		directories must exist
	 * 
	 */
	public static boolean initDirs(String dirs) {
		File fp = new File(dirs);
		if ( !fp.exists() ) { 
			if ( fp.mkdir() || fp.mkdirs() ) {
				System.out.println("Create Successfully: " + dirs);
				return true;
			}
		}
		return false;
	}
	
	
	/**
	 * Delete all the files whose name contains substring {@code pattern} under {@code root}. <p>
	 * 
	 * An example: 
	 * {@code String root = "/home/angus/garbage/";
	 * delSpecificile(new File(root), "garbage");}
	 * 
	 * @param root
	 * 		the directory
	 * @param pattern
	 * 		used as the regex to match file names under {@code root}
	 */
	public static void delSpecificile(File root, String pattern) {
		 
		 File[] files = root.listFiles();
		 for ( File file : files ) {
			 if ( file.isFile() && file.getName().contains(pattern) ) {
				 file.delete();
				 System.out.println("->Deleting " + file.getAbsolutePath());
			 } else if ( file.isDirectory() ) {
				 delSpecificile(file, pattern);
			 }
		 }
	}
	
}
