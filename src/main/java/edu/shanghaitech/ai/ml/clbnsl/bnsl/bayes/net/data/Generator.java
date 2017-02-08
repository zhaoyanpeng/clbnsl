package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;

/**
 * This class is unnecessary. I keep it just because there are
 * some useful methods that could be referred to in future.
 *
 * @author Yanpeng Zhao
 */
public class Generator {
	// re-generate the same data set, will be reinitialized
	protected long seed = System.currentTimeMillis();

	protected int iScale = 0;  // index of the starting scale
	protected int nScale = 0;  // index of the last scale

	protected int iTestDS = 0; // index of the starting data set
	protected int nTestDS = 0; // index of the last data set
	
	protected int[] scales = null; // different data set size

	protected String basePath = ""; // root
	protected String netPath = "";  // input
	protected String resPath = "";  // output
	
	protected FileWriter writer;
	protected BufferedReader reader;
	
	public Generator() {}
	
	
	/**************************************************************************
	 * 0st section: initialization
	 **************************************************************************/
	/**
	 * Used as the constructor.
	 * 
	 */
	protected void initConfig(
			int seed, 
			int iScale, 
			int nScale, 
			int iTestDS, 
			int nTestDS, 
			String basePath) {
		
		this.seed = seed;
		this.iScale = iScale;
		this.nScale = nScale;
		this.iTestDS = iTestDS;
		this.nTestDS = nTestDS;
		this.basePath = basePath;
		this.netPath = basePath + "net/";
		this.resPath = basePath + "res/";
		
		// initialize directories
		AUtils.initDirs(this.netPath);
		AUtils.initDirs(this.netPath);
		
		// initialize the sequence of the dataset size
		scales = AUtils.getScaleSequence(0, nScale);
	}
	
	
	/**************************************************************************
	 * 1st section: using tools from 
	 * 				http://compbio.cs.huji.ac.il/LibB/programs.html
	 **************************************************************************/
	/**
	 * @deprecated 
	 * 		The tools are too old to correctly output the results. But the method
	 * 		of executing CMD on windows is sort of useful.
	 * 
	 */
	protected void evokeCMD(String command) {
		try {
			Process process;
	    	Runtime runtime = Runtime.getRuntime();
			/*System.out.println(go);*/
			process = runtime.exec(command);
			
			// blocking, get IO instead of evoking 'Process.waitFor()'
			InputStream is = process.getInputStream();
			BufferedReader br = new BufferedReader(
					new InputStreamReader(is));
			String line;
			while ((line = br.readLine()) != null) {
				System.out.println("cmd output: " + line);
			}
			br.close();
			// @debug message
			System.out.println("Execute Successfully: " + command);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}
	
	
	/**
     * @deprecated
     * 		There's nothing to do with thread, but when you have 
     * 		too many samples to generate, this would be helpful.
     * 
     */
    protected class GenSampleThread extends Thread {
		@Override
		public void run() {
			// TODO Auto-generated method stub
	    	Runtime runtime = Runtime.getRuntime();
	    	String command = "cmd /c start cmd.exe /c " + 
	    			netPath + "GenInstance";
	    	Process process;
	    	for ( int i = 0; i < nTestDS; i++ ) {
	    		String name = AUtils.nameOfDS[i];
	    		String savePath = netPath + name + "/seed" + 
	    				String.valueOf(seed) + "/";
	    		// make directory
	    		if ( !AUtils.initDirs(savePath) ) {
	    			System.out.println("Dir Err: " + savePath);
	    		}
	    		for ( int j = 0; j < scales.length; j++ ) {
	        		String scale = String.valueOf(scales[j]);
	        		String go = command + " " + 
	        				netPath + name + "/" + name + 
	        				AUtils.FSUFFIX_NET + " " +
	        				"-# " + scale + " " + 
	        				"-i " + savePath + scale + ".txt" + " " +
	        				"-s " + String.valueOf(seed); 
	        		/*System.out.println(go);*/
	        		try {
						process = runtime.exec(go);
						// it couldn't end current IO, see method "evokeCMD"
						process.waitFor(); 
						System.out.println("Generate Successfully: " + go);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}	     		
	        	}
	    	} 
		}
    }
    
	
	/**************************************************************************
	 * 2st section: net file parsing, generate corresponding GT files
	 **************************************************************************/
	/**
	 * There are different formats of the net file.
	 * 
	 * @deprecated 
	 * 		I have given up parsing net files.
	 * 
	 */
	protected void parseNetFile(
			String pathOfNetFile, 
			String pathToSave, 
			int netFileType) throws Exception {
		String varPattern = "";
		String parPattern = "";
		switch( netFileType ) {
		case AUtils.NET_FILE:
			varPattern = AUtils.NET_VAR_PATTERN;
			parPattern = AUtils.NET_PAR_PATTERN;
			break;
		case AUtils.NET_FILE_LIB:
			varPattern = AUtils.NET_VAR_PATTERN_LIB;
			parPattern = AUtils.NET_PAR_PATTERN_LIB;
			break;
		default:
			System.out.println("->Undefined NetFileType: " + netFileType);
 			System.exit(0);
		}
		parseNetFile(pathOfNetFile, pathToSave, varPattern, parPattern);
	}
	
	
	/**
	 * Generate GT represented by index of the node according to the .net file.
	 * 
	 * @deprecated 
	 * 		I have given up parsing net files.
	 * 
	 */
	protected void parseNetFile(
			String pathOfDotNetFile, 
			String pathToSave,
			String varPattern,
			String parPattern) throws Exception {
		reader = new BufferedReader(
				new FileReader(new File(pathOfDotNetFile)));
		int iNode = 0;
		String line = "";
		Map<String, Integer> nodeAndIndex = 
				new LinkedHashMap<String, Integer>();
		Map<String, String[]> nodeAndParents = 
				new LinkedHashMap<String, String[]>();
		Pattern vpattern = Pattern.compile(varPattern);
		Pattern ppattern = Pattern.compile(parPattern);
		
		while( (line = reader.readLine()) != null ) {
			// filter variables and index them
			Matcher matcher = vpattern.matcher(line);
			if ( matcher.find() ) { 
				nodeAndIndex.put(matcher.group(1).trim(), iNode);
				System.out.println(matcher.group(1).trim() + "--" + iNode);
				iNode++;
			} else if ( line.startsWith("(parents") ) {
				// filter variables' parents and store them with their index
				matcher = ppattern.matcher(line);
				if ( matcher.find() ) {
					String[] parents = matcher.group(2).trim().split(" ");
					nodeAndParents.put(matcher.group(1).trim(), parents);
				}
				while( (line = reader.readLine()) != null ) {
					matcher = ppattern.matcher(line);
					if ( matcher.find() ) {
						String[] parents = matcher.group(2).trim().split(" ");
						nodeAndParents.put(matcher.group(1).trim(), parents);
					}
				}
			}
		}
		reader.close();
		// @debug
		System.out.println();
		writeGTFile(pathToSave, nodeAndIndex, nodeAndParents);
	}
	
	
	/**
	 * Write the description of net structure into file.
	 * 
	 * @deprecated 
	 * 		I have given up parsing net files.
	 * 
	 */
	protected void writeGTFile(String pathToSave,
			Map<String, Integer> nodeAndIndex,
			Map<String, String[]> nodeAndParents) throws Exception {
		writer = new FileWriter(pathToSave);
		StringBuffer structure = new StringBuffer();
		// saint check
		System.out.println();
		for ( Map.Entry<String, Integer> entry : nodeAndIndex.entrySet() ) {
			if ( nodeAndParents.get(entry.getKey()) == null ) {
				System.out.println("Err: parents == null; iNode = " + 
						entry.getValue() + "; nodeName: " + entry.getKey());
				System.exit(0);
			}
			// @debug
			/*System.out.println(entry.getKey() + "->" + entry.getValue());*/
		}
		// every node
		int nNode = nodeAndIndex.size();
		for ( Map.Entry<String, Integer> entry : nodeAndIndex.entrySet() ) {
			Integer iNode = entry.getValue();
			String nameOfNode = entry.getKey();
			String[] parents = nodeAndParents.get(nameOfNode);
			// @debug
			System.out.println(nameOfNode + "->" + iNode + "|" + parents.length);
			int nParent = 0;
			String indexOfParent = "" ;
			for ( int i = 0; i < parents.length; i++ ) {
				// @debug
				System.out.println("----------------" + "|" + 
						parents[i].trim() + "|" + 
						((parents[i].trim()) == "") + "; " + 
						((parents[i].trim()).equals("")) + "; ");
				if ( parents[i].trim().equals("") || parents[i].trim() == null) {
					// TODO
				} else {
					indexOfParent = indexOfParent + 
							nodeAndIndex.get(parents[i].trim())
							 + " "; 
					nParent++;
				}
			}
			structure.append(
					iNode.toString() + " " + 
					Integer.toString(nParent) + " " + 
					indexOfParent + "\n");
		}
		writer.write((Integer.toString(nNode) + "\n" +structure.toString()));
		writer.close();
		// @debug
		System.out.println("\n"); 
	}
}
