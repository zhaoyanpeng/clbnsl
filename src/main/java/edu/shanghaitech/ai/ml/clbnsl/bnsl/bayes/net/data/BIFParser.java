package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.BayesNetNode;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.Recorder;

/**
 * Parse BIF file to obtain BN structure and the associated distribution.
 * 
 * @author Yanpeng Zhao
 * 5/7/2015
 */
public class BIFParser extends Recorder {
	
	private String bifFile = "";
	private String netFile = "";
	private String nameDS  = "";
	
	private StringBuffer log = null;
	
	private FileWriter writer;
	private BufferedReader reader;
	
	private Logger logCons;
	private Logger logFile;
	
	public BIFParser() {
		this.log = new StringBuffer();
	}

	public void updateFileConfig(
			String bifFile, 
			String netFile,
			String logDest,
			String name) {
		this.bifFile = bifFile;
		this.netFile = netFile;
		this.logFile = logUtils.getFileLogger(AUtils.LOGGER_XML_CONF, logDest);
		this.logCons = logUtils.getConsoleLogger(AUtils.LOGGER_XML_CONF);
		this.nameDS    = name;
	}

	
	protected void printErrorMsg(int numLine, String msg) {
		logCons.error("Nonstandard BIF file at line " + numLine + ":\t" + msg);
		logFile.error("Nonstandard BIF file at line " + numLine + ":\t" + msg);
		System.exit(0); 
	}
	
	
	protected void printNodesInfo(ArrayList<BayesNetNode> nodes) {
		BayesNetNode node;
		log.delete(0, log.length());
		for ( int i = 0; i < nodes.size(); i++ ) {
			node = nodes.get(i);
			String[] varSet = node.getValSet();
			log.append(node.getName() + " | ");
			for ( int j = 0; j < varSet.length; j++ ) {
				log.append(varSet[j] + " ");
			}
			log.append("\n");
		}
		logFile.debug(log.toString());
		logCons.warn(log.toString());
	}
	
	
	protected int getIValue(
			ArrayList<BayesNetNode> nodes, 
			String name, 
			String value) {
		int iValue = -1;
		String[] valSet;
		BayesNetNode node;
		// more safe
		name = name.trim();
		value = value.trim();
		
		// @debug
		logFile.debug("name = " + name + "; value = " + value);
		logCons.debug("name = " + name + "; value = " + value);
		for ( int i = 0; i < nodes.size(); i++ ) {
			node = nodes.get(i);
			if ( name.equals(node.getName().trim()) ) {
				// @debug
				logFile.debug("Find name: " + name);
				logCons.debug("Find name: " + name);
				
				valSet = node.getValSet();
				for ( int j = 0; j < valSet.length; j++ ) {
					if ( value.equals(valSet[j].trim()) ) {
						iValue =  j;
						// @debug
						logFile.debug("Find value: " + value);
						logCons.debug("Find value: " + value);
						break;
					}
				}
				break;
			}
		}
		return iValue;
	}
	
	
	protected int getINode(ArrayList<BayesNetNode> nodes, String nodeName) {
		int iNode = -1;
		for ( int i = 0; i < nodes.size(); i++ ) {
			if ( nodeName.trim().equals(nodes.get(i).name.trim()) ) {
				iNode = i;
				break;
			}
		}
		if ( iNode < 0 ) { printErrorMsg(-1, "Cannot find |" + nodeName + "|"); }
		return iNode;
	}
	
	
	protected void updateNodeInfo(
			ArrayList<BayesNetNode> nodes, 
			String nodeName, 
			String[] parSet) {
		int cardinality = 1, iNode;
		int[] pars = new int[parSet.length];
		for ( int i = 0; i < parSet.length; i++ ) {
			iNode = getINode(nodes, parSet[i]);
			cardinality *= nodes.get(iNode).valSet.length;
			pars[i] = iNode;
		}
		iNode = getINode(nodes, nodeName);
		BayesNetNode node = nodes.get(iNode);
		node.setCardinalityOfParSet(cardinality);
		node.setParSet(pars);
	}
	
	
	protected void addCPD2Node(
			ArrayList<BayesNetNode> nodes, 
			String nodeName, 
			String[] cpds, 
			int irow, 
			int nLine) {
		int iNode = getINode(nodes, nodeName);
		BayesNetNode node = nodes.get(iNode);
		if ( cpds == null || cpds.length != node.valSet.length ) {
			printErrorMsg(nLine, "CPDs empty.");
		} else {
			for ( int i = 0; i < cpds.length; i++ ) {
				node.setCPD(irow, i, Double.valueOf(cpds[i].trim()));
			}
		}
	}
	
	
	protected void saveNode2File(ArrayList<BayesNetNode> nodes) {
		for ( int ii = 0; ii < nodes.size(); ii++ ) {
			logFile.debug(nodes.get(ii));
			logCons.debug(nodes.get(ii));
		}
	}
	
	
	public ArrayList<BayesNetNode> getNodeInfo() throws Exception {
		reader = new BufferedReader(
				new FileReader(new File(bifFile)));
		Pattern pVar = Pattern.compile(AUtils.BIF_VAR_PATTERN);
		Pattern pVal = Pattern.compile(AUtils.BIF_VAL_PATTERN);
		Pattern pPar = Pattern.compile(AUtils.BIF_PAR_PATTERN);
		Pattern pParSingle = Pattern.compile(AUtils.BIF_PAR_SINGLE_PATTERN);
		Pattern pPro = Pattern.compile(AUtils.BIF_PRO_PATTERN);
		Pattern pProSingle = Pattern.compile(AUtils.BIF_PRO_SINGLE_PATTERN);
		Matcher matcher;
		
		int nLine = 0;
		String line = "";
		ArrayList<BayesNetNode> nodes = new ArrayList<BayesNetNode>();
		boolean isPrint = true;
		while ( (line = reader.readLine()) != null ) {
			nLine++;
			if ( line.startsWith("variable") ) {
				BayesNetNode node = new BayesNetNode();
				matcher = pVar.matcher(line);
				if ( matcher.find() ) { // name of the node
					node.setName(matcher.group(1).trim());
				} else {
					printErrorMsg(nLine, null);
				}
				if ( (line = reader.readLine()) != null ) {
					nLine++;
					matcher = pVal.matcher(line);
					if ( matcher.find() ) { // values of the node
						String[] valSet = matcher.group(1).trim().split(",");
						node.setValSet(valSet);
					} else {
						printErrorMsg(nLine, null);
					}
				}
				nodes.add(node);
				// @debug
				logFile.debug(node.getName() + " | " + matcher.group(1).trim());
				logCons.debug(node.getName() + " | " + matcher.group(1).trim());

			} else if ( line.startsWith("probability") ) {
				if ( !isPrint ) { printNodesInfo(nodes); isPrint = true; }
				boolean single = false;
				String nodeName = null;
				String[] parSet = null;
				String[] valSet = null;
				String[] cpds   = null;
				matcher = pPar.matcher(line);
				if ( matcher.find() ) {
					nodeName = matcher.group(1).trim();
					parSet = matcher.group(2).trim().split(",");
				} else {
					matcher = pParSingle.matcher(line);
					if ( matcher.find() ) {
						single = true;
						nodeName = matcher.group(1).trim();
						parSet = new String[0]; // Avoid NullPointerException
					} else {
						printErrorMsg(nLine, null);
					}
				}
/************************************************************************************/					
				updateNodeInfo(nodes, nodeName, parSet); 
/************************************************************************************/					
				log.delete(0, log.length());
				log.append("------------------------------\n" + 
						"name: " + nodeName + " | ");
				for ( int i = 0; i < parSet.length; i++ ) {
					log.append(parSet[i] + " ");
				}
				log.append("\n------------------------------");
				logFile.debug(log.toString());
				logCons.debug(log.toString());
				
				if ( single ) {
					if ( (line = reader.readLine()) != null ) {
						nLine++;
						matcher = pProSingle.matcher(line);
						if ( matcher.find() ) {
/************************************************************************************/								
							cpds = matcher.group(1).trim().split(",");
							addCPD2Node(nodes, nodeName, cpds, 0, nLine);
/************************************************************************************/							
						} else {
							printErrorMsg(nLine, null);
						}
					}
				} else {
					while ( (line = reader.readLine() ) != null ) {
						
						// @debug
						logFile.debug("nline = " + line);
						logCons.debug("nline = " + line);
						
						nLine++;
						matcher = pPro.matcher(line);
						if ( matcher.find() ) {
							valSet = matcher.group(1).trim().split(",");
							if ( valSet.length == parSet.length ) {
								int iCPD = 0;
								for ( int i = 0; i < valSet.length; i++ ) {
									int iNode = getINode(nodes, parSet[i]);
									int iValue = getIValue(nodes, parSet[i], valSet[i]);
/************************************************************************************/										
									iCPD = iCPD * nodes.get(iNode).valSet.length + iValue; 									
								}		
								cpds = matcher.group(2).trim().split(",");
								addCPD2Node(nodes, nodeName, cpds, iCPD, nLine);
/************************************************************************************/	
								
							} 
						} else if ( line.trim().equals("}") ) {
							// @debug
							logFile.debug("Encounter '}' break now");
							logCons.debug("Encounter '}' break now");
							break;
						} else {
							printErrorMsg(nLine, null);
						}
					}
				}
			}
		}
		// @debug
		saveNode2File(nodes);
		reader.close();
		return nodes;
	}

	
	public void convert() throws Exception {
		reader = new BufferedReader(
				new FileReader(new File(bifFile)));
		writer = new FileWriter(netFile);
		Pattern pVar = Pattern.compile(AUtils.BIF_VAR_PATTERN);
		Pattern pVal = Pattern.compile(AUtils.BIF_VAL_PATTERN);
		Pattern pPar = Pattern.compile(AUtils.BIF_PAR_PATTERN);
		Pattern pParSingle = Pattern.compile(AUtils.BIF_PAR_SINGLE_PATTERN);
		Pattern pPro = Pattern.compile(AUtils.BIF_PRO_PATTERN);
		Pattern pProSingle = Pattern.compile(AUtils.BIF_PRO_SINGLE_PATTERN);
		Matcher matcher;
		
		int nLine = 0;
		String line = "";
		StringBuffer sb = new StringBuffer();
		ArrayList<BayesNetNode> nodes = new ArrayList<BayesNetNode>();
		boolean isPrint = false;
		while ( (line = reader.readLine()) != null ) {
			nLine++;
			if ( line.startsWith("variable") ) {
				BayesNetNode node = new BayesNetNode();
				matcher = pVar.matcher(line);
				if ( matcher.find() ) { // name of the node
					node.setName(matcher.group(1).trim());
				} else {
					printErrorMsg(nLine, null);
				}
				if ( (line = reader.readLine()) != null ) {
					nLine++;
					matcher = pVal.matcher(line);
					if ( matcher.find() ) { // values of the node
						String[] valSet = matcher.group(1).trim().split(",");
						node.setValSet(valSet);
					} else {
						printErrorMsg(nLine, null);
					}
				}
				nodes.add(node);
				// @debug
				logFile.debug(node.getName() + " | " + matcher.group(1).trim());
				logCons.debug(node.getName() + " | " + matcher.group(1).trim());
			} else if ( line.startsWith("probability") ) {
				if ( !isPrint ) { printNodesInfo(nodes); isPrint = true; }
				boolean single = false;
				String nodeName = null;
				String[] parSet = null;
				String[] valSet = null;
				matcher = pPar.matcher(line);
				if ( matcher.find() ) {
					nodeName = matcher.group(1).trim();
					parSet = matcher.group(2).trim().split(",");
					sb.append("(parents " + nodeName + 
							" ( " + matcher.group(2).trim().replace(",", " ") + " ) (");
				} else {
					matcher = pParSingle.matcher(line);
					if ( matcher.find() ) {
						single = true;
						nodeName = matcher.group(1).trim();
						parSet = new String[0]; // Avoid NullPointerException
						sb.append("(parents " + nodeName + " ( ) ( ");
					} else {
						printErrorMsg(nLine, null);
					}
				}
				
				log.delete(0, log.length());
				log.append("------------------------------\n" + 
						"name: " + nodeName + " | ");
				for ( int i = 0; i < parSet.length; i++ ) {
					log.append(parSet[i] + " ");
				}
				log.append("\n------------------------------");
				logFile.debug(log.toString());
				logCons.debug(log.toString());
				
				if ( single ) {
					if ( (line = reader.readLine()) != null ) {
						nLine++;
						matcher = pProSingle.matcher(line);
						if ( matcher.find() ) {
							sb.append(matcher.group(1).trim().replace(",", " ") + " ))\n\n");
						} else {
							printErrorMsg(nLine, null);
						}
					}
				} else {
					while ( (line = reader.readLine() ) != null ) {
						// @debug
						logFile.debug("nline = " + line);
						logCons.debug("nline = " + line);
						nLine++;
						matcher = pPro.matcher(line);
						if ( matcher.find() ) {
							valSet = matcher.group(1).trim().split(",");
							if ( valSet.length == parSet.length ) {
								for ( int i = 0; i < valSet.length; i++ ) {
									int iValue = getIValue(nodes, parSet[i], valSet[i]);
									if ( iValue >= 0 ) {
										valSet[i] = String.valueOf(iValue);
									} else {
										printErrorMsg(nLine, ("No value = " + valSet[i] + 
												" in node = " + parSet[i]) + 
												" at index = " + iValue);
									}
								}
								sb.append("\n(( ");
								for ( int i = 0; i < valSet.length; i++ ) {
									sb.append(valSet[i] + " ");
								}
								sb.append(") " + matcher.group(2).trim().replace(",", " ") + " )");
							} 
						} else if ( line.trim().equals("}") ) {
							sb.append("))\n\n");
							// @debug
							logFile.debug("Encounter '}' break now");
							logCons.debug("Encounter '}' break now");
							break;
						} else {
							printErrorMsg(nLine, null);
						}
					}
				}
			}
		}
		
		// net string
		BayesNetNode node;
		log.delete(0, log.length());
		for ( int i = 0; i < nodes.size(); i++ ) {
			node = nodes.get(i);
			log.append("(var " + node.getName() + " ( ");
			for ( int j = 0; j < node.getValSet().length; j++ ) {
				log.append(j + " ");
			}
			log.append("))\n");
		}
		log.append("\n");
		sb.insert(0, log.toString());
		sb.insert(0, ("(network " + nameDS + " :probability)\n"));
		// @debug
		logFile.debug(sb.toString());
		logCons.debug(sb.toString());
		// write
		writer.write(sb.toString());
		reader.close();
		writer.close();
	}

	
	public static void main(String[] args) throws Exception {
		String[] bifFiles = {"alarm", "andes", "asia", "barley", "cancer", 
				"child", "diabetes", "earthquake", "hailfinder", "hepar2",
				"insurance", "link", "mildew", "munin", "pathfinder", 
				"pigs", "sachs", "survey", "water", "win95pts"};
		String basePath = "E:/BNLearnBif2Net/";
		BIFParser parser = new BIFParser();
		for ( int i = 0; i < bifFiles.length; i++ ) {
			String bifFile = basePath + "bif/" + bifFiles[i] + ".bif";
			String netFile = basePath + "net/" + bifFiles[i] + ".net";
			String logFile = basePath + "log/" + bifFiles[i];
			
			parser.updateFileConfig(bifFile, netFile, logFile, bifFiles[i]);
			parser.convert();
		}
		parser.logCons.warn("Done! Congratulations!");
	}
	
}
