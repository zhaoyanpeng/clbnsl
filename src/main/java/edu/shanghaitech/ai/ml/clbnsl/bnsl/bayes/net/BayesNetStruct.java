package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data.BIFParser;
import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;

/**
 * @author Yanpeng Zhao
 * 5/1/2015
 */
public class BayesNetStruct {
	// description of Bayesian Network
	public String name = null;
	// number of the edges of the network
	public int nEdge = 0;
	// storing Bayesian Net that is represented by the ordered edge set,
	// and could represent a CPDAG, used for evaluation based on SHD
	public BayesNetEdge[] BNEdges = null;
	// number of the nodes contained in the net
	public int nNode = 0;
	// storing Bayesian Net that is represented by the node set,
	// used for evaluation based on info theory
	public BayesNetNode[] BNNodes = null;
	// storing Bayesian Net that is represented by a dag
	public char[][] dag = null;
	// storing Bayesian Net represented by cpdag, 
	// when the structure is initialized from cpdag
	public char[][] cpdag = null;
	// including details of the nodes
	private String headerPath = null;
	// including relationships among the nodes
	private String structPath = null;
	// file reader
	private BufferedReader reader = null;
	// would be replaced by logger
	private boolean debug = false;
	// default
	private int sourceTpye = AUtils.STRUCT_FROM_ALIST;
	
	
	public BayesNetStruct() {}
	
	
	public BayesNetStruct(String headerPath, String structPath, int sourceid) throws Exception {
		this.headerPath = headerPath;
		this.structPath = structPath;
		
		switch ( sourceid ) {
		case AUtils.STRUCT_FROM_ALIST:
			this.sourceTpye = AUtils.STRUCT_FROM_ALIST;
			getStructFromAlist();
			break;
		case AUtils.STRUCT_FROM_AMATRIX:
			this.sourceTpye = AUtils.STRUCT_FROM_AMATRIX;
			getStructFromAmatrix();
			break;
		case AUtils.STRUCT_FROM_CPDAG:
			this.sourceTpye = AUtils.STRUCT_FROM_CPDAG;
			getStructFromCPDAG();
			break;
		case AUtils.STRUCT_FROM_BIF:
			this.sourceTpye = AUtils.STRUCT_FROM_BIF;
			getStructFromBif();
			break;
		default:
			System.out.println("Err: un-defined soure id " + sourceid);
			System.exit(0);
		}
	}
	
	
	/**
	 * Record relationships between parents and children.
	 */
	protected boolean initStruct() {
		try {
			reader = new BufferedReader(
						new FileReader(new File(structPath)));
			
			int nPar;
			String[] temp;
			String line = "";
			
			// the number of nodes
			line = reader.readLine();
			this.nNode = Integer.parseInt(line.trim());
			
			// initialize BNNodes
			this.BNNodes = new BayesNetNode[nNode];
			
			this.nEdge = 0;
			for ( int i = 0; i < nNode; i++ ) {
				BNNodes[i] = new BayesNetNode();
				line = reader.readLine();
				
				temp = line.split(" ");
				nPar = Integer.parseInt(temp[1].trim());
				// total edges
				this.nEdge += nPar;
				
				int[] parSet = new int[nPar];
				for ( int j = 2; j < nPar + 2; j++ ) {
					parSet[j - 2] = Integer.parseInt(temp[j].trim());
				}
				BNNodes[i].setIName(i);
				BNNodes[i].setParSet(parSet);
			}
			this.BNEdges = new BayesNetEdge[nEdge];
			
			// @debug
			/*System.out.println("edge size: " + BNEdges.length);*/
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	protected boolean initStructFromCPDAG() {
		try {
			reader = new BufferedReader(
					new FileReader(new File(structPath)));
			String line = reader.readLine();
			String[] firstLine = line.split("@");
			this.nNode = Integer.valueOf(firstLine[0].trim());
			this.cpdag = new char[nNode][nNode]; // reserve memory
			
			int count;
			char[] temp;
			for ( int i = 0; i < nNode; i++ ) { 
				line = reader.readLine();
				temp = line.toCharArray();
				count = 0;
				for ( int j = 0; j < temp.length; j++ ) {
					if ( '0' <= temp[j] && temp[j] <= '9' ) {
						this.cpdag[i][count] = (char) (temp[j] - '0');
						count++;
					}
				}
			}
			
			this.nEdge = 0;
			for ( int i = 0; i < nNode; i++ ) {
				for ( int j = 0; j < nNode; j++ ) {
					if ( cpdag[i][j] == 1 ) { 
						this.nEdge++; 
					} else if ( cpdag[i][j] == 2 && i > j ) {
						this.nEdge++;
					}
				}
			}
			this.BNEdges = new BayesNetEdge[nEdge];
			
			count = 0;
			for ( int i = 0; i < nNode; i++ ) {
				for ( int j = 0; j < nNode; j++ ) {
					if ( cpdag[i][j] == 1 ) {
						BNEdges[count] = new BayesNetEdge(i, j, (char) 1);
						count++;
					} else if ( cpdag[i][j] == 2 && i > j ) {
						BNEdges[count] = new BayesNetEdge(i, j, (char) 2);
						count++;
					}
				}
			}
			
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	protected boolean initStructFromAmatrix() {
		try {
			reader = new BufferedReader(
					new FileReader(new File(structPath)));
					
	/**************************************************************************
	 * 0st phase: initialize dag
	 *************************************************************************/
			int count;
			String line = reader.readLine();
			String[] firstLine = line.split("@");
			this.nNode = Integer.valueOf(firstLine[0].trim());
			this.dag = new char[nNode][nNode]; // reserve memory
			char[] temp = line.toCharArray();
			
			for ( int i = 0; i < nNode; i++ ) { 
				line = reader.readLine();
				temp = line.toCharArray();
				
				count = 0;
				for ( int j = 0; j < temp.length; j++ ) {
					if ( '0' <= temp[j] && temp[j] <= '9' ) {
						this.dag[i][count] = (char) (temp[j] - '0');
						count++;
					}
				}
			}
			
	/**************************************************************************
	 * 0st phase: initialize nodes' parent set
	 *************************************************************************/
			this.nEdge = 0;
			this.BNNodes = new BayesNetNode[nNode];
			for ( int i = 0; i < nNode; i++ ) {
				int[] parSet = getParents(i);
				this.nEdge += getIndegree(i);
				
				BNNodes[i] = new BayesNetNode();
				BNNodes[i].setIName(i);
				BNNodes[i].setParSet(parSet);
			}
			this.BNEdges = new BayesNetEdge[nEdge];
			
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	protected boolean initStructFromBif() {
		try {
			BIFParser parser = new BIFParser();
			// headerPath is used as logPath
			parser.updateFileConfig(structPath, null, headerPath, name);
			ArrayList<BayesNetNode> nodes = parser.getNodeInfo();
			
			this.nNode = nodes.size();
			this.BNNodes = new BayesNetNode[nNode];
			for ( int i = 0; i < nNode; i++ ) {
				BNNodes[i] = new BayesNetNode();
				BNNodes[i].copy(nodes.get(i));
			}
			nodes.clear();
			
			// @debug
//			for ( int i = 0; i < nNode; i++ ) {
//				System.out.println(BNNodes[i]);
//			}
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	/**
	 * Details of nodes.
	 */
	protected boolean initNodes() {
		try {
			reader = new BufferedReader(
					new FileReader(new File(headerPath)));
			// name of the net
			int iNode = 0;
			String line = reader.readLine();
			Pattern pattern = Pattern.compile(AUtils.ARFF_NAME_PATTERN);
			Matcher matcher = pattern.matcher(line);
			if ( matcher.find() ) {
				this.name = matcher.group(1).trim();
			}
			
			pattern = Pattern.compile(AUtils.ARFF_VAR_PATTERN);
			while ( (line = reader.readLine()) != null ) {
				matcher = pattern.matcher(line);
				// avoid from the case when the number of attributes in the 
				// header file is larger than <tt>nNode</tt>
				if ( matcher.find() && iNode < nNode ) {
					BNNodes[iNode].setName(matcher.group(1).trim());
					String[] varSet = matcher.group(2).split(",");
					for ( int i = 0; i < varSet.length; i++ ) {
						varSet[i] = varSet[i].trim();
					}
					BNNodes[iNode].setValSet(varSet);
					iNode++;
				}
			}
			
			// initialize cardinalityOfPar
			for ( int i = 0; i < nNode; i++ ) {
				int cardinality = 1;
				int[] parSet = BNNodes[i].getParSet();
				for ( int j = 0; j < parSet.length; j++ ) {
					cardinality *= BNNodes[parSet[j]].getValSet().length;
				}
				BNNodes[i].setCardinalityOfParSet(cardinality);
			}
			return true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
	}
	
	
	/**
	 * Represented by a matrix.
	 */
	protected boolean initDAG() {
		try {
			this.dag = new char[nNode][nNode];
			for ( int i = 0; i < nNode; i++ ) {
				for ( int j = 0; j < nNode; j++ ) {
					dag[i][j] = 0;
				}
			}
			
			for ( int i = 0; i < nNode; i++ ) {
				int[] parSet = BNNodes[i].getParSet();
				for ( int j = 0; j < parSet.length; j++ ) {
					int iPar = parSet[j];
					dag[iPar][i] = 1;
				}
			}
			
			return true;
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}
	}
	
	
	/**
	 * CPDAG is stored as the edge set.
	 */
	public void dag2cpdag() {
		
		orderEdge(); // before all the stuffs below
		
		int parent, child;
		BayesNetEdge[] edges = this.BNEdges;
		// traverse the unknown edges
		for ( int i = 0; i < nEdge; i++ ) {
			if ( edges[i].status == 0 ) {
				parent = edges[i].x; // x
				child = edges[i].y;  // y
				boolean fin = false;
				for ( int j = 0; j < nEdge; j++ ) {
					try {
						if ( edges[j].y == parent && edges[j].status == 1 ) {
							if ( !this.isParent(child, edges[j].x) ) {
								// x->y included
								for ( int k = 0; k < nEdge; k++ ) {
									if ( edges[k].y == child ) {
										edges[k].setStatus((char)1);
									}
								}
								fin = true;
							} else {
								edges[j].setStatus((char)1);
							}
						}
					} catch (Exception e) {
						System.out.println(j + "\t" + edges.length);
						System.out.println(e.toString());
						System.exit(0);;
					}
				}
				
				if ( !fin ) {
					boolean exist = false;
					for ( int j = 0; j < nEdge; j++ ) {
						// existing z->y s.t. z != x & z is not a parent of x
						if ( edges[j].y == child && 
							 edges[j].x != parent && 
							 !this.isParent(parent, edges[j].x) ) {
							// x->y included
							for ( int k = 0; k < nEdge; k++ ) {
								if ( edges[k].y == child && 
									 edges[k].status == 0 ) {
									edges[k].setStatus((char)1);
								}
							}
							exist = true;
							break;
						} 
					}
					// no such z->y s.t. ...
					if ( !exist ) {
						// x->y included
						for ( int k = 0; k < nEdge; k++ ) {
							if ( edges[k].y == child && 
								 edges[k].status == 0 ) {
								edges[k].setStatus((char)2);
							}
						}
					}
				}
			}
		}
		// @debug
		if ( debug ) {
			System.out.println("->Ordered Edges " + nEdge);
			for ( int i = 0; i < nEdge; i++ ) {
				System.out.println(edges[i].x + " " + edges[i].y + " " + 
						(int)(edges[i].status));
			}
			System.out.println();
		}
	}
	
	
	/**
	 * Initializing the ordered edge set.
	 */
	protected void orderEdge() {
		// order & edges 
		int[] orders = topologicalSort();
		int iEdge = 0;
		for ( int i = 1; i < nNode; i++ ) {
			for ( int j = i - 1; j >= 0; j-- ) {
				// "status = 2" means "unknown"
				if ( this.isParent(orders[i], orders[j])) {
					this.BNEdges[iEdge] = new BayesNetEdge(orders[j], orders[i], (char)0);
					iEdge++;
				}
			}
		}
		// @debug
		/*System.out.println("true edge size: " + iEdge);*/
		// @debug
		if ( debug ) {
			System.out.println(this.name + " CPDAG Info");
			System.out.println("->Ordered Nodes " + nNode);
			for ( int i = 0; i < nNode; i++ ) {
				System.out.print(orders[i] + " ");
			}
			System.out.println();
		}
	}
	
	
	/**
	 * Initializing the ordered node set.
	 */
	public int[] topologicalSort() {
		int[] inDegree = new int[nNode];
		Queue<Integer> nodeOfNoPar = new LinkedList<Integer>();
		for ( int i = 0; i < nNode; i++ ) {
			if ( this.getIndegree(i) == 0 ) {
				nodeOfNoPar.offer(i);
			}
			inDegree[i] = this.getIndegree(i);
		}
		
		int count = 0, iNode;
		int[] orders = new int[nNode];
		while ( !nodeOfNoPar.isEmpty() ) {
			iNode = nodeOfNoPar.poll().intValue();
			orders[count] = iNode;
			count++;
			int[] children = this.getChildren(iNode);
			for ( int i = 0; i < children.length; i++ ) {
				inDegree[children[i]] = inDegree[children[i]] - 1;
				if ( inDegree[children[i]] == 0 ) {
					nodeOfNoPar.offer(children[i]);
				}
			}
		}
		// @debug
		if ( count != nNode ) { System.out.println("-->There is a critical error."); }
		return orders;
	}
	
	
	public boolean isReversible(int child, int parent) {
		for ( int i = 0; i < nEdge; i++ ) {
			if ( BNEdges[i].x == child && BNEdges[i].y == parent ) {
				if ( BNEdges[i].status == 2 ) {
					return true;
				} else {
					return false;
				}
			}
		}
		return false;
	}
	
	
	public boolean isContained(BayesNetEdge edge) {
		boolean isContained = false;
		for ( int i = 0; i < nEdge; i++ ) {
			if ( (BNEdges[i].x == edge.x && BNEdges[i].y == edge.y) ||
				 (BNEdges[i].x == edge.y && BNEdges[i].y == edge.x) ) {
				isContained = true;
				break;
			}
		}
		return isContained;
	}
	
	
	public boolean isReversed(BayesNetEdge edge, boolean isDAG) {
		boolean isReversed = false;
		int i = 0; 
		for ( ; i < nEdge; i++ ) {
			if ( (BNEdges[i].x == edge.x && BNEdges[i].y == edge.y) ||
				 (BNEdges[i].x == edge.y && BNEdges[i].y == edge.x) ) {
				// contained & reversed
				if ( (BNEdges[i].x == edge.y && BNEdges[i].y == edge.x) ) {
					isReversed = true;
				}
				break;
			}
		}
		
		if ( isDAG ) {
			return isReversed;
		} else { // CPDAG reversible or not
			if ( isReversed ) {
				return BNEdges[i].status == 1;
			} else {
				return false;
			}
			
		} 
	}

	
	public boolean isChild(int iNode, int iChild) {
		return dag[iNode][iChild] != 0 ? true : false;
	}
	
	
	public boolean isParent(int iNode, int iParent) {
		return dag[iParent][iNode] != 0 ? true : false;
	}
	
	
	public int getIndegree(int iNode) {
		int count = 0;
		for ( int i = 0; i < nNode; i++ ) {
			if ( dag[i][iNode] != 0 ) {
				count++;
			}
		}
		return count;
	}
	
	
	public int getOutdegree(int iNode) {
		int count = 0;
		for ( int i = 0; i < nNode; i++ ) {
			if ( dag[iNode][i] != 0 ) {
				count++;
			}
		}
		return count;
	}
	
	
	public int[] getChildren(int iNode) {
		int count = 0;
		int[] children = new int[getOutdegree(iNode)];
		for ( int i = 0; i < nNode; i++ ) {
			if ( dag[iNode][i] != 0 ) {
				children[count] = i;
				count++;
			}
		}
		return children;
	}
	
	
	public int[] getParents(int iNode) {
		int count = 0;
		int[] parents = new int[getIndegree(iNode)];
		for ( int i = 0; i < nNode; i++ ) {
			if ( dag[i][iNode] != 0 ) { 
				parents[count] = i;
				count++;
			}
		}
		return parents;
	}
	
	
	public int getMaxIndegree() {
		int maxDegree = -1, degree;
		for ( int i = 0; i < nNode; i++ ) {
			degree = getIndegree(i);
			maxDegree = degree > maxDegree ? degree : maxDegree;
		}
		return maxDegree;
	}
	
	
	public int getMaxOutdegree() {
		int maxDegree = -1, degree;
		for ( int i = 0; i < nNode; i++ ) {
			degree = getOutdegree(i);
			maxDegree = degree > maxDegree ? degree : maxDegree;
		}
		return maxDegree;
	}

	/**
	 * Function calls must obey the following order due to the dependencies among the functions.
	 * Keyword "throws" is preferred since it could give clues where the errors occur.
	 */
	public void getStructFromAlist() throws Exception {
		
		if ( !initStruct() ) {
			throw new FileNotFoundException("Err: initStructure");
			// System.exit(0);
		}
		
		if ( !initNodes() ) {
			System.out.println("Err: initNodes.");
			new Throwable();
			// System.exit(0);
		}
		
		if ( !initDAG() ) {
			System.out.println("Err: initDAG.");
			new Throwable();
			// System.exit(0);
		}
		dag2cpdag();
	}
	
	
	public void getStructFromAmatrix()  throws Exception {
		if ( !initStructFromAmatrix() ) {
			throw new FileNotFoundException("Err: initStructure");
			// System.exit(0);
		}
		if ( !initNodes() ) {
			System.out.println("Err: initNodes.");
			System.exit(0);
		}
		dag2cpdag();
	}
	
	
	public void getStructFromCPDAG() {
		if ( !initStructFromCPDAG() ) {
			System.out.println("Err: initStructure.");
			System.exit(0);
		}
	}
	
	
	public void getStructFromBif() {
		if ( !initStructFromBif() ) {
			System.out.println("Err: initStructure.");
			System.exit(0);
		}
		if ( !initDAG() ) {
			System.out.println("Err: initDAG.");
			System.exit(0);
		}
	}
	
	
	public void setDebug(boolean debug) {
		this.debug = debug;
	}
	
	
	public String getName() {
		return this.name;
	}
	
	
	public int getNEdge() {
		return this.nEdge;
	}

	
	public BayesNetEdge[] getBNEdge() {
		return this.BNEdges;
	}
	
	
	public int getNNode() {
		return this.nNode;
	}
	
	
	public BayesNetNode[] getBNNodes() {
		return this.BNNodes;
	}
	
	
	public char[][] getDAG() {
		return this.dag;
	}
	
	
	public int getSourceType() {
		return this.sourceTpye;
	}
	
	
	public void printDAG() {
		for ( int i = 0; i < nNode; i++ ) {
			for ( int j = 0; j < nNode; j++ ) {
				System.out.print((int)dag[i][j] + " ");
			}
			System.out.println();
		}
	}
	
	
	public void printCPDAG() {
		for ( int i = 0; i < nEdge; i++ ) {
			System.out.println(
					BNEdges[i].x + " " + 
					BNEdges[i].y + " " + 
					(int) BNEdges[i].status);
		}
	}
	
	
	@Override
	public String toString() {
		StringBuffer net = new StringBuffer();
		net.append("@relation " + this.name + "\n");
		// because the structure from CPDAG does not have configured nodes
		if ( sourceTpye != AUtils.STRUCT_FROM_CPDAG ) { 
			// details of the nodes
			for ( int i =  0; i < nNode; i++ ) {
				String[] varSet = BNNodes[i].getValSet();
				net.append("@attribute " + BNNodes[i].name);
				
				net.append(" {");
				for ( int j = 0; j < varSet.length - 1; j++ ) {
					net.append(varSet[j] + ",");
				}
				net.append(varSet[varSet.length - 1] + "}\n");
			}
			net.append("\n");
			
			// details of the structure
			net.append("@parents\n" + this.nNode + "\n");
			for ( int i =  0; i < nNode; i++ ) {
				int[] parSet = BNNodes[i].getParSet();
				
				net.append(String.valueOf(i) + " " + 
						String.valueOf(parSet.length) + " ");
				if ( parSet.length > 0 ) {
					for ( int j = 0; j < parSet.length - 1; j++ ) {
						net.append(parSet[j] + " ");
					}
					net.append(parSet[parSet.length - 1]);
					net.append(" " + BNNodes[i].getCardinalityOfParSet());
				}
				net.append("\n");
			}
			net.append("\n");
			
			// details of the structure
			net.append("@parents from dag\n" + this.nNode + "\n");
			for ( int i =  0; i < nNode; i++ ) {
				int[] parSet = this.getParents(i);
				
				net.append(String.valueOf(i) + " " + 
						String.valueOf(parSet.length) + " ");
				if ( parSet.length > 0 ) {
					for ( int j = 0; j < parSet.length - 1; j++ ) {
						net.append(parSet[j] + " ");
					}
					net.append(parSet[parSet.length - 1]);
				}
				net.append("\n");
			}
			net.append("\n");
			
			// details of the structure
			net.append("@children from dag\n" + this.nNode + "\n");
			for ( int i =  0; i < nNode; i++ ) {
				int[] chiSet = this.getChildren(i);
				
				net.append(String.valueOf(i) + " " + 
						String.valueOf(chiSet.length) + " ");
				if ( chiSet.length > 0 ) {
					for ( int j = 0; j < chiSet.length - 1; j++ ) {
						net.append(chiSet[j] + " ");
					}
					net.append(chiSet[chiSet.length - 1]);
				}
				net.append("\n");
			}
			net.append("\n");
			
			// structure represented by the dag
			net.append("@dag " + this.nEdge + "\n" + this.nNode + "\n");
			for ( int i = 0; i < nNode; i++ ) {
				for ( int j = 0; j < nNode; j++ ) {
					net.append((int)dag[i][j] + " ");
				}
				net.append("\n");
			}
		} else {
			// structure represented by the dag
			net.append("@cpdag " + this.nEdge + "\n" + this.nNode + "\n");
			for ( int i = 0; i < nNode; i++ ) {
				for ( int j = 0; j < nNode; j++ ) {
					net.append((int)cpdag[i][j] + " ");
				}
				net.append("\n");
			}
		}
		return net.toString();
	}
	
	
	public static void main(String[] args) throws Exception {
//		String structPath = "E:/BIFBIFBIF/bif/alarm.bif";
//		String headerPath = "E:/BIFBIFBIF/bif/alarm";
//		BayesNetStruct bns = new BayesNetStruct(headerPath, structPath, AUtils.STRUCT_FROM_BIF);
//		System.out.print(bns);
	}

}
