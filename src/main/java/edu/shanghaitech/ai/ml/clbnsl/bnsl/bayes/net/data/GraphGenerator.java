package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.net.data;

import java.io.FileWriter;
import java.util.Random;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;

/**
 * Reference: {@link http://mathematica.stackexchange.com/questions/608/how-to-generate-random-directed-acyclic-graphs}
 * 
 * @author Yanpeng Zhao
 * 7/25/2015
 */
public class GraphGenerator extends Generator {
	
	private Random random;
	
	public GraphGenerator(int seed) {
		this.random = new Random(seed);
	}
	

	public String generateDAG(int v, int e, String dagFilePath) {
		int va, vb, inter;
		int[] order = new int[v];
		char[][] dag = new char[v][v];
		
		initDAG(dag, v);
		permute(order, v);
		
		for ( int i = 0; i < e; ) {
			va = random.nextInt(v);
			vb = random.nextInt(v);
			if ( va == vb ) { continue; }
			if ( va > vb ) { inter = va; va = vb; vb = inter; }
			va = order[va];
			vb = order[vb];
			if ( dag[va][vb] == 0 ) { dag[va][vb] = 1; i++; }
		}
		
		StringBuffer sb = new StringBuffer();
		for ( int i = 0; i < v; i++ ) {
			for ( int j = 0; j < v; j++ ) {
				if ( dag[i][j] != 0 ) {
					sb.append(j + " ");
				}
			}
			sb.append("\n");
		}
		// @debug
//		printOrder(order, v);
//		printDAG(dag, v);
		return sb.toString();
	}
	
	
	protected void permute(int[] order, int v) {
		int j, k;
		initOrder(order, v);
		for ( int i = 0; i < v - 1; i++ ) {
			j = i + random.nextInt(v - i);
			k = order[i];
			order[i] = order[j];
			order[j] = k;
		}
	}
	
	
	protected void initOrder(int[] order, int v) {
		for ( int i = 0; i < v; i++ ) {
			order[i] = i;
		}
	}
	
	
	protected void initDAG(char[][] dag, int v) {
		for ( int i = 0; i < v; i++ ) {
			for ( int j = 0; j < v; j++ ) {
				dag[i][j] = 0;
			}
		}
	}
	
	
	protected void printOrder(int[] order, int v) {
		System.out.println("-----\tORDER\t-----");
		for ( int i = 0; i < v; i++ ) {
			System.out.print(order[i] + " ");
		}
		System.out.println();
	}
	
	
	protected void printDAG(char[][] dag, int v) {
		System.out.println("-----\tDAG\t-----");
		for ( int i = 0; i < v; i++ ) {
			for ( int j = 0; j < v; j++ ) {
				System.out.print((int) dag[i][j] + " ");
			}
			System.out.println();
		}
	}
	
	
	public void genRandomDAG(int seed) throws Exception {
		String[] names = AUtils.nameOfDS;
		int[] nVariables = AUtils.N_VARIABLE;
		String basePath = "E:/NewData/res/";
		
		String savePath = null;
		FileWriter writer = null;
		for ( int i = 0; i < names.length; i++ ) {
			savePath = basePath + names[i] + "/" + names[i] + "_" + String.valueOf(seed) + "_" + AUtils.FSUFFIX_FNDAG;
			writer = new FileWriter(savePath);
			writer.write(generateDAG(nVariables[i], nVariables[i] * (nVariables[i] - 1) / 2, null));
			writer.close();
			System.out.println(generateDAG(nVariables[i], nVariables[i] * (nVariables[i] - 1) / 2, null)); 
		}
	}
	
	
	public static void main(String[] args) throws Exception {
//		int seed = 0;
//		GraphGenerator gg = new GraphGenerator(seed);
//		gg.genRandomDAG(seed);
	}

}
