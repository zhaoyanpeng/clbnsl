package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.evaluation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;

/**
 * @author Yanpeng Zhao
 * 3/25/2015
 */
public class EvaToolkit {
	
	private final static String[] names = AUtils.nameOfDS;
	private int seed = 0;
	
	/**
	 * We experiment on five different datasets for each network and report the averaged results.
	 * And a dataset contains six different data scale <tt>100, 500, 1000, 5000, 10000, 50000</tt>. <p>
	 * 
	 * In the generated <tt>../name/name< * >AVE.TXT</tt>, each row corresponds to each dataset of the network,
	 * and each column corresponds to each data scale of the dataset. <p>
	 * 
	 * @throws Exception
	 */
	public void averageDistribution() throws Exception {
		int itsESS = (int) AUtils.ESS_CL;
		int iTest = 0;
    	int nTest = 10;
    	int iSeed = 0;
    	int nSeed = 5;
		
		String name = "";
		
		String evafile = "";
		String avefile = "";
		String line    = "";	
		
		String basePath = "E:/NewData"; // root
		
		String algid = "_20160311_";    // flag
		String prefix = "";
		
		int row = 4, col = 4;           // store temporary records, each row corresponds to each dataset of the network
		row = 6;                        // each column corresponds to each data scale of the dataset.
		
		// overwrite
//		algid = "_MMHC_";
//		prefix = "_B";
//		itsESS = 10;
//		row = 7;
		
		FileWriter writer = null;
		BufferedReader reader = null;
		
		StringBuffer sb = new StringBuffer();
		
		Pattern pattern = Pattern.compile(AUtils.EVA_PATTERN);
		Matcher matcher = null;

		double[][] sumofshd = new double[row][col];
		
		for ( int i = iTest; i < nTest; i++ ) {
			// initialization
			for ( int j = 0; j < row; j++ ) {
				for ( int k = 0; k < col; k++ ) {
					sumofshd[j][k] = 0;
				}
			}
			name = names[i];
			sb.delete(0, sb.length());
			// read
			for ( int j = iSeed; j < nSeed; j++ ) {
				seed = j;
				evafile = basePath + "/res/" + name + "/seed" + String.valueOf(seed) + "/" + name + prefix + algid + String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_EVA;
				avefile = basePath + "/res/" + name + "/" + name + prefix + algid + String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_EVA_AVE;
				
				// @debug
				System.out.println("ReadPath: " + evafile);
				
				reader = new BufferedReader(new FileReader(new File(evafile)));
				for ( int k = 0; k < row; k++ ) {
					if ( (line = reader.readLine()) != null ) {
						matcher = pattern.matcher(line);
						if ( matcher.find() ) {
							sumofshd[k][0] += Double.valueOf(matcher.group(1).trim()).doubleValue();
							sumofshd[k][1] += Double.valueOf(matcher.group(2).trim()).doubleValue();
							sumofshd[k][2] += Double.valueOf(matcher.group(3).trim()).doubleValue();
							sumofshd[k][3] += Double.valueOf(matcher.group(4).trim()).doubleValue();
						} else {
							// errors
						}
					} else {
						// errors
					}
				}
			}
			
			// record of sum
			for ( int j = 0; j < row; j++ ) {
				for ( int k = 0; k < col; k++ ) {
					sb.append(sumofshd[j][k] + "\t");
				}
				sb.append("\n");
			}
			// record of ave
			sb.append("\n");
			for ( int j = 0; j < row; j++ ) {
				for ( int k = 0; k < col; k++ ) {
					sb.append(sumofshd[j][k] / (nSeed - iSeed) + "\t");
				}
				sb.append("\n");
			}
			// @debug
			System.out.println("SavePath: " + avefile);
			// write
			writer = new FileWriter(avefile);
			writer.write(sb.toString());
			writer.close();
		}
	}
	
	
	/**
	 * We experiment on five different datasets for each network and report the averaged results.
	 * And a dataset contains six different data scale <tt>100, 500, 1000, 5000, 10000, 50000</tt>. <p>
	 * 
	 * In the generated <tt>../name/name< * >AVE.TXT</tt>, each row corresponds to each dataset of the network,
	 * and each column corresponds to each data scale of the dataset. <p>
	 * 
	 * @throws Exception
	 */
	public void averageSHD() throws Exception {
		int itsESS = (int) AUtils.ESS_CL;
		int iTest = 0;
    	int nTest = 10;
    	int iSeed = 0;
    	int nSeed = 5;
		
		String name = "";
		
		String shdfile = "";
		String avefile = "";
		String line    = "";
		
		String basePath = "E:/NewData";
		
		String algid = "_20160311_";
		String prefix = "";
		
		int row = 4, col = 4, sum;      // store temporary records
		row = 6;                 
		
		// rewrite
//		algid = "_MMHC_";
//		prefix = "";
//		itsESS = 10;
//		row = 6;

		FileWriter writer = null;
		BufferedReader reader = null;
		
		StringBuffer sb = new StringBuffer();
		
		Pattern pattern = Pattern.compile(AUtils.SHD_PATTERN);
		Matcher matcher = null;
		
		
		int[][] sumofshd = new int[row][col];
		int[][] sampleshd = new int[row][nSeed - iSeed];
		double[] meanshd = new double[row];
		double[] stdshd = new double[row];
		
		for ( int i = iTest; i < nTest; i++ ) {
			// initialization
			for ( int j = 0; j < row; j++ ) {
				for ( int k = 0; k < col; k++ ) {
					sumofshd[j][k] = 0;
				}
			}
			name = names[i];
			sb.delete(0, sb.length());
			// read
			for ( int j = iSeed; j < nSeed; j++ ) {
				seed = j;
				shdfile = basePath + "/res/" + name + "/seed" + String.valueOf(seed) + "/" + name + prefix + algid + String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_SHD;
				avefile = basePath + "/res/" + name + "/" + name + prefix + algid + String.valueOf(itsESS) + "_" + AUtils.FSUFFIX_SHD_AVE;
				
				System.out.println("ReadPath: " + shdfile); // @debug
				
				reader = new BufferedReader(new FileReader(new File(shdfile)));
				for ( int k = 0; k < row; k++ ) {
					if ( (line = reader.readLine()) != null ) {
						matcher = pattern.matcher(line);
						if ( matcher.find() ) {
							sumofshd[k][0] += Integer.valueOf(matcher.group(1).trim()).intValue();
							sumofshd[k][1] += Integer.valueOf(matcher.group(2).trim()).intValue();
							sumofshd[k][2] += Integer.valueOf(matcher.group(3).trim()).intValue();
							sumofshd[k][3] += Integer.valueOf(matcher.group(4).trim()).intValue();
							
							sampleshd[k][j] = Integer.valueOf(matcher.group(1).trim()).intValue() + 
									Integer.valueOf(matcher.group(2).trim()).intValue() + 
									Integer.valueOf(matcher.group(4).trim()).intValue();
						} else {
							// errors
						}
					} else {
						// errors
					}
				}
			}
			
			// record of sum
			for ( int j = 0; j < row; j++ ) {
				sum = 0;
				for ( int k = 0; k < col; k++ ) {
					sb.append(sumofshd[j][k] + "\t");
					
					if ( col >= 2 && k != col - 2 ) {
						sum += sumofshd[j][k];
					}
				}
				sb.append(sum + "\n");
			}
			// record of mean
			sb.append("\n");
			for ( int j = 0; j < row; j++ ) {
				sum = 0;
				for ( int k = 0; k < col; k++ ) {
					sb.append(sumofshd[j][k] / (double) (nSeed - iSeed) + "\t");
					
					if ( col >= 2 && k != col - 2 ) {
						sum += sumofshd[j][k];
					}
				}
				// mean shd
				meanshd[j] = sum / (double) (nSeed - iSeed);
				sb.append(meanshd[j] + "\n");
			}
			// calculate std
			sb.append("\n");
			for ( int j = 0; j < row; j++ ) {
				for ( int k = 0; k < nSeed - iSeed; k++ ) {
					sb.append(sampleshd[j][k] + "\t");
				}
				sb.append("\n");
			}
			sb.append("\n");
			double variance = 0;
			for ( int j = 0; j < row; j++ ) {
				variance = 0;
				for ( int k = 0; k < nSeed - iSeed; k++ ) {
					variance += Math.pow((sampleshd[j][k] - meanshd[j]), 2);
				}
				stdshd[j] = Math.sqrt(variance / (nSeed - iSeed - 1));
				sb.append(stdshd[j] + "\t");
			}
			
			// @debug
			System.out.println("SavePath: " + avefile);
						
			// write
			writer = new FileWriter(avefile);
			writer.write(sb.toString());
			writer.close();
		}
	}

	
	/**
	 * The frequencies of step size at which the optimal structure is obtained.
	 * 
	 * @throws Exception
	 */
	public void analyseOptimalStep() throws Exception {
		int ess = (int) AUtils.ESS_CL;
		int iTest = 0;
    	int nTest = 10;
		
		String name = "";
		String stepFile = "";
		String stepStatistic = "";
		
		String basePath = "E:/NewData/"; // root 
		String algid    = "_20160311_";     // flag
		String prefix   = "";
		
		FileWriter writer = null;
		BufferedReader reader = null;
		StringBuffer sb = new StringBuffer();
		
		String line = "";
		String[] steps = null;
		
		int i0t = 0, i1t = 0, i2t = 0, i3t = 0;
		int i0, i1, i2, i3, istep = -1, tstep = 0;
		
		for ( int i = iTest; i < nTest; i++ ) {
			name = names[i];
			stepFile = basePath + "res/" + name + "/" + name + prefix + algid + String.valueOf(ess) + "_" + AUtils.FSUFFIX_BESTEP;
			reader = new BufferedReader(new FileReader(new File(stepFile)));
			
			i0 = 0;
			i1 = 0;
			i2 = 0;
			i3 = 0;
			
			while ( (line = reader.readLine()) != null ) {
				steps = line.split(" ");
				if ( steps != null ) {
					for ( int j = 0; j < steps.length; j++ ) {
						istep = Integer.parseInt(steps[j]);
						if ( istep == 0 ) {
							i0++;
						} else if ( istep == 1 ) {
							i1++;
						} else if ( istep == 2 ) {
							i2++;
						} else if ( istep == 3 ) {
							i3++;
						}
					}
				} else { System.err.println("Step file error: " + stepFile); }
			}
			reader.close();
			
			i0t += i0;
			i1t += i1;
			i2t += i2;
			i3t += i3;
			
			sb.append(name + " & " + i0 + " & " + i1 + " & " + i2 + " & " + i3 + "\n");
		}
		tstep = i0t + i1t + i2t + i3t;
		sb.append("total" + " & " + i0t + " " + i1t + " " + i2t + " " + i3t + "\n");
		sb.append("ratio" + " & " + ((double) i0t / tstep) + " " + ((double) i1t / tstep) + " " + ((double) i2t / tstep) + " " + ((double) i3t / tstep) + "\n");
		
		stepStatistic = basePath + "res/step_statistic.txt";
		writer = new FileWriter(stepStatistic);
		writer.write(sb.toString());
		writer.close();
	}
	

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		EvaToolkit et = new EvaToolkit();
//		et.averageSHD();
//		et.averageDistribution();
		
//		et.analyseOptimalStep();
	}
	
}
