package edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.search.local;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;

import edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils.AUtils;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.BayesNet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.ParentSet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.search.SearchAlgorithm;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instance;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Statistics;

/**
 * Modified from Weka.
 * 
 * @author Yanpeng Zhao
 */
@SuppressWarnings("unused")
public class LocalScoreSearchAlgorithm extends SearchAlgorithm {

	// penalty of the score
	private double penaltyFactor = 0.0;
	private boolean bUsePenalty = false;
	private double tempss = 0.0;

	private int howToMakeScore = AUtils.SCORE_DEFAULT;
	private double ess = AUtils.ESS_CL;

	// variables used for debugging time-consuming
	protected double stime = 0;
	protected double etime = 0;
	protected String method = "";

	protected int stage = 0;
	protected ArrayList<ArrayList<Integer>> indexPartitions;
	protected ArrayList<ArrayList<Integer>> fullConnectedDAG;

	protected void timeDetail(String description) {
		System.out.println("!->" + description + " "
				+ (this.etime - this.stime) / (double) 1000 + "s.");
		if ((this.etime - this.stime) > 5e3) {
			System.out.println("(->" + description + " consume more than 5s.");
		}
	}

	public void setPCSets(Vector<Vector<Integer>> pcSets) {
	}

	public void setFullConnectedDAG(ArrayList<ArrayList<Integer>> fcDAG) {
	}

	public void setPenaltyFactor(double penaltyFactor) {
		this.penaltyFactor = penaltyFactor;
	}

	public double getPenaltyFactor() {
		return this.penaltyFactor;
	}

	public void setHowToMakeScore(int howToMakeScore) {
		this.howToMakeScore = howToMakeScore;
	}

	public void setBUsePenalty(boolean bUsePenalty) {
		this.bUsePenalty = bUsePenalty;
	}

	public void setMaxNumParent(int maxNumParent) {
		this.m_nMaxNrOfParents = maxNumParent;
	}

	@SuppressWarnings("static-access")
	public double calcNodeScoreCurricula(int iNode) {
		Instances instances = m_BayesNet.m_Instances;
		int nPartition = instances.getNumPattern();

		switch (howToMakeScore) {
		case AUtils.SCORE_DEFAULT: {

			// update ess
			this.tempss = this.ess;
			if (nPartition != 0) {
				this.tempss = this.ess / nPartition;
			}

			// penalty factor
			this.penaltyFactor = (double) 1000 / instances.size()
					+ instances.numAttributes() / (double) 100;

			return calcNodeScorePlainCurricula(instances, iNode, nPartition);
		}
		case AUtils.SCORE_ADD: {

			double score0 = calcNodeScorePlainCurricula(instances, iNode,
					nPartition);

			this.tempss = this.ess;
			double score1 = calcNodeScorePlainCurricula(instances, iNode, 0);

			System.out.println(score0 + "\t" + score1);
			try {
				Thread.currentThread().sleep(200);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return (score0 + score1);
		}
		case AUtils.SCORE_NORM_1: {
			this.tempss = this.ess;
			double score1 = calcNodeScorePlainCurricula(instances, iNode, 0);
			return score1;
		}
		case AUtils.SCORE_NORM_2: {
			return -(calcNodeScorePlainCurricula(instances, iNode, nPartition) + calcNodeScorePlainCurricula(
					instances, iNode, 0));
		}
		case AUtils.SCORE_EMPTY: {
			this.tempss = this.ess;
			this.penaltyFactor = (double) 1000 / instances.size()
					+ instances.numAttributes() / (double) 100; // pv
			return calcNodeScorePlainCurricula(instances, iNode, 0);
			// to do
		}
		default: {
			return 0;
		}
		}

	}

	private double calcNodeScorePlainCurricula(Instances instances, int iNode,
			int nPartition) {

		ParentSet oParentSet = m_BayesNet.getParentSet(iNode);

		// determine cardinality of parent set & reserve space for frequency
		// counts
		int nCardinality = oParentSet.getCardinalityOfParents();
		int numValues = instances.attribute(iNode).numValues();

		if (nPartition > 0) {

			return withPartition(instances, oParentSet, iNode, numValues,
					nPartition, nCardinality);
		} else {

			return withoutPartition(instances, oParentSet, iNode, numValues,
					nCardinality);
		}

	}

	protected double withPartition(Instances instances, ParentSet oParentSet,
			int iNode, int numValues, int nPartition, int nCardinality) {

		Integer key = null;
		Instance instance = null;
		double score = 0, scoreTmp = 0, iCPT;
		ArrayList<Integer> indexPartition = null;

		HashMap<Integer, Integer> counts = new HashMap<Integer, Integer>();

		// stime = System.currentTimeMillis();

		for (int i = 0; i < nPartition; i++) {

			counts.clear();
			indexPartition = indexPartitions.get(i);
			for (int j = 0; j < indexPartition.size(); j++) {

				iCPT = 0;
				instance = instances.get(indexPartition.get(j).intValue());
				for (int ii = 0; ii < oParentSet.getNrOfParents(); ii++) {
					int nParent = oParentSet.getParent(ii);
					iCPT = iCPT * instances.attribute(nParent).numValues()
							+ instance.value(nParent);
				}

				key = new Integer(numValues * ((int) iCPT)
						+ (int) instance.value(iNode));
				if (counts.containsKey(key)) {
					counts.put(key, new Integer(counts.get(key).intValue() + 1));
				} else {
					counts.put(key, new Integer(1));
				}
			}

			// scoreTmp = calcScoreOfCountsCurricula(counts, nCardinality,
			// numValues, instances, true);
			scoreTmp = calcScoreOfCountsCurricula(counts, nCardinality,
					numValues, instances, 0.01);
			score += scoreTmp;
		}

		// etime = System.currentTimeMillis();
		// System.out.print(nCardinality + "\t");
		// this.timeDetail(method);

		counts.clear();
		return (score - this.penaltyFactor * oParentSet.getNrOfParents());
	}

	protected double withoutPartition(Instances instances,
			ParentSet oParentSet, int iNode, int numValues, int nCardinality) {

		Integer key = null;
		Instance instance = null;
		double score = 0, iCPT;

		HashMap<Integer, Integer> counts = new HashMap<Integer, Integer>();

		counts.clear();
		Enumeration<Instance> enumInsts = instances.enumerateInstances();

		while (enumInsts.hasMoreElements()) {
			iCPT = 0;
			instance = enumInsts.nextElement();
			for (int iParent = 0; iParent < oParentSet.getNrOfParents(); iParent++) {
				int nParent = oParentSet.getParent(iParent);
				iCPT = iCPT * instances.attribute(nParent).numValues()
						+ instance.value(nParent);
			}

			key = new Integer(numValues * ((int) iCPT)
					+ (int) instance.value(iNode));
			if (counts.containsKey(key)) {
				counts.put(key, new Integer(counts.get(key).intValue() + 1));
			} else {
				counts.put(key, new Integer(1));
			}
		}
		score = calcScoreOfCountsCurricula(counts, nCardinality, numValues,
				instances, 0.01);

		counts.clear();
		return (score - this.penaltyFactor * oParentSet.getNrOfParents());
	}

	/**
	 * Deep optimization of the computing of BDeu score. Refer to
	 * {@code weka.bnsl.bayes.net.estimate.BayesNetEstimator.calcScoreBDeu(...)}
	 * for the original version.
	 * 
	 */
	protected double calcScoreOfCountsCurricula(
			HashMap<Integer, Integer> counts, int nCardinality, int numValues,
			Instances instances, double deepOptimization) {

		double fLogScore = 0.0, nSumOfCounts;
		int number, nZero = 0, nZero1 = 0, count, nij, nijk;

		boolean allZero;
		ArrayList<Integer> Nij = new ArrayList<Integer>();
		ArrayList<Integer> Nijk = new ArrayList<Integer>();
		for (int i = 0; i < nCardinality; i++) {

			count = 0;
			allZero = true;
			for (int j = 0; j < numValues; j++) {
				if (counts.containsKey(new Integer(i * numValues + j))) {
					number = counts.get(new Integer(i * numValues + j))
							.intValue();

					Nijk.add(new Integer(number));
					count += number;

					allZero = false;
				} else {
					nZero1++;
				}
			}

			if (allZero) {
				nZero++;
			} else {
				Nij.add(new Integer(count));
			}
		}

		// try to reduce the call frequency of the gamma function that consumes
		// much time
		for (int i = 0; i < Nijk.size(); i++) {
			nijk = Nijk.get(i);
			fLogScore += Statistics.lnGamma(tempss / (numValues * nCardinality)
					+ nijk);
		}

		for (int i = 0; i < Nij.size(); i++) {
			nij = Nij.get(i);
			fLogScore -= Statistics.lnGamma(tempss / nCardinality + nij);
		}

		fLogScore += (nZero1 - nCardinality * numValues)
				* Statistics.lnGamma(tempss / (numValues * nCardinality));
		fLogScore += (nCardinality - nZero)
				* Statistics.lnGamma(tempss / nCardinality);

		Nij.clear();
		Nijk.clear();

		return fLogScore;
	} // CalcNodeScore

	/**
	 * Shallow optimization of the computing of BDeu score. Refer to
	 * {@code weka.bnsl.bayes.net.estimate.BayesNetEstimator.calcScoreBDeu(...)}
	 * for the original version.
	 * 
	 */
	protected double calcScoreOfCountsCurricula(
			HashMap<Integer, Integer> counts, int nCardinality, int numValues,
			Instances instances, boolean shallowOptimization) {

		int number, nZero = 0, j, itemp;
		double fLogScore = 0.0, nSumOfCounts;

		ArrayList<Integer> filterCardinality = new ArrayList<Integer>();
		for (int i = 0; i < nCardinality; i++) {

			for (j = 0; j < numValues; j++) {
				if (counts.containsKey(new Integer(i * numValues + j))) {
					break;
				}
			}

			if (j == numValues) {
				nZero++;
			} else {
				filterCardinality.add(new Integer(i));
			}
		}

		int nZero1 = 0;
		for (int i = 0; i < filterCardinality.size(); i++) {

			nSumOfCounts = 0;
			itemp = filterCardinality.get(i).intValue();
			for (j = 0; j < numValues; j++) {
				number = counts.containsKey(new Integer(itemp * numValues + j)) ? counts
						.get(new Integer(itemp * numValues + j)).intValue() : 0;

				if (number == 0) {
					nZero1++;
				} else {
					fLogScore += Statistics.lnGamma(tempss
							/ (numValues * nCardinality) + number);
				}

				nSumOfCounts += tempss / (numValues * nCardinality) + number;
			}
			fLogScore -= Statistics.lnGamma(nSumOfCounts);
		}

		fLogScore += ((nZero - nCardinality) * numValues + nZero1)
				* Statistics.lnGamma(tempss / (numValues * nCardinality));
		fLogScore += (nCardinality - nZero)
				* Statistics.lnGamma(tempss / nCardinality);

		filterCardinality.clear();
		return fLogScore;
	} // CalcNodeScore

	public double calcScoreWithExtraParentCurricula(int iNode,
			int nCandidateParent) {

		ParentSet oParentSet = m_BayesNet.getParentSet(iNode);

		// sanity check: nCandidateParent should not be in parent set already
		if (oParentSet.contains(nCandidateParent)) {
			return -1e100;
		}

		// set up candidate parent
		oParentSet.addParent(nCandidateParent, m_BayesNet.m_Instances);

		// stime = System.currentTimeMillis();
		this.method = "Extra";

		double logScore = calcNodeScoreCurricula(iNode);

		// etime = System.currentTimeMillis();
		// this.timeDetail("Extra");

		// delete temporarily added parent
		oParentSet.deleteLastParent(m_BayesNet.m_Instances);

		return logScore;
	} // CalcScoreWithExtraParent

	public double calcScoreWithMissingParentCurricula(int iNode,
			int nCandidateParent) {

		ParentSet oParentSet = m_BayesNet.getParentSet(iNode);

		// sanity check: nCandidateParent should be in parent set already
		if (!oParentSet.contains(nCandidateParent)) {
			return -1e100;
		}

		// set up candidate parent
		int iParent = oParentSet.deleteParent(nCandidateParent,
				m_BayesNet.m_Instances);

		// stime = System.currentTimeMillis();
		this.method = "Missing";

		double logScore = calcNodeScoreCurricula(iNode);

		// etime = System.currentTimeMillis();
		// this.timeDetail("Missing");

		// restore temporarily deleted parent
		oParentSet.addParent(nCandidateParent, iParent, m_BayesNet.m_Instances);

		return logScore;
	} // CalcScoreWithMissingParent

	public boolean beWithExtraParent(int iNode, int nCandidateParent) {
		Instances instances = m_BayesNet.m_Instances;
		ParentSet oParentSet = m_BayesNet.getParentSet(iNode);

		long boundary = (long) oParentSet.getCardinalityOfParents()
				* instances.attribute(iNode).numValues()
				* instances.attribute(nCandidateParent).numValues();
		if (boundary >= Integer.MAX_VALUE) {
			// @debugSystem.out.println("%->directly set it to 0.");
			System.out.println("%->cardinality "
					+ oParentSet.getCardinalityOfParents() + "( "
					+ oParentSet.getNrOfParents() + ")\t>=\t"
					+ Integer.MAX_VALUE + "(int.max)\tdirectly set it to 0.");
			return false;
		} else {
			return true;
		}
	}

	protected BayesNet m_BayesNet;

	public LocalScoreSearchAlgorithm() {
	} // c'tor

	/**
	 * Determine the network structure/graph of the network
	 * 
	 * @param bayesNet
	 *            the network
	 * @param instances
	 *            the data to be used
	 * @throws Exception
	 * 
	 */
	@Override
	public void buildStructure(BayesNet bayesNet, Instances instances)
			throws Exception {
		m_BayesNet = bayesNet;
		super.buildStructure(bayesNet, instances);
	}

}
