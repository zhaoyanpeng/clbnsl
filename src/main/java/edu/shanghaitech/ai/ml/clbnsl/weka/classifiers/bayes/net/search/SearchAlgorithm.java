package edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.search;

import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.BayesNet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.ParentSet;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * Modified from Weka.
 * 
 * @author Yanpeng Zhao
 */
public class SearchAlgorithm {

	protected int m_nMaxNrOfParents = 1;

	public SearchAlgorithm() {
	}

	/**
	 * AddArcMakesSense checks whether adding the arc from iAttributeTail to
	 * iAttributeHead does not already exists and does not introduce a cycle
	 * 
	 * @param bayesNet
	 * @param instances
	 * @param iAttributeHead
	 *            index of the attribute that becomes head of the arrow
	 * @param iAttributeTail
	 *            index of the attribute that becomes tail of the arrow
	 * @return true if adding arc is allowed, otherwise false
	 */
	protected boolean addArcMakesSense(BayesNet bayesNet, Instances instances,
			int iAttributeHead, int iAttributeTail) {
		if (iAttributeHead == iAttributeTail) {
			return false;
		}

		// sanity check: arc should not be in parent set already
		if (isArc(bayesNet, iAttributeHead, iAttributeTail)) {
			return false;
		}

		// sanity check: arc should not introduce a cycle
		int nNodes = instances.numAttributes();
		boolean[] bDone = new boolean[nNodes];

		for (int iNode = 0; iNode < nNodes; iNode++) {
			bDone[iNode] = false;
		}

		// check for cycles
		bayesNet.getParentSet(iAttributeHead).addParent(iAttributeTail,
				instances);

		for (int iNode = 0; iNode < nNodes; iNode++) {

			// find a node for which all parents are 'done'
			boolean bFound = false;

			for (int iNode2 = 0; !bFound && iNode2 < nNodes; iNode2++) {
				if (!bDone[iNode2]) {
					boolean bHasNoParents = true;

					for (int iParent = 0; iParent < bayesNet.getParentSet(
							iNode2).getNrOfParents(); iParent++) {
						if (!bDone[bayesNet.getParentSet(iNode2).getParent(
								iParent)]) {
							bHasNoParents = false;
						}
					}

					if (bHasNoParents) {
						bDone[iNode2] = true;
						bFound = true;
					}
				}
			}

			if (!bFound) {
				bayesNet.getParentSet(iAttributeHead).deleteLastParent(
						instances);
				return false;
			}
		}

		bayesNet.getParentSet(iAttributeHead).deleteLastParent(instances);

		return true;
	}

	/**
	 * reverseArcMakesSense checks whether the arc from iAttributeTail to
	 * iAttributeHead exists and reversing does not introduce a cycle
	 * 
	 * @param bayesNet
	 * @param instances
	 * @param iAttributeHead
	 *            index of the attribute that is head of the arrow
	 * @param iAttributeTail
	 *            index of the attribute that is tail of the arrow
	 * @return true if the arc from iAttributeTail to iAttributeHead exists and
	 *         reversing does not introduce a cycle
	 */
	protected boolean reverseArcMakesSense(BayesNet bayesNet,
			Instances instances, int iAttributeHead, int iAttributeTail) {

		if (iAttributeHead == iAttributeTail) {
			return false;
		}

		// sanity check: arc should be in parent set already
		if (!isArc(bayesNet, iAttributeHead, iAttributeTail)) {
			return false;
		}

		// sanity check: arc should not introduce a cycle
		int nNodes = instances.numAttributes();
		boolean[] bDone = new boolean[nNodes];

		for (int iNode = 0; iNode < nNodes; iNode++) {
			bDone[iNode] = false;
		}

		// check for cycles
		bayesNet.getParentSet(iAttributeTail).addParent(iAttributeHead,
				instances);

		for (int iNode = 0; iNode < nNodes; iNode++) {

			// find a node for which all parents are 'done'
			boolean bFound = false;

			for (int iNode2 = 0; !bFound && iNode2 < nNodes; iNode2++) {
				if (!bDone[iNode2]) {
					ParentSet parentSet = bayesNet.getParentSet(iNode2);
					boolean bHasNoParents = true;
					for (int iParent = 0; iParent < parentSet.getNrOfParents(); iParent++) {
						if (!bDone[parentSet.getParent(iParent)]) {

							// this one has a parent which is not 'done' UNLESS
							// it is the arc to be reversed
							if (!(iNode2 == iAttributeHead && parentSet
									.getParent(iParent) == iAttributeTail)) {
								bHasNoParents = false;
							}
						}
					}

					if (bHasNoParents) {
						bDone[iNode2] = true;
						bFound = true;
					}
				}
			}

			if (!bFound) {
				bayesNet.getParentSet(iAttributeTail).deleteLastParent(
						instances);
				return false;
			}
		}

		bayesNet.getParentSet(iAttributeTail).deleteLastParent(instances);
		return true;
	}

	/**
	 * IsArc checks whether the arc from iAttributeTail to iAttributeHead
	 * already exists
	 * 
	 * @param bayesNet
	 * @param iAttributeHead
	 *            index of the attribute that becomes head of the arrow
	 * @param iAttributeTail
	 *            iAttributeTail index of the attribute that becomes tail of the
	 *            arrow
	 * @return true if the arc from iAttributeTail to iAttributeHead already
	 *         exists
	 */
	protected boolean isArc(BayesNet bayesNet, int iAttributeHead,
			int iAttributeTail) {
		for (int iParent = 0; iParent < bayesNet.getParentSet(iAttributeHead)
				.getNrOfParents(); iParent++) {
			if (bayesNet.getParentSet(iAttributeHead).getParent(iParent) == iAttributeTail) {
				return true;
			}
		}
		return false;
	}

	/**
	 * buildStructure determines the network structure/graph of the network. The
	 * default behavior is creating a network where all nodes have the first
	 * node as its parent (i.e., a BayesNet that behaves like a naive Bayes
	 * classifier). This method can be overridden by derived classes to restrict
	 * the class of network structures that are acceptable.
	 * 
	 * @param bayesNet
	 *            the network
	 * @param instances
	 *            the data to use
	 */
	public void buildStructure(BayesNet bayesNet, Instances instances)
			throws Exception {
		search(bayesNet, instances);
	}

	protected void search(BayesNet bayesNet, Instances instances)
			throws Exception {
		// placeholder with implementation in derived classes
	}

	/**
	 * Update step length of curriculum learning.
	 * 
	 */
	public void updateCL(int stepLength, int[] listOfCL) {
		// placeholder with implementation in derived classes
	}

	public void setStepSize(int stepSize) {
		// placeholder with implementation in derived classes
	}

}
