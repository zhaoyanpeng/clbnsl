package edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes;

import java.util.ArrayList;

import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.ParentSet;
import edu.shanghaitech.ai.ml.clbnsl.weka.classifiers.bayes.net.search.SearchAlgorithm;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * Modified from Weka.
 * 
 * @author Yanpeng Zhao
 */
public class BayesNet {

	public Instances m_Instances;
	private ParentSet[] m_ParentSets;
	private SearchAlgorithm m_SearchAlgorithm = null;

	private double timeBoundary = 0.0;

	public void buildClassifier(Instances instances) throws Exception {

		// remove instances with missing class
		instances = new Instances(instances);

		// Copy the instances
		m_Instances = new Instances(instances);

		// build the network structure
		initStructure();

		// build the network structure
		buildStructure();

		// Save space
		m_Instances = new Instances(m_Instances, 0);
	}

	public void initStructure() throws Exception {
		// reserve memory
		m_ParentSets = new ParentSet[m_Instances.numAttributes()];
		for (int iAttribute = 0; iAttribute < m_Instances.numAttributes(); iAttribute++) {
			m_ParentSets[iAttribute] = new ParentSet(
					m_Instances.numAttributes());
		}
	}

	public void buildStructure() throws Exception {
		m_SearchAlgorithm.buildStructure(this, m_Instances);
	}

	public void setTimeBoundary(double timeBoundary) {
		this.timeBoundary = timeBoundary;
	}

	public double getTimeBoundary() {
		return this.timeBoundary;
	}

	public void setSearchAlgorithm(SearchAlgorithm newSearchAlgorithm) {
		m_SearchAlgorithm = newSearchAlgorithm;
	}

	public int getNNode() {
		return m_Instances.numAttributes();
	}

	public ParentSet getParentSet(int iNode) {
		return m_ParentSets[iNode];
	}

	public void updateSearchAlgorithm(int stepLength, int[] listOfCL) {
		this.m_SearchAlgorithm.updateCL(stepLength, listOfCL);
	}

	public void updateSearchAlgorithm(int stepLength) {
		this.m_SearchAlgorithm.setStepSize(stepLength);
	}

	protected void copy2DArrayList(ArrayList<ArrayList<Integer>> des,
			ArrayList<ArrayList<Integer>> src) {
		for (int i = 0; i < src.size(); i++) {
			ArrayList<Integer> list = new ArrayList<Integer>();
			for (Integer iNode : src.get(i)) {
				list.add(iNode);
			}
			des.add(list);
		}
	}

	/**
	 * Given a complete directed acyclic graph, represented by the pairs of (n,
	 * P_{n}), we first empty the parent sets of the nodes also existing in the
	 * sub-net learned by our algorithm, and then re-initialize them to the
	 * learned sub-net. Now we have two separated directed acyclic graphs, the
	 * next is adding all the nodes in the unmodified part to the parent sets of
	 * the nodes in the learned sub-net.
	 * <p>
	 * 
	 * We can prove the procedure described above wouldn't produce cycles. First
	 * there couldn't be cycles in two separated directed acyclic graphs. Assume
	 * there is a cycle after the above operation (empty, replace, add...). Then
	 * the cycle must result from the added directed edges, which implies the
	 * added directed edges give rise to the cycle. But the added directed edges
	 * could only start from the unmodified part to the learned sub-net, so it
	 * couldn't produce the cycle.
	 * 
	 */
	public String toString(int[] listOfCL, int end,
			ArrayList<ArrayList<Integer>> fullConnectedDAG) {

		ArrayList<ArrayList<Integer>> copyFullConnectedDAG = new ArrayList<ArrayList<Integer>>();
		// make a new network
		ArrayList<Integer> list = null;
		copy2DArrayList(copyFullConnectedDAG, fullConnectedDAG);
		for (int i = 0; i < end; i++) {
			int iNode = listOfCL[i];
			ParentSet parentSet = m_ParentSets[iNode];

			list = copyFullConnectedDAG.get(iNode);
			list.clear();
			for (int j = 0; j < parentSet.getNrOfParents(); j++) {
				list.add(new Integer(parentSet.getParent(j)));
			}
			/* System.out.print(iNode + "\t|"); */
			/* System.out.println(list); */
			for (int j = end; j < listOfCL.length; j++) {
				list.add(new Integer(listOfCL[j]));
			}
			/* System.out.println("One----->" + list); */
			for (int j = end; j < listOfCL.length; j++) {
				int jNode = listOfCL[j];
				list = copyFullConnectedDAG.get(jNode);
				if (list.contains(new Integer(iNode))) {
					list.remove(new Integer(iNode));
				}
			}
		}

		// return as string
		StringBuffer sb = new StringBuffer();
		sb.append(listOfCL.length + "\n");
		for (int i = 0; i < listOfCL.length; i++) {
			list = copyFullConnectedDAG.get(i);
			sb.append(i + " " + list.size() + " ");
			for (Integer node : list) {
				sb.append(node.intValue() + " ");
			}
			sb.append("\n");
		}
		return sb.toString();
	}

	public String toString() {

		StringBuffer text = new StringBuffer();

		if (m_Instances == null) {
			text.append(": No model built yet.");
		} else {
			// flatten BayesNet down to text
			text.append(m_Instances.numAttributes() + "\n");

			for (int i = 0; i < m_Instances.numAttributes(); i++) {

				ParentSet parentSet = m_ParentSets[i];
				int numParent = parentSet.getNrOfParents();

				text.append(i + " " + numParent + " ");
				for (int j = 0; j < numParent; j++) {

					text.append(parentSet.getParent(j) + " ");
				}
				text.append("\n");
			}
		}
		return text.toString();
	}
}