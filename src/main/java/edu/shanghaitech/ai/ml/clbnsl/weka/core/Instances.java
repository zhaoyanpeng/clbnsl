/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Instances.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package edu.shanghaitech.ai.ml.clbnsl.weka.core;

import java.io.IOException;
import java.io.Reader;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import edu.shanghaitech.ai.ml.clbnsl.weka.core.converters.ArffLoader.ArffReader;

/**
 * Class for handling an ordered set of weighted instances.
 * <p>
 * 
 * Typical usage:
 * <p>
 * 
 * <pre>
 * import weka.core.converters.ConverterUtils.DataSource;
 * ...
 * 
 * // Read all the instances in the file (ARFF, CSV, XRFF, ...)
 * DataSource source = new DataSource(filename);
 * Instances instances = source.getDataSet();
 * 
 * // Make the last attribute be the class
 * instances.setClassIndex(instances.numAttributes() - 1);
 * 
 * // Print header and instances.
 * System.out.println("\nDataset:\n");
 * System.out.println(instances);
 * 
 * ...
 * </pre>
 * <p>
 * 
 * All methods that change a set of instances are safe, ie. a change of a set of
 * instances does not affect any other sets of instances. All methods that
 * change a datasets's attribute information clone the dataset before it is
 * changed.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 11342 $
 */
public class Instances extends AbstractList<Instance> {

	/**********************************************************************************************
	 * @author Yanpeng Zhao
	 *********************************************************************************************/

	protected int numPattern = 0;

	public void setNumPattern(int numPattern) {
		this.numPattern = numPattern;
	}

	public int getNumPattern() {
		return numPattern;
	}

	/**********************************************************************************************
	 * @author Yanpeng Zhao
	 *********************************************************************************************/

	/** The filename extension that should be used for arff files */
	public final static String FILE_EXTENSION = ".arff";

	/**
	 * The filename extension that should be used for bin. serialized instances
	 * files
	 */
	public final static String SERIALIZED_OBJ_FILE_EXTENSION = ".bsi";

	/** The keyword used to denote the start of an arff header */
	public final static String ARFF_RELATION = "@relation";

	/** The keyword used to denote the start of the arff data section */
	public final static String ARFF_DATA = "@data";

	/** The dataset's name. */
	protected/* @spec_public non_null@ */String m_RelationName;

	/** The attribute information. */
	protected/* @spec_public non_null@ */ArrayList<Attribute> m_Attributes;
	/*
	 * public invariant (\forall int i; 0 <= i && i < m_Attributes.size();
	 * m_Attributes.get(i) != null);
	 */

	/** A map to quickly find attribute indices based on their names. */
	protected HashMap<String, Integer> m_NamesToAttributeIndices;

	/** The instances. */
	protected/* @spec_public non_null@ */ArrayList<Instance> m_Instances;

	/** The class attribute's index */
	protected int m_ClassIndex;
	// @ protected invariant classIndex() == m_ClassIndex;

	/**
	 * The lines read so far in case of incremental loading. Since the
	 * StreamTokenizer will be re-initialized with every instance that is read,
	 * we have to keep track of the number of lines read so far.
	 * 
	 * @see #readInstance(Reader)
	 */
	protected int m_Lines = 0;

	/**
	 * Reads an ARFF file from a reader, and assigns a weight of one to each
	 * instance. Lets the index of the class attribute be undefined (negative).
	 * 
	 * @param reader
	 *            the reader
	 * @throws IOException
	 *             if the ARFF file is not read successfully
	 */
	public Instances(/* @non_null@ */Reader reader) throws IOException {
		ArffReader arff = new ArffReader(reader, 1000, false);
		initialize(arff.getData(), 1000);
		arff.setRetainStringValues(true);
		Instance inst;
		while ((inst = arff.readInstance(this)) != null) {
			m_Instances.add(inst);
		}
		compactify();
	}

	/**
	 * Reads the header of an ARFF file from a reader and reserves space for the
	 * given number of instances. Lets the class index be undefined (negative).
	 * 
	 * @param reader
	 *            the reader
	 * @param capacity
	 *            the capacity
	 * @throws IllegalArgumentException
	 *             if the header is not read successfully or the capacity is
	 *             negative.
	 * @throws IOException
	 *             if there is a problem with the reader.
	 * @deprecated instead of using this method in conjunction with the
	 *             <code>readInstance(Reader)</code> method, one should use the
	 *             <code>ArffLoader</code> or <code>DataSource</code> class
	 *             instead.
	 * @see weka.core.converters.ArffLoader
	 * @see weka.core.converters.ConverterUtils.DataSource
	 */
	// @ requires capacity >= 0;
	// @ ensures classIndex() == -1;
	@Deprecated
	public Instances(/* @non_null@ */Reader reader, int capacity)
			throws IOException {

		ArffReader arff = new ArffReader(reader, 0);
		Instances header = arff.getStructure();
		initialize(header, capacity);
		m_Lines = arff.getLineNo();
	}

	/**
	 * Constructor copying all instances and references to the header
	 * information from the given set of instances.
	 * 
	 * @param dataset
	 *            the set to be copied
	 */
	public Instances(/* @non_null@ */Instances dataset) {

		this(dataset, dataset.numInstances());

		dataset.copyInstances(0, this, dataset.numInstances());
	}

	/**
	 * Constructor creating an empty set of instances. Copies references to the
	 * header information from the given set of instances. Sets the capacity of
	 * the set of instances to 0 if its negative.
	 * 
	 * @param dataset
	 *            the instances from which the header information is to be taken
	 * @param capacity
	 *            the capacity of the new dataset
	 */
	public Instances(/* @non_null@ */Instances dataset, int capacity) {
		initialize(dataset, capacity);
	}

	/**
	 * initializes with the header information of the given dataset and sets the
	 * capacity of the set of instances.
	 * 
	 * @param dataset
	 *            the dataset to use as template
	 * @param capacity
	 *            the number of rows to reserve
	 */
	protected void initialize(Instances dataset, int capacity) {
		if (capacity < 0) {
			capacity = 0;
		}

		// Strings only have to be "shallow" copied because
		// they can't be modified.
		m_ClassIndex = dataset.m_ClassIndex;
		m_RelationName = dataset.m_RelationName;
		m_Attributes = dataset.m_Attributes;
		m_NamesToAttributeIndices = dataset.m_NamesToAttributeIndices;
		m_Instances = new ArrayList<Instance>(capacity);
	}

	/**
	 * Creates a new set of instances by copying a subset of another set.
	 * 
	 * @param source
	 *            the set of instances from which a subset is to be created
	 * @param first
	 *            the index of the first instance to be copied
	 * @param toCopy
	 *            the number of instances to be copied
	 * @throws IllegalArgumentException
	 *             if first and toCopy are out of range
	 */
	// @ requires 0 <= first;
	// @ requires 0 <= toCopy;
	// @ requires first + toCopy <= source.numInstances();
	public Instances(/* @non_null@ */Instances source, int first, int toCopy) {

		this(source, toCopy);

		if ((first < 0) || ((first + toCopy) > source.numInstances())) {
			throw new IllegalArgumentException(
					"Parameters first and/or toCopy out " + "of range");
		}
		source.copyInstances(first, this, toCopy);
	}

	/**
	 * Creates an empty set of instances. Uses the given attribute information.
	 * Sets the capacity of the set of instances to 0 if its negative. Given
	 * attribute information must not be changed after this constructor has been
	 * used.
	 * 
	 * @param name
	 *            the name of the relation
	 * @param attInfo
	 *            the attribute information
	 * @param capacity
	 *            the capacity of the set
	 * @throws IllegalArgumentException
	 *             if attribute names are not unique
	 */
	public Instances(/* @non_null@ */String name,
	/* @non_null@ */ArrayList<Attribute> attInfo, int capacity) {

		// check whether the attribute names are unique
		HashSet<String> names = new HashSet<String>();
		StringBuffer nonUniqueNames = new StringBuffer();
		for (Attribute att : attInfo) {
			if (names.contains(att.name())) {
				nonUniqueNames.append("'" + att.name() + "' ");
			}
			names.add(att.name());
		}
		if (names.size() != attInfo.size()) {
			throw new IllegalArgumentException(
					"Attribute names are not unique!" + " Causes: "
							+ nonUniqueNames.toString());
		}
		names.clear();

		m_RelationName = name;
		m_ClassIndex = -1;
		m_Attributes = attInfo;
		m_NamesToAttributeIndices = new HashMap<String, Integer>(
				(int) (numAttributes() / 0.75));
		for (int i = 0; i < numAttributes(); i++) {
			attribute(i).setIndex(i);
			m_NamesToAttributeIndices.put(attribute(i).name(), i);
		}
		m_Instances = new ArrayList<Instance>(capacity);
	}

	/**
	 * Create a copy of the structure if the data has string or relational
	 * attributes, "cleanses" string types (i.e. doesn't contain references to
	 * the strings seen in the past) and all relational attributes.
	 * 
	 * @return a copy of the instance structure.
	 */
	public Instances stringFreeStructure() {

		ArrayList<Attribute> newAtts = new ArrayList<Attribute>();
		for (Attribute att : m_Attributes) {
			if (att.type() == Attribute.STRING) {
				newAtts.add(new Attribute(att.name(), (List<String>) null, att
						.index()));
			} else if (att.type() == Attribute.RELATIONAL) {
				newAtts.add(new Attribute(att.name(), new Instances(att
						.relation(), 0), att.index()));
			}
		}
		if (newAtts.size() == 0) {
			return new Instances(this, 0);
		}
		ArrayList<Attribute> atts = Utils.cast(m_Attributes.clone());
		for (Attribute att : newAtts) {
			atts.set(att.index(), att);
		}
		Instances result = new Instances(this, 0);
		result.m_Attributes = atts;
		return result;
	}

	/**
	 * Adds one instance to the end of the set. Shallow copies instance before
	 * it is added. Increases the size of the dataset if it is not large enough.
	 * Does not check if the instance is compatible with the dataset. Note:
	 * String or relational values are not transferred.
	 * 
	 * @param instance
	 *            the instance to be added
	 */
	@Override
	public boolean add(/* @non_null@ */Instance instance) {

		Instance newInstance = (Instance) instance.copy();

		newInstance.setDataset(this);
		m_Instances.add(newInstance);

		return true;
	}

	/**
	 * Adds one instance at the given position in the list. Shallow copies
	 * instance before it is added. Increases the size of the dataset if it is
	 * not large enough. Does not check if the instance is compatible with the
	 * dataset. Note: String or relational values are not transferred.
	 * 
	 * @param index
	 *            position where instance is to be inserted
	 * @param instance
	 *            the instance to be added
	 */
	// @ requires 0 <= index;
	// @ requires index < m_Instances.size();
	@Override
	public void add(int index, /* @non_null@ */Instance instance) {

		Instance newInstance = (Instance) instance.copy();

		newInstance.setDataset(this);
		m_Instances.add(index, newInstance);
	}

	/**
	 * Returns an attribute.
	 * 
	 * @param index
	 *            the attribute's index (index starts with 0)
	 * @return the attribute at the given position
	 */
	// @ requires 0 <= index;
	// @ requires index < m_Attributes.size();
	// @ ensures \result != null;
	public/* @pure@ */Attribute attribute(int index) {

		return m_Attributes.get(index);
	}

	/**
	 * Returns an attribute given its name. If there is more than one attribute
	 * with the same name, it returns the first one. Returns null if the
	 * attribute can't be found.
	 * 
	 * @param name
	 *            the attribute's name
	 * @return the attribute with the given name, null if the attribute can't be
	 *         found
	 */
	public/* @pure@ */Attribute attribute(String name) {

		Integer index = m_NamesToAttributeIndices.get(name);
		if (index != null) {
			return attribute(index);
		}

		return null;
	}

	/**
	 * Checks for attributes of the given type in the dataset
	 * 
	 * @param attType
	 *            the attribute type to look for
	 * @return true if attributes of the given type are present
	 */
	public boolean checkForAttributeType(int attType) {

		int i = 0;

		while (i < m_Attributes.size()) {
			if (attribute(i++).type() == attType) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Checks for string attributes in the dataset
	 * 
	 * @return true if string attributes are present, false otherwise
	 */
	public/* @pure@ */boolean checkForStringAttributes() {
		return checkForAttributeType(Attribute.STRING);
	}

	/**
	 * Checks if the given instance is compatible with this dataset. Only looks
	 * at the size of the instance and the ranges of the values for nominal and
	 * string attributes.
	 * 
	 * @param instance
	 *            the instance to check
	 * @return true if the instance is compatible with the dataset
	 */
	public/* @pure@ */boolean checkInstance(Instance instance) {

		if (instance.numAttributes() != numAttributes()) {
			return false;
		}
		for (int i = 0; i < numAttributes(); i++) {
			if (instance.isMissing(i)) {
				continue;
			} else if (attribute(i).isNominal() || attribute(i).isString()) {
				if (!(Utils.eq(instance.value(i), (int) instance.value(i)))) {
					return false;
				} else if (Utils.sm(instance.value(i), 0)
						|| Utils.gr(instance.value(i), attribute(i).numValues())) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * Returns the class attribute.
	 * 
	 * @return the class attribute
	 * @throws UnassignedClassException
	 *             if the class is not set
	 */
	// @ requires classIndex() >= 0;
	public/* @pure@ */Attribute classAttribute() {

		if (m_ClassIndex < 0) {
			throw new IllegalArgumentException(
					"Class index is negative (not set)!");
		}
		return attribute(m_ClassIndex);
	}

	/**
	 * Returns the class attribute's index. Returns negative number if it's
	 * undefined.
	 * 
	 * @return the class index as an integer
	 */
	// ensures \result == m_ClassIndex;
	public/* @pure@ */int classIndex() {

		return m_ClassIndex;
	}

	/**
	 * Compactifies the set of instances. Decreases the capacity of the set so
	 * that it matches the number of instances in the set.
	 */
	public void compactify() {

		m_Instances.trimToSize();
	}

	/**
	 * Removes all instances from the set.
	 */
	public void delete() {

		m_Instances = new ArrayList<Instance>();
	}

	/**
	 * Removes an instance at the given position from the set.
	 * 
	 * @param index
	 *            the instance's position (index starts with 0)
	 */
	// @ requires 0 <= index && index < numInstances();
	public void delete(int index) {

		m_Instances.remove(index);
	}

	/**
	 * Deletes an attribute at the given position (0 to numAttributes() - 1).
	 * Attribute objects after the deletion point are copied so that their
	 * indices can be decremented. Creates a fresh list to hold the old and new
	 * attribute objects.
	 * 
	 * @param position
	 *            the attribute's position (position starts with 0)
	 * @throws IllegalArgumentException
	 *             if the given index is out of range or the class attribute is
	 *             being deleted
	 */
	// @ requires 0 <= position && position < numAttributes();
	// @ requires position != classIndex();
	public void deleteAttributeAt(int position) {

		if ((position < 0) || (position >= m_Attributes.size())) {
			throw new IllegalArgumentException("Index out of range");
		}
		if (position == m_ClassIndex) {
			throw new IllegalArgumentException("Can't delete class attribute");
		}

		ArrayList<Attribute> newList = new ArrayList<Attribute>(
				m_Attributes.size() - 1);
		HashMap<String, Integer> newMap = new HashMap<String, Integer>(
				(int) ((m_Attributes.size() - 1) / 0.75));
		for (int i = 0; i < position; i++) {
			Attribute att = m_Attributes.get(i);
			newList.add(att);
			newMap.put(att.name(), i);
		}
		for (int i = position + 1; i < m_Attributes.size(); i++) {
			Attribute newAtt = (Attribute) m_Attributes.get(i).copy();
			newAtt.setIndex(i - 1);
			newList.add(newAtt);
			newMap.put(newAtt.name(), i - 1);
		}
		m_Attributes = newList;
		m_NamesToAttributeIndices = newMap;

		if (m_ClassIndex > position) {
			m_ClassIndex--;
		}
		for (int i = 0; i < numInstances(); i++) {
			instance(i).setDataset(null);
			instance(i).deleteAttributeAt(position);
			instance(i).setDataset(this);
		}
	}

	/**
	 * Deletes all attributes of the given type in the dataset. A deep copy of
	 * the attribute information is performed before an attribute is deleted.
	 * 
	 * @param attType
	 *            the attribute type to delete
	 * @throws IllegalArgumentException
	 *             if attribute couldn't be successfully deleted (probably
	 *             because it is the class attribute).
	 */
	public void deleteAttributeType(int attType) {
		int i = 0;
		while (i < m_Attributes.size()) {
			if (attribute(i).type() == attType) {
				deleteAttributeAt(i);
			} else {
				i++;
			}
		}
	}

	/**
	 * Deletes all string attributes in the dataset. A deep copy of the
	 * attribute information is performed before an attribute is deleted.
	 * 
	 * @throws IllegalArgumentException
	 *             if string attribute couldn't be successfully deleted
	 *             (probably because it is the class attribute).
	 * @see #deleteAttributeType(int)
	 */
	public void deleteStringAttributes() {
		deleteAttributeType(Attribute.STRING);
	}

	/**
	 * Removes all instances with missing values for a particular attribute from
	 * the dataset.
	 * 
	 * @param attIndex
	 *            the attribute's index (index starts with 0)
	 */
	// @ requires 0 <= attIndex && attIndex < numAttributes();
	public void deleteWithMissing(int attIndex) {

		ArrayList<Instance> newInstances = new ArrayList<Instance>(
				numInstances());

		for (int i = 0; i < numInstances(); i++) {
			if (!instance(i).isMissing(attIndex)) {
				newInstances.add(instance(i));
			}
		}
		m_Instances = newInstances;
	}

	/**
	 * Removes all instances with missing values for a particular attribute from
	 * the dataset.
	 * 
	 * @param att
	 *            the attribute
	 */
	public void deleteWithMissing(/* @non_null@ */Attribute att) {

		deleteWithMissing(att.index());
	}

	/**
	 * Returns an enumeration of all the attributes. The class attribute (if
	 * set) is skipped by this enumeration.
	 * 
	 * @return enumeration of all the attributes.
	 */
	public/* @non_null pure@ */Enumeration<Attribute> enumerateAttributes() {

		return new WekaEnumeration<Attribute>(m_Attributes, m_ClassIndex);
	}

	/**
	 * Returns an enumeration of all instances in the dataset.
	 * 
	 * @return enumeration of all instances in the dataset
	 */
	public/* @non_null pure@ */Enumeration<Instance> enumerateInstances() {

		return new WekaEnumeration<Instance>(m_Instances);
	}

	/**
	 * Checks if two headers are equivalent. If not, then returns a message why
	 * they differ.
	 * 
	 * @param dataset
	 *            another dataset
	 * @return null if the header of the given dataset is equivalent to this
	 *         header, otherwise a message with details on why they differ
	 */
	public String equalHeadersMsg(Instances dataset) {
		// Check class and all attributes
		if (m_ClassIndex != dataset.m_ClassIndex) {
			return "Class index differ: " + (m_ClassIndex + 1) + " != "
					+ (dataset.m_ClassIndex + 1);
		}

		if (m_Attributes.size() != dataset.m_Attributes.size()) {
			return "Different number of attributes: " + m_Attributes.size()
					+ " != " + dataset.m_Attributes.size();
		}

		for (int i = 0; i < m_Attributes.size(); i++) {
			String msg = attribute(i).equalsMsg(dataset.attribute(i));
			if (msg != null) {
				return "Attributes differ at position " + (i + 1) + ":\n" + msg;
			}
		}

		return null;
	}

	/**
	 * Checks if two headers are equivalent.
	 * 
	 * @param dataset
	 *            another dataset
	 * @return true if the header of the given dataset is equivalent to this
	 *         header
	 */
	public/* @pure@ */boolean equalHeaders(Instances dataset) {
		return (equalHeadersMsg(dataset) == null);
	}

	/**
	 * Returns the first instance in the set.
	 * 
	 * @return the first instance in the set
	 */
	// @ requires numInstances() > 0;
	public/* @non_null pure@ */Instance firstInstance() {

		return m_Instances.get(0);
	}

	/**
	 * Returns a random number generator. The initial seed of the random number
	 * generator depends on the given seed and the hash code of a string
	 * representation of a instances chosen based on the given seed.
	 * 
	 * @param seed
	 *            the given seed
	 * @return the random number generator
	 */
	public Random getRandomNumberGenerator(long seed) {

		Random r = new Random(seed);
		r.setSeed(instance(r.nextInt(numInstances())).toStringNoWeight()
				.hashCode() + seed);
		return r;
	}

	/**
	 * Inserts an attribute at the given position (0 to numAttributes()) and
	 * sets all values to be missing. Shallow copies the attribute before it is
	 * inserted. Existing attribute objects at and after the insertion point are
	 * also copied so that their indices can be incremented. Creates a fresh
	 * list to hold the old and new attribute objects.
	 * 
	 * @param att
	 *            the attribute to be inserted
	 * @param position
	 *            the attribute's position (position starts with 0)
	 * @throws IllegalArgumentException
	 *             if the given index is out of range
	 */
	// @ requires 0 <= position;
	// @ requires position <= numAttributes();
	public void insertAttributeAt(/* @non_null@ */Attribute att, int position) {

		if ((position < 0) || (position > m_Attributes.size())) {
			throw new IllegalArgumentException("Index out of range");
		}
		if (attribute(att.name()) != null) {
			throw new IllegalArgumentException("Attribute name '" + att.name()
					+ "' already in use at position #"
					+ attribute(att.name()).index());
		}
		att = (Attribute) att.copy();
		att.setIndex(position);

		ArrayList<Attribute> newList = new ArrayList<Attribute>(
				m_Attributes.size() + 1);
		HashMap<String, Integer> newMap = new HashMap<String, Integer>(
				(int) ((m_Attributes.size() + 1) / 0.75));
		for (int i = 0; i < position; i++) {
			Attribute oldAtt = m_Attributes.get(i);
			newList.add(oldAtt);
			newMap.put(oldAtt.name(), i);
		}
		newList.add(att);
		newMap.put(att.name(), position);
		for (int i = position; i < m_Attributes.size(); i++) {
			Attribute newAtt = (Attribute) m_Attributes.get(i).copy();
			newAtt.setIndex(i + 1);
			newList.add(newAtt);
			newMap.put(newAtt.name(), i + 1);
		}
		m_Attributes = newList;
		m_NamesToAttributeIndices = newMap;

		for (int i = 0; i < numInstances(); i++) {
			instance(i).setDataset(null);
			instance(i).insertAttributeAt(position);
			instance(i).setDataset(this);
		}
		if (m_ClassIndex >= position) {
			m_ClassIndex++;
		}
	}

	/**
	 * Returns the instance at the given position.
	 * 
	 * @param index
	 *            the instance's index (index starts with 0)
	 * @return the instance at the given position
	 */
	// @ requires 0 <= index;
	// @ requires index < numInstances();
	public/* @non_null pure@ */Instance instance(int index) {

		return m_Instances.get(index);
	}

	/**
	 * Returns the instance at the given position.
	 * 
	 * @param index
	 *            the instance's index (index starts with 0)
	 * @return the instance at the given position
	 */
	// @ requires 0 <= index;
	// @ requires index < numInstances();
	@Override
	public/* @non_null pure@ */Instance get(int index) {

		return m_Instances.get(index);
	}

	/**
	 * Returns the kth-smallest attribute value of a numeric attribute.
	 * 
	 * @param att
	 *            the Attribute object
	 * @param k
	 *            the value of k
	 * @return the kth-smallest value
	 */
	public double kthSmallestValue(Attribute att, int k) {

		return kthSmallestValue(att.index(), k);
	}

	/**
	 * Returns the kth-smallest attribute value of a numeric attribute. NOTE
	 * CHANGE: Missing values (NaN values) are now treated as Double.MAX_VALUE.
	 * Also, the order of the instances in the data is no longer affected.
	 * 
	 * @param attIndex
	 *            the attribute's index
	 * @param k
	 *            the value of k
	 * @return the kth-smallest value
	 */
	public double kthSmallestValue(int attIndex, int k) {

		if (!attribute(attIndex).isNumeric()) {
			throw new IllegalArgumentException(
					"Instances: attribute must be numeric to compute kth-smallest value.");
		}

		if ((k < 1) || (k > numInstances())) {
			throw new IllegalArgumentException(
					"Instances: value for k for computing kth-smallest value too large.");
		}

		double[] vals = new double[numInstances()];
		for (int i = 0; i < vals.length; i++) {
			double val = instance(i).value(attIndex);
			if (Utils.isMissingValue(val)) {
				vals[i] = Double.MAX_VALUE;
			} else {
				vals[i] = val;
			}
		}
		return Utils.kthSmallestValue(vals, k);
	}

	/**
	 * Returns the last instance in the set.
	 * 
	 * @return the last instance in the set
	 */
	// @ requires numInstances() > 0;
	public/* @non_null pure@ */Instance lastInstance() {

		return m_Instances.get(m_Instances.size() - 1);
	}

	/**
	 * Returns the mean (mode) for a numeric (nominal) attribute as a
	 * floating-point value. Returns 0 if the attribute is neither nominal nor
	 * numeric. If all values are missing it returns zero.
	 * 
	 * @param attIndex
	 *            the attribute's index (index starts with 0)
	 * @return the mean or the mode
	 */
	public/* @pure@ */double meanOrMode(int attIndex) {

		double result, found;
		int[] counts;

		if (attribute(attIndex).isNumeric()) {
			result = found = 0;
			for (int j = 0; j < numInstances(); j++) {
				if (!instance(j).isMissing(attIndex)) {
					found += instance(j).weight();
					result += instance(j).weight()
							* instance(j).value(attIndex);
				}
			}
			if (found <= 0) {
				return 0;
			} else {
				return result / found;
			}
		} else if (attribute(attIndex).isNominal()) {
			counts = new int[attribute(attIndex).numValues()];
			for (int j = 0; j < numInstances(); j++) {
				if (!instance(j).isMissing(attIndex)) {
					counts[(int) instance(j).value(attIndex)] += instance(j)
							.weight();
				}
			}
			return Utils.maxIndex(counts);
		} else {
			return 0;
		}
	}

	/**
	 * Returns the mean (mode) for a numeric (nominal) attribute as a
	 * floating-point value. Returns 0 if the attribute is neither nominal nor
	 * numeric. If all values are missing it returns zero.
	 * 
	 * @param att
	 *            the attribute
	 * @return the mean or the mode
	 */
	public/* @pure@ */double meanOrMode(Attribute att) {

		return meanOrMode(att.index());
	}

	/**
	 * Returns the number of attributes.
	 * 
	 * @return the number of attributes as an integer
	 */
	// @ ensures \result == m_Attributes.size();
	public/* @pure@ */int numAttributes() {

		return m_Attributes.size();
	}

	/**
	 * Returns the number of class labels.
	 * 
	 * @return the number of class labels as an integer if the class attribute
	 *         is nominal, 1 otherwise.
	 * @throws UnassignedClassException
	 *             if the class is not set
	 */
	// @ requires classIndex() >= 0;
	public/* @pure@ */int numClasses() {

		if (m_ClassIndex < 0) {
			System.err.println("Class index is negative (not set)!");
		}
		if (!classAttribute().isNominal()) {
			return 1;
		} else {
			return classAttribute().numValues();
		}
	}

	/**
	 * Returns the number of distinct values of a given attribute. Returns the
	 * number of instances if the attribute is a string attribute. The value
	 * 'missing' is not counted.
	 * 
	 * @param attIndex
	 *            the attribute (index starts with 0)
	 * @return the number of distinct values of a given attribute
	 */
	// @ requires 0 <= attIndex;
	// @ requires attIndex < numAttributes();
	public/* @pure@ */int numDistinctValues(int attIndex) {

		if (attribute(attIndex).isNumeric()) {
			double[] attVals = attributeToDoubleArray(attIndex);
			int[] sorted = Utils.sort(attVals);
			double prev = 0;
			int counter = 0;
			for (int i = 0; i < sorted.length; i++) {
				Instance current = instance(sorted[i]);
				if (current.isMissing(attIndex)) {
					break;
				}
				if ((i == 0) || (current.value(attIndex) > prev)) {
					prev = current.value(attIndex);
					counter++;
				}
			}
			return counter;
		} else {
			return attribute(attIndex).numValues();
		}
	}

	/**
	 * Returns the number of distinct values of a given attribute. Returns the
	 * number of instances if the attribute is a string attribute. The value
	 * 'missing' is not counted.
	 * 
	 * @param att
	 *            the attribute
	 * @return the number of distinct values of a given attribute
	 */
	public/* @pure@ */int numDistinctValues(/* @non_null@ */Attribute att) {

		return numDistinctValues(att.index());
	}

	/**
	 * Returns the number of instances in the dataset.
	 * 
	 * @return the number of instances in the dataset as an integer
	 */
	// @ ensures \result == m_Instances.size();
	public/* @pure@ */int numInstances() {

		return m_Instances.size();
	}

	/**
	 * Returns the number of instances in the dataset.
	 * 
	 * @return the number of instances in the dataset as an integer
	 */
	// @ ensures \result == m_Instances.size();
	@Override
	public/* @pure@ */int size() {

		return m_Instances.size();
	}

	/**
	 * Shuffles the instances in the set so that they are ordered randomly.
	 * 
	 * @param random
	 *            a random number generator
	 */
	public void randomize(Random random) {

		for (int j = numInstances() - 1; j > 0; j--) {
			swap(j, random.nextInt(j + 1));
		}
	}

	/**
	 * Reads a single instance from the reader and appends it to the dataset.
	 * Automatically expands the dataset if it is not large enough to hold the
	 * instance. This method does not check for carriage return at the end of
	 * the line.
	 * 
	 * @param reader
	 *            the reader
	 * @return false if end of file has been reached
	 * @throws IOException
	 *             if the information is not read successfully
	 * @deprecated instead of using this method in conjunction with the
	 *             <code>readInstance(Reader)</code> method, one should use the
	 *             <code>ArffLoader</code> or <code>DataSource</code> class
	 *             instead.
	 * @see weka.core.converters.ArffLoader
	 * @see weka.core.converters.ConverterUtils.DataSource
	 */
	@Deprecated
	public boolean readInstance(Reader reader) throws IOException {

		ArffReader arff = new ArffReader(reader, this, m_Lines, 1);
		Instance inst = arff.readInstance(arff.getData(), false);
		m_Lines = arff.getLineNo();
		if (inst != null) {
			add(inst);
			return true;
		} else {
			return false;
		}
	}

	/**
	 * Replaces the attribute at the given position (0 to numAttributes()) with
	 * the given attribute and sets all its values to be missing. Shallow copies
	 * the given attribute before it is inserted. Creates a fresh list to hold
	 * the old and new attribute objects.
	 * 
	 * @param att
	 *            the attribute to be inserted
	 * @param position
	 *            the attribute's position (position starts with 0)
	 * @throws IllegalArgumentException
	 *             if the given index is out of range
	 */
	// @ requires 0 <= position;
	// @ requires position <= numAttributes();
	public void replaceAttributeAt(/* @non_null@ */Attribute att, int position) {

		if ((position < 0) || (position > m_Attributes.size())) {
			throw new IllegalArgumentException("Index out of range");
		}

		// Does the new attribute have a different name?
		if (!att.name().equals(m_Attributes.get(position).name())) {

			// Need to check if attribute name already exists
			Attribute candidate = attribute(att.name());
			if ((candidate != null) && (position != candidate.index())) {
				throw new IllegalArgumentException("Attribute name '"
						+ att.name() + "' already in use at position #"
						+ attribute(att.name()).index());
			}
		}
		att = (Attribute) att.copy();
		att.setIndex(position);

		ArrayList<Attribute> newList = new ArrayList<Attribute>(
				m_Attributes.size());
		HashMap<String, Integer> newMap = new HashMap<String, Integer>(
				(int) ((m_Attributes.size() + 1) / 0.75));
		for (int i = 0; i < position; i++) {
			Attribute oldAtt = m_Attributes.get(i);
			newList.add(oldAtt);
			newMap.put(oldAtt.name(), i);
		}
		newList.add(att);
		newMap.put(att.name(), position);
		for (int i = position + 1; i < m_Attributes.size(); i++) {
			Attribute newAtt = (Attribute) m_Attributes.get(i);
			newList.add(newAtt);
			newMap.put(newAtt.name(), i);
		}
		m_Attributes = newList;
		m_NamesToAttributeIndices = newMap;

		for (int i = 0; i < numInstances(); i++) {
			instance(i).setDataset(null);
			instance(i).setMissing(position);
			instance(i).setDataset(this);
		}
		if (m_ClassIndex >= position) {
			m_ClassIndex++;
		}
	}

	/**
	 * Returns the relation's name.
	 * 
	 * @return the relation's name as a string
	 */
	// @ ensures \result == m_RelationName;
	public/* @pure@ */String relationName() {

		return m_RelationName;
	}

	/**
	 * Removes the instance at the given position.
	 * 
	 * @param index
	 *            the instance's index (index starts with 0)
	 * @return the instance at the given position
	 */
	// @ requires 0 <= index;
	// @ requires index < numInstances();
	@Override
	public Instance remove(int index) {

		return m_Instances.remove(index);
	}

	/**
	 * Renames an attribute. This change only affects this dataset.
	 * 
	 * @param att
	 *            the attribute's index (index starts with 0)
	 * @param name
	 *            the new name
	 */
	public void renameAttribute(int att, String name) {

		Attribute existingAtt = attribute(name);
		if (existingAtt != null) {
			if (att == existingAtt.index()) {
				return; // Old name is equal to new name
			} else {
				throw new IllegalArgumentException("Attribute name '" + name
						+ "' already present at position #"
						+ existingAtt.index());
			}
		}

		Attribute newAtt = attribute(att).copy(name);
		ArrayList<Attribute> newVec = new ArrayList<Attribute>(numAttributes());
		HashMap<String, Integer> newMap = new HashMap<String, Integer>(
				(int) (numAttributes() / 0.75));
		for (Attribute attr : m_Attributes) {
			if (attr.index() == att) {
				newVec.add(newAtt);
				newMap.put(name, att);
			} else {
				newVec.add(attr);
				newMap.put(attr.name(), attr.index());
			}
		}
		m_Attributes = newVec;
		m_NamesToAttributeIndices = newMap;
	}

	/**
	 * Renames an attribute. This change only affects this dataset.
	 * 
	 * @param att
	 *            the attribute
	 * @param name
	 *            the new name
	 */
	public void renameAttribute(Attribute att, String name) {

		renameAttribute(att.index(), name);
	}

	/**
	 * Renames the value of a nominal (or string) attribute value. This change
	 * only affects this dataset.
	 * 
	 * @param att
	 *            the attribute's index (index starts with 0)
	 * @param val
	 *            the value's index (index starts with 0)
	 * @param name
	 *            the new name
	 */
	public void renameAttributeValue(int att, int val, String name) {

		Attribute newAtt = (Attribute) attribute(att).copy();
		ArrayList<Attribute> newVec = new ArrayList<Attribute>(numAttributes());

		newAtt.setValue(val, name);
		for (Attribute attr : m_Attributes) {
			if (attr.index() == att) {
				newVec.add(newAtt);
			} else {
				newVec.add(attr);
			}
		}
		m_Attributes = newVec;
	}

	/**
	 * Renames the value of a nominal (or string) attribute value. This change
	 * only affects this dataset.
	 * 
	 * @param att
	 *            the attribute
	 * @param val
	 *            the value
	 * @param name
	 *            the new name
	 */
	public void renameAttributeValue(Attribute att, String val, String name) {

		int v = att.indexOfValue(val);
		if (v == -1) {
			throw new IllegalArgumentException(val + " not found");
		}
		renameAttributeValue(att.index(), v, name);
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement.
	 * 
	 * @param random
	 *            a random number generator
	 * @return the new dataset
	 */
	public Instances resample(Random random) {

		Instances newData = new Instances(this, numInstances());
		while (newData.numInstances() < numInstances()) {
			newData.add(instance(random.nextInt(numInstances())));
		}
		return newData;
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the current instance weights. The weights of the
	 * instances in the new dataset are set to one. See also
	 * resampleWithWeights(Random, double[], boolean[]).
	 * 
	 * @param random
	 *            a random number generator
	 * @return the new dataset
	 */
	public Instances resampleWithWeights(Random random) {

		return resampleWithWeights(random, false);
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the current instance weights. The weights of the
	 * instances in the new dataset are set to one. See also
	 * resampleWithWeights(Random, double[], boolean[]).
	 * 
	 * @param random
	 *            a random number generator
	 * @param sampled
	 *            an array indicating what has been sampled
	 * @return the new dataset
	 */
	public Instances resampleWithWeights(Random random, boolean[] sampled) {

		return resampleWithWeights(random, sampled, false);
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the current instance weights. The weights of the
	 * instances in the new dataset are set to one. See also
	 * resampleWithWeights(Random, double[], boolean[]).
	 * 
	 * @param random
	 *            a random number generator
	 * @param representUsingWeights
	 *            if true, copies are represented using weights in resampled
	 *            data
	 * @return the new dataset
	 */
	public Instances resampleWithWeights(Random random,
			boolean representUsingWeights) {

		return resampleWithWeights(random, null, representUsingWeights);
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the current instance weights. The weights of the
	 * instances in the new dataset are set to one. See also
	 * resampleWithWeights(Random, double[], boolean[]).
	 * 
	 * @param random
	 *            a random number generator
	 * @param sampled
	 *            an array indicating what has been sampled
	 * @param representUsingWeights
	 *            if true, copies are represented using weights in resampled
	 *            data
	 * @return the new dataset
	 */
	public Instances resampleWithWeights(Random random, boolean[] sampled,
			boolean representUsingWeights) {

		double[] weights = new double[numInstances()];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = instance(i).weight();
		}
		return resampleWithWeights(random, weights, sampled,
				representUsingWeights);
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the given weight vector. See also
	 * resampleWithWeights(Random, double[], boolean[]).
	 * 
	 * @param random
	 *            a random number generator
	 * @param weights
	 *            the weight vector
	 * @return the new dataset
	 * @throws IllegalArgumentException
	 *             if the weights array is of the wrong length or contains
	 *             negative weights.
	 */
	public Instances resampleWithWeights(Random random, double[] weights) {

		return resampleWithWeights(random, weights, null);
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the given weight vector. The weights of the
	 * instances in the new dataset are set to one. The length of the weight
	 * vector has to be the same as the number of instances in the dataset, and
	 * all weights have to be positive. Uses Walker's method, see pp. 232 of
	 * "Stochastic Simulation" by B.D. Ripley (1987).
	 * 
	 * @param random
	 *            a random number generator
	 * @param weights
	 *            the weight vector
	 * @param sampled
	 *            an array indicating what has been sampled, can be null
	 * @return the new dataset
	 * @throws IllegalArgumentException
	 *             if the weights array is of the wrong length or contains
	 *             negative weights.
	 */
	public Instances resampleWithWeights(Random random, double[] weights,
			boolean[] sampled) {

		return resampleWithWeights(random, weights, sampled, false);
	}

	/**
	 * Creates a new dataset of the same size using random sampling with
	 * replacement according to the given weight vector. The weights of the
	 * instances in the new dataset are set to one. The length of the weight
	 * vector has to be the same as the number of instances in the dataset, and
	 * all weights have to be positive. Uses Walker's method, see pp. 232 of
	 * "Stochastic Simulation" by B.D. Ripley (1987).
	 * 
	 * @param random
	 *            a random number generator
	 * @param weights
	 *            the weight vector
	 * @param sampled
	 *            an array indicating what has been sampled, can be null
	 * @param representUsingWeights
	 *            if true, copies are represented using weights in resampled
	 *            data
	 * @return the new dataset
	 * @throws IllegalArgumentException
	 *             if the weights array is of the wrong length or contains
	 *             negative weights.
	 */
	public Instances resampleWithWeights(Random random, double[] weights,
			boolean[] sampled, boolean representUsingWeights) {

		if (weights.length != numInstances()) {
			throw new IllegalArgumentException(
					"weights.length != numInstances.");
		}

		Instances newData = new Instances(this, numInstances());
		if (numInstances() == 0) {
			return newData;
		}

		// Walker's method, see pp. 232 of "Stochastic Simulation" by B.D.
		// Ripley
		double[] P = new double[weights.length];
		System.arraycopy(weights, 0, P, 0, weights.length);
		Utils.normalize(P);
		double[] Q = new double[weights.length];
		int[] A = new int[weights.length];
		int[] W = new int[weights.length];
		int M = weights.length;
		int NN = -1;
		int NP = M;
		for (int I = 0; I < M; I++) {
			if (P[I] < 0) {
				throw new IllegalArgumentException(
						"Weights have to be positive.");
			}
			Q[I] = M * P[I];
			if (Q[I] < 1.0) {
				W[++NN] = I;
			} else {
				W[--NP] = I;
			}
		}
		if (NN > -1 && NP < M) {
			for (int S = 0; S < M - 1; S++) {
				int I = W[S];
				int J = W[NP];
				A[I] = J;
				Q[J] += Q[I] - 1.0;
				if (Q[J] < 1.0) {
					NP++;
				}
				if (NP >= M) {
					break;
				}
			}
			// A[W[M]] = W[M];
		}

		for (int I = 0; I < M; I++) {
			Q[I] += I;
		}

		// Do we need to keep track of how many copies to use?
		int[] counts = null;
		if (representUsingWeights) {
			counts = new int[M];
		}

		for (int i = 0; i < numInstances(); i++) {
			int ALRV;
			double U = M * random.nextDouble();
			int I = (int) U;
			if (U < Q[I]) {
				ALRV = I;
			} else {
				ALRV = A[I];
			}
			if (representUsingWeights) {
				counts[ALRV]++;
			} else {
				newData.add(instance(ALRV));
			}
			if (sampled != null) {
				sampled[ALRV] = true;
			}
			if (!representUsingWeights) {
				newData.instance(newData.numInstances() - 1).setWeight(1);
			}
		}

		// Add data based on counts if weights should represent numbers of
		// copies.
		if (representUsingWeights) {
			for (int i = 0; i < counts.length; i++) {
				if (counts[i] > 0) {
					newData.add(instance(i));
					newData.instance(newData.numInstances() - 1).setWeight(
							counts[i]);
				}
			}
		}

		return newData;
	}

	/**
	 * Replaces the instance at the given position. Shallow copies instance
	 * before it is added. Does not check if the instance is compatible with the
	 * dataset. Note: String or relational values are not transferred.
	 * 
	 * @param index
	 *            position where instance is to be inserted
	 * @param instance
	 *            the instance to be inserted
	 * @return the instance previously at that position
	 */
	// @ requires 0 <= index;
	// @ requires index < m_Instances.size();
	@Override
	public Instance set(int index, /* @non_null@ */Instance instance) {

		Instance newInstance = (Instance) instance.copy();
		Instance oldInstance = m_Instances.get(index);

		newInstance.setDataset(this);
		m_Instances.set(index, newInstance);

		return oldInstance;
	}

	/**
	 * Sets the class attribute.
	 * 
	 * @param att
	 *            attribute to be the class
	 */
	public void setClass(Attribute att) {

		m_ClassIndex = att.index();
	}

	/**
	 * Sets the class index of the set. If the class index is negative there is
	 * assumed to be no class. (ie. it is undefined)
	 * 
	 * @param classIndex
	 *            the new class index (index starts with 0)
	 * @throws IllegalArgumentException
	 *             if the class index is too big or < 0
	 */
	public void setClassIndex(int classIndex) {

		if (classIndex >= numAttributes()) {
			throw new IllegalArgumentException("Invalid class index: "
					+ classIndex);
		}
		m_ClassIndex = classIndex;
	}

	/**
	 * Sets the relation's name.
	 * 
	 * @param newName
	 *            the new relation name.
	 */
	public void setRelationName(/* @non_null@ */String newName) {

		m_RelationName = newName;
	}

	/**
	 * Sorts a nominal attribute (stable, linear-time sort). Instances are
	 * sorted based on the attribute label ordering specified in the header.
	 * 
	 * @param attIndex
	 *            the attribute's index (index starts with 0)
	 */
	protected void sortBasedOnNominalAttribute(int attIndex) {

		// Figure out number of instances for each attribute value
		// and store original list of instances away
		int[] counts = new int[attribute(attIndex).numValues()];
		Instance[] backup = new Instance[numInstances()];
		int j = 0;
		for (Instance inst : this) {
			backup[j++] = inst;
			if (!inst.isMissing(attIndex)) {
				counts[(int) inst.value(attIndex)]++;
			}
		}

		// Indices to figure out where to add instances
		int[] indices = new int[counts.length];
		int start = 0;
		for (int i = 0; i < counts.length; i++) {
			indices[i] = start;
			start += counts[i];
		}
		for (Instance inst : backup) { // Use backup here
			if (!inst.isMissing(attIndex)) {
				m_Instances.set(indices[(int) inst.value(attIndex)]++, inst);
			} else {
				m_Instances.set(start++, inst);
			}
		}
	}

	/**
	 * Sorts the instances based on an attribute. For numeric attributes,
	 * instances are sorted in ascending order. For nominal attributes,
	 * instances are sorted based on the attribute label ordering specified in
	 * the header. Instances with missing values for the attribute are placed at
	 * the end of the dataset.
	 * 
	 * @param attIndex
	 *            the attribute's index (index starts with 0)
	 */
	public void sort(int attIndex) {

		if (!attribute(attIndex).isNominal()) {

			// Use quicksort from Utils class for sorting
			double[] vals = new double[numInstances()];
			Instance[] backup = new Instance[vals.length];
			for (int i = 0; i < vals.length; i++) {
				Instance inst = instance(i);
				backup[i] = inst;
				double val = inst.value(attIndex);
				if (Utils.isMissingValue(val)) {
					vals[i] = Double.MAX_VALUE;
				} else {
					vals[i] = val;
				}
			}

			int[] sortOrder = Utils.sortWithNoMissingValues(vals);
			for (int i = 0; i < vals.length; i++) {
				m_Instances.set(i, backup[sortOrder[i]]);
			}
		} else {
			sortBasedOnNominalAttribute(attIndex);
		}
	}

	/**
	 * Sorts the instances based on an attribute. For numeric attributes,
	 * instances are sorted into ascending order. For nominal attributes,
	 * instances are sorted based on the attribute label ordering specified in
	 * the header. Instances with missing values for the attribute are placed at
	 * the end of the dataset.
	 * 
	 * @param att
	 *            the attribute
	 */
	public void sort(Attribute att) {

		sort(att.index());
	}

	/**
	 * Sorts the instances based on an attribute, using a stable sort. For
	 * numeric attributes, instances are sorted in ascending order. For nominal
	 * attributes, instances are sorted based on the attribute label ordering
	 * specified in the header. Instances with missing values for the attribute
	 * are placed at the end of the dataset.
	 * 
	 * @param attIndex
	 *            the attribute's index (index starts with 0)
	 */
	public void stableSort(int attIndex) {

		if (!attribute(attIndex).isNominal()) {

			// Use quicksort from Utils class for sorting
			double[] vals = new double[numInstances()];
			Instance[] backup = new Instance[vals.length];
			for (int i = 0; i < vals.length; i++) {
				Instance inst = instance(i);
				backup[i] = inst;
				vals[i] = inst.value(attIndex);
			}

			int[] sortOrder = Utils.stableSort(vals);
			for (int i = 0; i < vals.length; i++) {
				m_Instances.set(i, backup[sortOrder[i]]);
			}
		} else {
			sortBasedOnNominalAttribute(attIndex);
		}
	}

	/**
	 * Sorts the instances based on an attribute, using a stable sort. For
	 * numeric attributes, instances are sorted into ascending order. For
	 * nominal attributes, instances are sorted based on the attribute label
	 * ordering specified in the header. Instances with missing values for the
	 * attribute are placed at the end of the dataset.
	 * 
	 * @param att
	 *            the attribute
	 */
	public void stableSort(Attribute att) {

		stableSort(att.index());
	}

	/**
	 * Stratifies a set of instances according to its class values if the class
	 * attribute is nominal (so that afterwards a stratified cross-validation
	 * can be performed).
	 * 
	 * @param numFolds
	 *            the number of folds in the cross-validation
	 * @throws UnassignedClassException
	 *             if the class is not set
	 */
	public void stratify(int numFolds) {

		if (numFolds <= 1) {
			throw new IllegalArgumentException(
					"Number of folds must be greater than 1");
		}
		if (m_ClassIndex < 0) {
			throw new IllegalArgumentException(
					"Class index is negative (not set)!");
		}
		if (classAttribute().isNominal()) {

			// sort by class
			int index = 1;
			while (index < numInstances()) {
				Instance instance1 = instance(index - 1);
				for (int j = index; j < numInstances(); j++) {
					Instance instance2 = instance(j);
					if ((instance1.classValue() == instance2.classValue())
							|| (instance1.classIsMissing() && instance2
									.classIsMissing())) {
						swap(index, j);
						index++;
					}
				}
				index++;
			}
			stratStep(numFolds);
		}
	}

	/**
	 * Computes the sum of all the instances' weights.
	 * 
	 * @return the sum of all the instances' weights as a double
	 */
	public/* @pure@ */double sumOfWeights() {

		double sum = 0;

		for (int i = 0; i < numInstances(); i++) {
			sum += instance(i).weight();
		}
		return sum;
	}

	/**
	 * Creates the test set for one fold of a cross-validation on the dataset.
	 * 
	 * @param numFolds
	 *            the number of folds in the cross-validation. Must be greater
	 *            than 1.
	 * @param numFold
	 *            0 for the first fold, 1 for the second, ...
	 * @return the test set as a set of weighted instances
	 * @throws IllegalArgumentException
	 *             if the number of folds is less than 2 or greater than the
	 *             number of instances.
	 */
	// @ requires 2 <= numFolds && numFolds < numInstances();
	// @ requires 0 <= numFold && numFold < numFolds;
	public Instances testCV(int numFolds, int numFold) {

		int numInstForFold, first, offset;
		Instances test;

		if (numFolds < 2) {
			throw new IllegalArgumentException(
					"Number of folds must be at least 2!");
		}
		if (numFolds > numInstances()) {
			throw new IllegalArgumentException(
					"Can't have more folds than instances!");
		}
		numInstForFold = numInstances() / numFolds;
		if (numFold < numInstances() % numFolds) {
			numInstForFold++;
			offset = numFold;
		} else {
			offset = numInstances() % numFolds;
		}
		test = new Instances(this, numInstForFold);
		first = numFold * (numInstances() / numFolds) + offset;
		copyInstances(first, test, numInstForFold);
		return test;
	}

	/**
	 * Returns the dataset as a string in ARFF format. Strings are quoted if
	 * they contain whitespace characters, or if they are a question mark.
	 * 
	 * @return the dataset in ARFF format as a string
	 */
	@Override
	public String toString() {

		StringBuffer text = new StringBuffer();

		text.append(ARFF_RELATION).append(" ")
				.append(Utils.quote(m_RelationName)).append("\n\n");
		for (int i = 0; i < numAttributes(); i++) {
			text.append(attribute(i)).append("\n");
		}
		text.append("\n").append(ARFF_DATA).append("\n");

		text.append(stringWithoutHeader());
		return text.toString();
	}

	/**
	 * Returns the instances in the dataset as a string in ARFF format. Strings
	 * are quoted if they contain whitespace characters, or if they are a
	 * question mark.
	 * 
	 * @return the dataset in ARFF format as a string
	 */
	protected String stringWithoutHeader() {

		StringBuffer text = new StringBuffer();

		for (int i = 0; i < numInstances(); i++) {
			text.append(instance(i));
			if (i < numInstances() - 1) {
				text.append('\n');
			}
		}
		return text.toString();
	}

	/**
	 * Creates the training set for one fold of a cross-validation on the
	 * dataset.
	 * 
	 * @param numFolds
	 *            the number of folds in the cross-validation. Must be greater
	 *            than 1.
	 * @param numFold
	 *            0 for the first fold, 1 for the second, ...
	 * @return the training set
	 * @throws IllegalArgumentException
	 *             if the number of folds is less than 2 or greater than the
	 *             number of instances.
	 */
	// @ requires 2 <= numFolds && numFolds < numInstances();
	// @ requires 0 <= numFold && numFold < numFolds;
	public Instances trainCV(int numFolds, int numFold) {

		int numInstForFold, first, offset;
		Instances train;

		if (numFolds < 2) {
			throw new IllegalArgumentException(
					"Number of folds must be at least 2!");
		}
		if (numFolds > numInstances()) {
			throw new IllegalArgumentException(
					"Can't have more folds than instances!");
		}
		numInstForFold = numInstances() / numFolds;
		if (numFold < numInstances() % numFolds) {
			numInstForFold++;
			offset = numFold;
		} else {
			offset = numInstances() % numFolds;
		}
		train = new Instances(this, numInstances() - numInstForFold);
		first = numFold * (numInstances() / numFolds) + offset;
		copyInstances(0, train, first);
		copyInstances(first + numInstForFold, train, numInstances() - first
				- numInstForFold);

		return train;
	}

	/**
	 * Creates the training set for one fold of a cross-validation on the
	 * dataset. The data is subsequently randomized based on the given random
	 * number generator.
	 * 
	 * @param numFolds
	 *            the number of folds in the cross-validation. Must be greater
	 *            than 1.
	 * @param numFold
	 *            0 for the first fold, 1 for the second, ...
	 * @param random
	 *            the random number generator
	 * @return the training set
	 * @throws IllegalArgumentException
	 *             if the number of folds is less than 2 or greater than the
	 *             number of instances.
	 */
	// @ requires 2 <= numFolds && numFolds < numInstances();
	// @ requires 0 <= numFold && numFold < numFolds;
	public Instances trainCV(int numFolds, int numFold, Random random) {

		Instances train = trainCV(numFolds, numFold);
		train.randomize(random);
		return train;
	}

	/**
	 * Computes the variance for all numeric attributes simultaneously. This is
	 * faster than calling variance() for each attribute. The resulting array
	 * has as many dimensions as there are attributes. Array elements
	 * corresponding to non-numeric attributes are set to 0.
	 * 
	 * @return the array containing the variance values
	 */
	public/* @pure@ */double[] variances() {

		double[] sum = new double[numAttributes()];
		double[] sumSquared = new double[numAttributes()];
		double[] sumOfWeights = new double[numAttributes()];

		for (int i = 0; i < numInstances(); i++) {
			for (int attIndex = 0; attIndex < numAttributes(); attIndex++) {
				if (attribute(attIndex).isNumeric()) {
					if (!instance(i).isMissing(attIndex)) {
						sum[attIndex] += instance(i).weight()
								* instance(i).value(attIndex);
						sumSquared[attIndex] += instance(i).weight()
								* instance(i).value(attIndex)
								* instance(i).value(attIndex);
						sumOfWeights[attIndex] += instance(i).weight();
					}
				}
			}
		}

		double[] variances = new double[numAttributes()];
		for (int attIndex = 0; attIndex < numAttributes(); attIndex++) {
			if (attribute(attIndex).isNumeric()) {
				if (sumOfWeights[attIndex] > 1) {
					double result = (sumSquared[attIndex] - (sum[attIndex]
							* sum[attIndex] / sumOfWeights[attIndex]))
							/ (sumOfWeights[attIndex] - 1);

					// We don't like negative variance
					if (result > 0) {
						variances[attIndex] = result;
					}
				}
			}
		}
		return variances;
	}

	/**
	 * Computes the variance for a numeric attribute.
	 * 
	 * @param attIndex
	 *            the numeric attribute (index starts with 0)
	 * @return the variance if the attribute is numeric
	 * @throws IllegalArgumentException
	 *             if the attribute is not numeric
	 */
	public/* @pure@ */double variance(int attIndex) {

		double sum = 0, sumSquared = 0, sumOfWeights = 0;

		if (!attribute(attIndex).isNumeric()) {
			throw new IllegalArgumentException(
					"Can't compute variance because attribute is "
							+ "not numeric!");
		}
		for (int i = 0; i < numInstances(); i++) {
			if (!instance(i).isMissing(attIndex)) {
				sum += instance(i).weight() * instance(i).value(attIndex);
				sumSquared += instance(i).weight()
						* instance(i).value(attIndex)
						* instance(i).value(attIndex);
				sumOfWeights += instance(i).weight();
			}
		}
		if (sumOfWeights <= 1) {
			return 0;
		}
		double result = (sumSquared - (sum * sum / sumOfWeights))
				/ (sumOfWeights - 1);

		// We don't like negative variance
		if (result < 0) {
			return 0;
		} else {
			return result;
		}
	}

	/**
	 * Computes the variance for a numeric attribute.
	 * 
	 * @param att
	 *            the numeric attribute
	 * @return the variance if the attribute is numeric
	 * @throws IllegalArgumentException
	 *             if the attribute is not numeric
	 */
	public/* @pure@ */double variance(Attribute att) {

		return variance(att.index());
	}

	/**
	 * Gets the value of all instances in this dataset for a particular
	 * attribute. Useful in conjunction with Utils.sort to allow iterating
	 * through the dataset in sorted order for some attribute.
	 * 
	 * @param index
	 *            the index of the attribute.
	 * @return an array containing the value of the desired attribute for each
	 *         instance in the dataset.
	 */
	// @ requires 0 <= index && index < numAttributes();
	public/* @pure@ */double[] attributeToDoubleArray(int index) {

		double[] result = new double[numInstances()];
		for (int i = 0; i < result.length; i++) {
			result[i] = instance(i).value(index);
		}
		return result;
	}

	/**
	 * Copies instances from one set to the end of another one.
	 * 
	 * @param from
	 *            the position of the first instance to be copied
	 * @param dest
	 *            the destination for the instances
	 * @param num
	 *            the number of instances to be copied
	 */
	// @ requires 0 <= from && from <= numInstances() - num;
	// @ requires 0 <= num;
	protected void copyInstances(int from, /* @non_null@ */Instances dest,
			int num) {

		for (int i = 0; i < num; i++) {
			dest.add(instance(from + i));
		}
	}

	/**
	 * Returns string including all instances, their weights and their indices
	 * in the original dataset.
	 * 
	 * @return description of instance and its weight as a string
	 */
	protected/* @pure@ */String instancesAndWeights() {

		StringBuffer text = new StringBuffer();

		for (int i = 0; i < numInstances(); i++) {
			text.append(instance(i) + " " + instance(i).weight());
			if (i < numInstances() - 1) {
				text.append("\n");
			}
		}
		return text.toString();
	}

	/**
	 * Help function needed for stratification of set.
	 * 
	 * @param numFolds
	 *            the number of folds for the stratification
	 */
	protected void stratStep(int numFolds) {

		ArrayList<Instance> newVec = new ArrayList<Instance>(m_Instances.size());
		int start = 0, j;

		// create stratified batch
		while (newVec.size() < numInstances()) {
			j = start;
			while (j < numInstances()) {
				newVec.add(instance(j));
				j = j + numFolds;
			}
			start++;
		}
		m_Instances = newVec;
	}

	/**
	 * Swaps two instances in the set.
	 * 
	 * @param i
	 *            the first instance's index (index starts with 0)
	 * @param j
	 *            the second instance's index (index starts with 0)
	 */
	// @ requires 0 <= i && i < numInstances();
	// @ requires 0 <= j && j < numInstances();
	public void swap(int i, int j) {

		Instance in = m_Instances.get(i);
		m_Instances.set(i, m_Instances.get(j));
		m_Instances.set(j, in);
	}

	/**
	 * Merges two sets of Instances together. The resulting set will have all
	 * the attributes of the first set plus all the attributes of the second
	 * set. The number of instances in both sets must be the same.
	 * 
	 * @param first
	 *            the first set of Instances
	 * @param second
	 *            the second set of Instances
	 * @return the merged set of Instances
	 * @throws IllegalArgumentException
	 *             if the datasets are not the same size
	 */
	public static Instances mergeInstances(Instances first, Instances second) {

		if (first.numInstances() != second.numInstances()) {
			throw new IllegalArgumentException(
					"Instance sets must be of the same size");
		}

		// Create the vector of merged attributes
		ArrayList<Attribute> newAttributes = new ArrayList<Attribute>(
				first.numAttributes() + second.numAttributes());
		for (Attribute att : first.m_Attributes) {
			newAttributes.add(att);
		}
		for (Attribute att : second.m_Attributes) {
			newAttributes.add((Attribute) att.copy()); // Need to copy because
														// indices will change.
		}

		// Create the set of Instances
		Instances merged = new Instances(first.relationName() + '_'
				+ second.relationName(), newAttributes, first.numInstances());
		// Merge each instance
		for (int i = 0; i < first.numInstances(); i++) {
			merged.add(first.instance(i).mergeInstance(second.instance(i)));
		}
		return merged;
	}

}
