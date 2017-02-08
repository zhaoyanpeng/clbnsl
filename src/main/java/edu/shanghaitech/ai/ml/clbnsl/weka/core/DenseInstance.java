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
 *    DenseInstance.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package edu.shanghaitech.ai.ml.clbnsl.weka.core;

/**
 * Class for handling an instance. All values (numeric, date, nominal, string or
 * relational) are internally stored as floating-point numbers. If an attribute
 * is nominal (or a string or relational), the stored value is the index of the
 * corresponding nominal (or string or relational) value in the attribute's
 * definition. We have chosen this approach in favor of a more elegant
 * object-oriented approach because it is much faster.
 * <p>
 * 
 * Typical usage (code from the main() method of this class):
 * <p>
 * 
 * <code>
 * ... <br>
 *      
 * // Create empty instance with three attribute values <br>
 * Instance inst = new DenseInstance(3); <br><br>
 *     
 * // Set instance's values for the attributes "length", "weight", and "position"<br>
 * inst.setValue(length, 5.3); <br>
 * inst.setValue(weight, 300); <br>
 * inst.setValue(position, "first"); <br><br>
 *   
 * // Set instance's dataset to be the dataset "race" <br>
 * inst.setDataset(race); <br><br>
 *   
 * // Print the instance <br>
 * System.out.println("The instance: " + inst); <br>
 * 
 * ... <br>
 * </code>
 * <p>
 * 
 * All methods that change an instance's attribute values are safe, ie. a change
 * of an instance's attribute values does not affect any other instances. All
 * methods that change an instance's attribute values clone the attribute value
 * vector before it is changed. If your application heavily modifies instance
 * values, it may be faster to create a new instance from scratch.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 10203 $
 */
public class DenseInstance extends AbstractInstance {

	/** for serialization */
	static final long serialVersionUID = 1482635194499365122L;

	/**
	 * Constructor that copies the attribute values and the weight from the
	 * given instance. It does NOT perform a deep copy of the attribute values
	 * if the instance provided is also of type DenseInstance (it simply copies
	 * the reference to the array of values), otherwise it does. Reference to
	 * the dataset is set to null. (ie. the instance doesn't have access to
	 * information about the attribute types)
	 * 
	 * @param instance
	 *            the instance from which the attribute values and the weight
	 *            are to be copied
	 */
	// @ ensures m_Dataset == null;
	public DenseInstance(/* @non_null@ */Instance instance) {

		if (instance instanceof DenseInstance) {
			m_AttValues = ((DenseInstance) instance).m_AttValues;
		} else {
			m_AttValues = instance.toDoubleArray();
		}
		m_Weight = instance.weight();
		m_Dataset = null;
	}

	/**
	 * Constructor that inititalizes instance variable with given values.
	 * Reference to the dataset is set to null. (ie. the instance doesn't have
	 * access to information about the attribute types)
	 * 
	 * @param weight
	 *            the instance's weight
	 * @param attValues
	 *            a vector of attribute values
	 */
	// @ ensures m_Dataset == null;
	public DenseInstance(double weight, /* @non_null@ */double[] attValues) {

		m_AttValues = attValues;
		m_Weight = weight;
		m_Dataset = null;
	}

	/**
	 * Constructor of an instance that sets weight to one, all values to be
	 * missing, and the reference to the dataset to null. (ie. the instance
	 * doesn't have access to information about the attribute types)
	 * 
	 * @param numAttributes
	 *            the size of the instance
	 */
	// @ requires numAttributes > 0; // Or maybe == 0 is okay too?
	// @ ensures m_Dataset == null;
	public DenseInstance(int numAttributes) {

		m_AttValues = new double[numAttributes];
		for (int i = 0; i < m_AttValues.length; i++) {
			m_AttValues[i] = Utils.missingValue();
		}
		m_Weight = 1;
		m_Dataset = null;
	}

	/**
	 * Produces a shallow copy of this instance. The copy has access to the same
	 * dataset. (if you want to make a copy that doesn't have access to the
	 * dataset, use <code>new DenseInstance(instance)</code>
	 * 
	 * @return the shallow copy
	 */
	// @ also ensures \result != null;
	// @ also ensures \result instanceof DenseInstance;
	// @ also ensures ((DenseInstance)\result).m_Dataset == m_Dataset;
	@Override
	public/* @pure@ */Object copy() {

		DenseInstance result = new DenseInstance(this);
		result.m_Dataset = m_Dataset;
		return result;
	}

	/**
	 * Returns the index of the attribute stored at the given position. Just
	 * returns the given value.
	 * 
	 * @param position
	 *            the position
	 * @return the index of the attribute stored at the given position
	 */
	@Override
	public/* @pure@ */int index(int position) {

		return position;
	}

	/**
	 * Merges this instance with the given instance and returns the result.
	 * Dataset is set to null. The returned instance is of the same type as this
	 * instance.
	 * 
	 * @param inst
	 *            the instance to be merged with this one
	 * @return the merged instances
	 */
	@Override
	public Instance mergeInstance(Instance inst) {

		int m = 0;
		double[] newVals = new double[numAttributes() + inst.numAttributes()];
		for (int j = 0; j < numAttributes(); j++, m++) {
			newVals[m] = value(j);
		}
		for (int j = 0; j < inst.numAttributes(); j++, m++) {
			newVals[m] = inst.value(j);
		}
		return new DenseInstance(1.0, newVals);
	}

	/**
	 * Returns the number of attributes.
	 * 
	 * @return the number of attributes as an integer
	 */
	// @ ensures \result == m_AttValues.length;
	@Override
	public/* @pure@ */int numAttributes() {

		return m_AttValues.length;
	}

	/**
	 * Returns the number of values present. Always the same as numAttributes().
	 * 
	 * @return the number of values
	 */
	// @ ensures \result == m_AttValues.length;
	@Override
	public/* @pure@ */int numValues() {

		return m_AttValues.length;
	}

	/**
	 * Replaces all missing values in the instance with the values contained in
	 * the given array. A deep copy of the vector of attribute values is
	 * performed before the values are replaced.
	 * 
	 * @param array
	 *            containing the means and modes
	 * @throws IllegalArgumentException
	 *             if numbers of attributes are unequal
	 */
	@Override
	public void replaceMissingValues(double[] array) {

		if ((array == null) || (array.length != m_AttValues.length)) {
			throw new IllegalArgumentException("Unequal number of attributes!");
		}
		freshAttributeVector();
		for (int i = 0; i < m_AttValues.length; i++) {
			if (isMissing(i)) {
				m_AttValues[i] = array[i];
			}
		}
	}

	/**
	 * Sets a specific value in the instance to the given value (internal
	 * floating-point format). Performs a deep copy of the vector of attribute
	 * values before the value is set.
	 * 
	 * @param attIndex
	 *            the attribute's index
	 * @param value
	 *            the new attribute value (If the corresponding attribute is
	 *            nominal (or a string) then this is the new value's index as a
	 *            double).
	 */
	@Override
	public void setValue(int attIndex, double value) {

		freshAttributeVector();
		m_AttValues[attIndex] = value;
	}

	/**
	 * Sets a specific value in the instance to the given value (internal
	 * floating-point format). Performs a deep copy of the vector of attribute
	 * values before the value is set. Does exactly the same thing as
	 * setValue().
	 * 
	 * @param indexOfIndex
	 *            the index of the attribute's index
	 * @param value
	 *            the new attribute value (If the corresponding attribute is
	 *            nominal (or a string) then this is the new value's index as a
	 *            double).
	 */
	@Override
	public void setValueSparse(int indexOfIndex, double value) {

		freshAttributeVector();
		m_AttValues[indexOfIndex] = value;
	}

	/**
	 * Returns the values of each attribute as an array of doubles.
	 * 
	 * @return an array containing all the instance attribute values
	 */
	@Override
	public double[] toDoubleArray() {

		double[] newValues = new double[m_AttValues.length];
		System.arraycopy(m_AttValues, 0, newValues, 0, m_AttValues.length);
		return newValues;
	}

	/**
	 * Returns the description of one instance (without weight appended). If the
	 * instance doesn't have access to a dataset, it returns the internal
	 * floating-point values. Quotes string values that contain whitespace
	 * characters.
	 * 
	 * This method is used by getRandomNumberGenerator() in Instances.java in
	 * order to maintain backwards compatibility with weka 3.4.
	 * 
	 * @return the instance's description as a string
	 */
	@Override
	public String toStringNoWeight() {
		return toStringNoWeight(AbstractInstance.s_numericAfterDecimalPoint);
	}

	/**
	 * Returns the description of one instance (without weight appended). If the
	 * instance doesn't have access to a dataset, it returns the internal
	 * floating-point values. Quotes string values that contain whitespace
	 * characters.
	 * 
	 * This method is used by getRandomNumberGenerator() in Instances.java in
	 * order to maintain backwards compatibility with weka 3.4.
	 * 
	 * @param afterDecimalPoint
	 *            maximum number of digits after the decimal point for numeric
	 *            values
	 * 
	 * @return the instance's description as a string
	 */
	@Override
	public String toStringNoWeight(int afterDecimalPoint) {
		StringBuffer text = new StringBuffer();

		for (int i = 0; i < m_AttValues.length; i++) {
			if (i > 0) {
				text.append(",");
			}
			text.append(toString(i, afterDecimalPoint));
		}

		return text.toString();
	}

	/**
	 * Returns an instance's attribute value in internal format.
	 * 
	 * @param attIndex
	 *            the attribute's index
	 * @return the specified value as a double (If the corresponding attribute
	 *         is nominal (or a string) then it returns the value's index as a
	 *         double).
	 */
	@Override
	public/* @pure@ */double value(int attIndex) {

		return m_AttValues[attIndex];
	}

	/**
	 * Deletes an attribute at the given position (0 to numAttributes() - 1).
	 * 
	 * @param position
	 *            the attribute's position
	 */
	@Override
	protected void forceDeleteAttributeAt(int position) {

		double[] newValues = new double[m_AttValues.length - 1];

		System.arraycopy(m_AttValues, 0, newValues, 0, position);
		if (position < m_AttValues.length - 1) {
			System.arraycopy(m_AttValues, position + 1, newValues, position,
					m_AttValues.length - (position + 1));
		}
		m_AttValues = newValues;
	}

	/**
	 * Inserts an attribute at the given position (0 to numAttributes()) and
	 * sets its value to be missing.
	 * 
	 * @param position
	 *            the attribute's position
	 */
	@Override
	protected void forceInsertAttributeAt(int position) {

		double[] newValues = new double[m_AttValues.length + 1];

		System.arraycopy(m_AttValues, 0, newValues, 0, position);
		newValues[position] = Utils.missingValue();
		System.arraycopy(m_AttValues, position, newValues, position + 1,
				m_AttValues.length - position);
		m_AttValues = newValues;
	}

	/**
	 * Clones the attribute vector of the instance and overwrites it with the
	 * clone.
	 */
	private void freshAttributeVector() {

		m_AttValues = toDoubleArray();
	}
}
