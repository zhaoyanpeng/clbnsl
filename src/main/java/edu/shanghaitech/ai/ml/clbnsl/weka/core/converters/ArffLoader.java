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
 *    ArffLoader.java
 *    Copyright (C) 2000-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package edu.shanghaitech.ai.ml.clbnsl.weka.core.converters;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.io.StringReader;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;

import edu.shanghaitech.ai.ml.clbnsl.weka.core.Attribute;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.DenseInstance;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instance;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;
import edu.shanghaitech.ai.ml.clbnsl.weka.core.Utils;

/**
 * <!-- globalinfo-start --> Reads a source that is in arff (attribute relation
 * file format) format.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 11136 $
 * @see Loader
 */
public class ArffLoader extends AbstractFileLoader {

	/** for serialization */
	static final long serialVersionUID = 2726929550544048587L;

	/** the file extension */
	public static String FILE_EXTENSION = Instances.FILE_EXTENSION;
	public static String FILE_EXTENSION_COMPRESSED = FILE_EXTENSION + ".gz";

	/** The reader for the source file. */
	protected transient Reader m_sourceReader = null;

	/** The parser for the ARFF file */
	protected transient ArffReader m_ArffReader = null;

	/**
	 * Whether the values of string attributes should be retained in memory when
	 * reading incrementally
	 */
	protected boolean m_retainStringVals;

	/**
	 * Reads data from an ARFF file, either in incremental or batch mode.
	 * <p/>
	 * 
	 * Typical code for batch usage:
	 * 
	 * <pre>
	 * BufferedReader reader = new BufferedReader(new FileReader(
	 * 		&quot;/some/where/file.arff&quot;));
	 * ArffReader arff = new ArffReader(reader);
	 * Instances data = arff.getData();
	 * data.setClassIndex(data.numAttributes() - 1);
	 * </pre>
	 * 
	 * Typical code for incremental usage:
	 * 
	 * <pre>
	 * BufferedReader reader = new BufferedReader(new FileReader(
	 * 		&quot;/some/where/file.arff&quot;));
	 * ArffReader arff = new ArffReader(reader, 1000);
	 * Instances data = arff.getStructure();
	 * data.setClassIndex(data.numAttributes() - 1);
	 * Instance inst;
	 * while ((inst = arff.readInstance(data)) != null) {
	 * 	data.add(inst);
	 * }
	 * </pre>
	 * 
	 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
	 * @author Len Trigg (trigg@cs.waikato.ac.nz)
	 * @author fracpete (fracpete at waikato dot ac dot nz)
	 * @version $Revision: 11136 $
	 */
	public static class ArffReader {

		/** the tokenizer for reading the stream */
		protected StreamTokenizer m_Tokenizer;

		protected List<Integer> m_stringAttIndices;

		/** the actual data */
		protected Instances m_Data;

		/** the number of lines read so far */
		protected int m_Lines;

		protected boolean m_batchMode = true;

		/**
		 * Whether the values for string attributes will accumulate in the
		 * header when reading incrementally
		 */
		protected boolean m_retainStringValues = false;

		/**
		 * Reads the data completely from the reader. The data can be accessed
		 * via the <code>getData()</code> method.
		 * 
		 * @param reader
		 *            the reader to use
		 * @throws IOException
		 *             if something goes wrong
		 * @see #getData()
		 */
		public ArffReader(Reader reader) throws IOException {
			m_retainStringValues = true;
			m_batchMode = true;
			m_Tokenizer = new StreamTokenizer(reader);
			initTokenizer();

			readHeader(1000);

			Instance inst;
			while ((inst = readInstance(m_Data)) != null) {
				m_Data.add(inst);
			}

			compactify();
		}

		public ArffReader(Reader reader, int capacity) throws IOException {
			this(reader, capacity, true);
		}

		/**
		 * Reads only the header and reserves the specified space for instances.
		 * Further instances can be read via <code>readInstance()</code>.
		 * 
		 * @param reader
		 *            the reader to use
		 * @param capacity
		 *            the capacity of the new dataset
		 * @param batch
		 *            true if reading in batch mode
		 * @throws IOException
		 *             if something goes wrong
		 * @throws IOException
		 *             if a problem occurs
		 * @see #getStructure()
		 * @see #readInstance(Instances)
		 */
		public ArffReader(Reader reader, int capacity, boolean batch)
				throws IOException {

			m_batchMode = batch;
			if (batch) {
				m_retainStringValues = true;
			}

			if (capacity < 0) {
				throw new IllegalArgumentException(
						"Capacity has to be positive!");
			}

			m_Tokenizer = new StreamTokenizer(reader);
			initTokenizer();

			readHeader(capacity);
		}

		/**
		 * Reads the data without header according to the specified template.
		 * The data can be accessed via the <code>getData()</code> method.
		 * 
		 * @param reader
		 *            the reader to use
		 * @param template
		 *            the template header
		 * @param lines
		 *            the lines read so far
		 * @param fieldSepAndEnclosures
		 *            an optional array of Strings containing the field
		 *            separator and enclosures to use instead of the defaults.
		 *            The first entry in the array is expected to be the single
		 *            character field separator to use; the remaining entries
		 *            (if any) are enclosure characters to use.
		 * @throws IOException
		 *             if something goes wrong
		 * @see #getData()
		 */
		public ArffReader(Reader reader, Instances template, int lines,
				String... fieldSepAndEnclosures) throws IOException {
			this(reader, template, lines, 100, true, fieldSepAndEnclosures);

			Instance inst;
			while ((inst = readInstance(m_Data)) != null) {
				m_Data.add(inst);
			}

			compactify();
		}

		/**
		 * Initializes the reader without reading the header according to the
		 * specified template. The data must be read via the
		 * <code>readInstance()</code> method.
		 * 
		 * @param reader
		 *            the reader to use
		 * @param template
		 *            the template header
		 * @param lines
		 *            the lines read so far
		 * @param capacity
		 *            the capacity of the new dataset
		 * @param fieldSepAndEnclosures
		 *            an optional array of Strings containing the field
		 *            separator and enclosures to use instead of the defaults.
		 *            The first entry in the array is expected to be the single
		 *            character field separator to use; the remaining entries
		 *            (if any) are enclosure characters to use.
		 * @throws IOException
		 *             if something goes wrong
		 * @see #getData()
		 */
		public ArffReader(Reader reader, Instances template, int lines,
				int capacity, String... fieldSepAndEnclosures)
				throws IOException {
			this(reader, template, lines, capacity, false,
					fieldSepAndEnclosures);
		}

		/**
		 * Initializes the reader without reading the header according to the
		 * specified template. The data must be read via the
		 * <code>readInstance()</code> method.
		 * 
		 * @param reader
		 *            the reader to use
		 * @param template
		 *            the template header
		 * @param lines
		 *            the lines read so far
		 * @param capacity
		 *            the capacity of the new dataset
		 * @param batch
		 *            true if the data is going to be read in batch mode
		 * @param fieldSepAndEnclosures
		 *            an optional array of Strings containing the field
		 *            separator and enclosures to use instead of the defaults.
		 *            The first entry in the array is expected to be the single
		 *            character field separator to use; the remaining entries
		 *            (if any) are enclosure characters to use.
		 * @throws IOException
		 *             if something goes wrong
		 * @see #getData()
		 */
		public ArffReader(Reader reader, Instances template, int lines,
				int capacity, boolean batch, String... fieldSepAndEnclosures)
				throws IOException {
			m_batchMode = batch;
			if (batch) {
				m_retainStringValues = true;
			}

			if (fieldSepAndEnclosures != null
					&& fieldSepAndEnclosures.length > 0) {

			}

			m_Lines = lines;
			m_Tokenizer = new StreamTokenizer(reader);
			initTokenizer();

			m_Data = new Instances(template, capacity);
		}

		/**
		 * compactifies the data
		 */
		protected void compactify() {
			if (m_Data != null) {
				m_Data.compactify();
			}
		}

		/**
		 * Throws error message with line number and last token read.
		 * 
		 * @param msg
		 *            the error message to be thrown
		 * @throws IOException
		 *             containing the error message
		 */
		protected void errorMessage(String msg) throws IOException {

			String str = msg + ", read " + m_Tokenizer.toString();

			// System.out.println("+@deubg: " + str);

			if (m_Lines > 0) {
				int line = Integer.parseInt(str.replaceAll(".* line ", ""));
				str = str.replaceAll(" line .*", " line "
						+ (m_Lines + line - 1));
			}
			throw new IOException(str);
		}

		/**
		 * returns the current line number
		 * 
		 * @return the current line number
		 */
		public int getLineNo() {
			return m_Lines + m_Tokenizer.lineno();
		}

		/**
		 * Gets index, checking for a premature and of line.
		 * 
		 * @throws IOException
		 *             if it finds a premature end of line
		 */
		protected void getIndex() throws IOException {
			if (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {
				errorMessage("premature end of line");
			}
			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				errorMessage("premature end of file");
			}
		}

		/**
		 * Reads a single instance using the tokenizer and returns it.
		 * 
		 * @param structure
		 *            the dataset header information, will get updated in case
		 *            of string or relational attributes
		 * @return null if end of file has been reached
		 * @throws IOException
		 *             if the information is not read successfully
		 */
		public Instance readInstance(Instances structure) throws IOException {
			return readInstance(structure, true);
		}

		/**
		 * Reads a single instance using the tokenizer and returns it.
		 * 
		 * @param structure
		 *            the dataset header information, will get updated in case
		 *            of string or relational attributes
		 * @param flag
		 *            if method should test for carriage return after each
		 *            instance
		 * @return null if end of file has been reached
		 * @throws IOException
		 *             if the information is not read successfully
		 */
		public Instance readInstance(Instances structure, boolean flag)
				throws IOException {
			return getInstance(structure, flag);
		}

		/**
		 * Reads a single instance using the tokenizer and returns it.
		 * 
		 * @param structure
		 *            the dataset header information, will get updated in case
		 *            of string or relational attributes
		 * @param flag
		 *            if method should test for carriage return after each
		 *            instance
		 * @return null if end of file has been reached
		 * @throws IOException
		 *             if the information is not read successfully
		 */
		protected Instance getInstance(Instances structure, boolean flag)
				throws IOException {
			m_Data = structure;

			// Check if any attributes have been declared.
			if (m_Data.numAttributes() == 0) {
				errorMessage("no header information available");
			}

			// Check if end of file reached.
			getFirstToken();
			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				return null;
			}

			return getInstanceFull(flag);

		}

		/**
		 * Reads a single instance using the tokenizer and returns it.
		 * 
		 * @param flag
		 *            if method should test for carriage return after each
		 *            instance
		 * @return null if end of file has been reached
		 * @throws IOException
		 *             if the information is not read successfully
		 */
		protected Instance getInstanceFull(boolean flag) throws IOException {
			double[] instance = new double[m_Data.numAttributes()];
			int index;

			// Get values for all attributes.
			for (int i = 0; i < m_Data.numAttributes(); i++) {
				// Get next token
				if (i > 0) {
					getNextToken();
				}

				// Check if value is missing.
				if (m_Tokenizer.ttype == '?') {
					instance[i] = Utils.missingValue();
				} else {

					// Check if token is valid.
					if (m_Tokenizer.ttype != StreamTokenizer.TT_WORD) {
						errorMessage("not a valid value");
					}

					switch (m_Data.attribute(i).type()) {

					case Attribute.NOMINAL:
						// Check if value appears in header.
						index = m_Data.attribute(i).indexOfValue(
								m_Tokenizer.sval);

						// System.out.println(i + "--" + m_Tokenizer.sval);

						if (index == -1) {
							errorMessage("nominal value not declared in header");
						}

						instance[i] = index;
						break;
					case Attribute.NUMERIC:
						// Check if value is really a number.
						try {
							instance[i] = Double.valueOf(m_Tokenizer.sval)
									.doubleValue();
						} catch (NumberFormatException e) {
							errorMessage("number expected");
						}
						break;
					case Attribute.STRING:
						if (m_batchMode || m_retainStringValues) {
							instance[i] = m_Data.attribute(i).addStringValue(
									m_Tokenizer.sval);
						} else {
							instance[i] = 0;
							m_Data.attribute(i)
									.setStringValue(m_Tokenizer.sval);
						}
						break;
					case Attribute.DATE:
						try {
							instance[i] = m_Data.attribute(i).parseDate(
									m_Tokenizer.sval);
						} catch (ParseException e) {
							errorMessage("unparseable date: "
									+ m_Tokenizer.sval);
						}
						break;
					case Attribute.RELATIONAL:
						try {
							ArffReader arff = new ArffReader(new StringReader(
									m_Tokenizer.sval), m_Data.attribute(i)
									.relation(), 0);
							Instances data = arff.getData();
							instance[i] = m_Data.attribute(i).addRelation(data);
						} catch (Exception e) {
							throw new IOException(e.toString() + " of line "
									+ getLineNo());
						}
						break;
					default:
						errorMessage("unknown attribute type in column " + i);
					}
				}
			}

			double weight = 1.0;
			if (flag) {

				weight = 1.0;

			}

			// Add instance to dataset
			Instance inst = new DenseInstance(weight, instance);
			inst.setDataset(m_Data);

			return inst;
		}

		/**
		 * Initializes the StreamTokenizer used for reading the ARFF file.
		 */
		protected void initTokenizer() {
			m_Tokenizer.resetSyntax();
			m_Tokenizer.whitespaceChars(0, ' ');
			m_Tokenizer.wordChars(' ' + 1, '\u00FF');
			m_Tokenizer.whitespaceChars(',', ',');
			m_Tokenizer.commentChar('%');
			m_Tokenizer.quoteChar('"');
			m_Tokenizer.quoteChar('\'');
			m_Tokenizer.ordinaryChar('{');
			m_Tokenizer.ordinaryChar('}');
			m_Tokenizer.eolIsSignificant(true);
		}

		/**
		 * Gets next token, skipping empty lines.
		 * 
		 * @throws IOException
		 *             if reading the next token fails
		 */
		protected void getFirstToken() throws IOException {

			while (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {
			}

			if ((m_Tokenizer.ttype == '\'') || (m_Tokenizer.ttype == '"')) {
				m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
			} else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD)
					&& (m_Tokenizer.sval.equals("?"))) {
				m_Tokenizer.ttype = '?';
			}
		}

		/**
		 * Gets token and checks if its end of line.
		 * 
		 * @param endOfFileOk
		 *            whether EOF is OK
		 * @throws IOException
		 *             if it doesn't find an end of line
		 */
		protected void getLastToken(boolean endOfFileOk) throws IOException {
			if ((m_Tokenizer.nextToken() != StreamTokenizer.TT_EOL)
					&& ((m_Tokenizer.ttype != StreamTokenizer.TT_EOF) || !endOfFileOk)) {
				errorMessage("end of line expected");
			}
		}

		/**
		 * Gets next token, checking for a premature and of line.
		 * 
		 * @throws IOException
		 *             if it finds a premature end of line
		 */
		protected void getNextToken() throws IOException {
			if (m_Tokenizer.nextToken() == StreamTokenizer.TT_EOL) {
				errorMessage("premature end of line");
			}

			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				errorMessage("premature end of file");
			} else if ((m_Tokenizer.ttype == '\'')
					|| (m_Tokenizer.ttype == '"')) {
				m_Tokenizer.ttype = StreamTokenizer.TT_WORD;
			} else if ((m_Tokenizer.ttype == StreamTokenizer.TT_WORD)
					&& (m_Tokenizer.sval.equals("?"))) {
				m_Tokenizer.ttype = '?';
			}
		}

		/**
		 * Reads and stores header of an ARFF file.
		 * 
		 * @param capacity
		 *            the number of instances to reserve in the data structure
		 * @throws IOException
		 *             if the information is not read successfully
		 */
		protected void readHeader(int capacity) throws IOException {
			m_Lines = 0;
			String relationName = "";

			// Get name of relation.
			getFirstToken();

			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				errorMessage("premature end of file");
			}

			if (Instances.ARFF_RELATION.equalsIgnoreCase(m_Tokenizer.sval)) {
				getNextToken();
				relationName = m_Tokenizer.sval;
				getLastToken(false);
			} else {
				errorMessage("keyword " + Instances.ARFF_RELATION + " expected");
			}

			// Create vectors to hold information temporarily.
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();

			// Get attribute declarations.
			getFirstToken();
			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				errorMessage("premature end of file");
			}

			while (Attribute.ARFF_ATTRIBUTE.equalsIgnoreCase(m_Tokenizer.sval)) {
				attributes = parseAttribute(attributes);
			}

			// Check if data part follows. We can't easily check for EOL.
			if (!Instances.ARFF_DATA.equalsIgnoreCase(m_Tokenizer.sval)) {
				errorMessage("keyword " + Instances.ARFF_DATA + " expected");
			}

			// Check if any attributes have been declared.
			if (attributes.size() == 0) {
				errorMessage("no attributes declared");
			}

			m_Data = new Instances(relationName, attributes, capacity);
		}

		/**
		 * Parses the attribute declaration.
		 * 
		 * @param attributes
		 *            the current attributes vector
		 * @return the new attributes vector
		 * @throws IOException
		 *             if the information is not read successfully
		 */
		protected ArrayList<Attribute> parseAttribute(
				ArrayList<Attribute> attributes) throws IOException {
			String attributeName;
			ArrayList<String> attributeValues;

			// Get attribute name.
			getNextToken();
			attributeName = m_Tokenizer.sval;
			getNextToken();

			// Attribute is nominal.
			attributeValues = new ArrayList<String>();
			m_Tokenizer.pushBack();

			// Get values for nominal attribute.
			if (m_Tokenizer.nextToken() != '{') {
				errorMessage("{ expected at beginning of enumeration");
			}
			while (m_Tokenizer.nextToken() != '}') {
				if (m_Tokenizer.ttype == StreamTokenizer.TT_EOL) {
					errorMessage("} expected at end of enumeration");
				} else {
					attributeValues.add(m_Tokenizer.sval);
				}
			}
			attributes.add(new Attribute(attributeName, attributeValues,
					attributes.size()));

			getLastToken(false);
			getFirstToken();
			if (m_Tokenizer.ttype == StreamTokenizer.TT_EOF) {
				errorMessage("premature end of file");
			}

			return attributes;
		}

		/**
		 * Returns the header format
		 * 
		 * @return the header format
		 */
		public Instances getStructure() {
			return new Instances(m_Data, 0);
		}

		/**
		 * Returns the data that was read
		 * 
		 * @return the data
		 */
		public Instances getData() {
			return m_Data;
		}

		/**
		 * Set whether to retain the values of string attributes in memory (in
		 * the header) when reading incrementally.
		 * 
		 * @param retain
		 *            true if string values are to be retained in memory when
		 *            reading incrementally
		 */
		public void setRetainStringValues(boolean retain) {
			m_retainStringValues = retain;
		}

		/**
		 * Get whether to retain the values of string attributes in memory (in
		 * the header) when reading incrementally.
		 * 
		 * @return true if string values are to be retained in memory when
		 *         reading incrementally
		 */
		public boolean getRetainStringValues() {
			return m_retainStringValues;
		}
	}

	/**
	 * Set whether to retain the values of string attributes in memory (in the
	 * header) when reading incrementally.
	 * 
	 * @param retain
	 *            true if string values are to be retained in memory when
	 *            reading incrementally
	 */
	public void setRetainStringVals(boolean retain) {
		m_retainStringVals = retain;
	}

	/**
	 * Get whether to retain the values of string attributes in memory (in the
	 * header) when reading incrementally.
	 * 
	 * @return true if string values are to be retained in memory when reading
	 *         incrementally
	 */
	public boolean getRetainStringVals() {
		return m_retainStringVals;
	}

	/**
	 * Resets the Loader ready to read a new data set or the same data set
	 * again.
	 * 
	 * @throws IOException
	 *             if something goes wrong
	 */
	@Override
	public void reset() throws IOException {
		m_structure = null;
		m_ArffReader = null;
		setRetrieval(NONE);

		if (m_File != null && !(new File(m_File).isDirectory())) {
			setFile(new File(m_File));
		}
	}

	/**
	 * sets the source File
	 * 
	 * @param file
	 *            the source file
	 * @throws IOException
	 *             if an error occurs
	 */
	@Override
	public void setFile(File file) throws IOException {
		m_File = file.getPath();
		setSource(file);
	}

	/**
	 * Resets the Loader object and sets the source of the data set to be the
	 * supplied InputStream.
	 * 
	 * @param in
	 *            the source InputStream.
	 * @throws IOException
	 *             always thrown.
	 */
	@Override
	public void setSource(InputStream in) throws IOException {
		m_File = (new File(System.getProperty("user.dir"))).getAbsolutePath();

		m_sourceReader = new BufferedReader(new InputStreamReader(in));
	}

	/**
	 * Determines and returns (if possible) the structure (internally the
	 * header) of the data set as an empty set of instances.
	 * 
	 * @return the structure of the data set as an empty set of Instances
	 * @throws IOException
	 *             if an error occurs
	 */
	@Override
	public Instances getStructure() throws IOException {

		if (m_structure == null) {

			if (m_sourceReader == null) {
				throw new IOException("No source has been specified");
			}

			try {
				m_ArffReader = new ArffReader(m_sourceReader, 1,
						(getRetrieval() == BATCH));
				m_ArffReader.setRetainStringValues(getRetainStringVals());
				m_structure = m_ArffReader.getStructure();
			} catch (Exception ex) {
				throw new IOException(
						"Unable to determine structure as arff (Reason: "
								+ ex.toString() + ").");
			}
		}

		return new Instances(m_structure, 0);
	}

	/**
	 * Return the full data set. If the structure hasn't yet been determined by
	 * a call to getStructure then method should do so before processing the
	 * rest of the data set.
	 * 
	 * @return the structure of the data set as an empty set of Instances
	 * @throws IOException
	 *             if there is no source or parsing fails
	 */
	@Override
	public Instances getDataSet() throws IOException {

		Instances insts = null;
		try {
			if (m_sourceReader == null) {
				throw new IOException("No source has been specified");
			}

			if (getRetrieval() == INCREMENTAL) {
				throw new IOException(
						"Cannot mix getting Instances in both incremental and batch modes");
			}

			setRetrieval(BATCH);

			if (m_structure == null) {
				getStructure();
			}

			// Read all instances
			insts = new Instances(m_structure, 0);
			Instance inst;
			while ((inst = m_ArffReader.readInstance(m_structure)) != null) {

				insts.add(inst);
			}

		} finally {
			if (m_sourceReader != null) {
				// close the stream
				m_sourceReader.close();
			}
		}

		return insts;
	}

	/**
	 * Read the data set incrementally---get the next instance in the data set
	 * or returns null if there are no more instances to get. If the structure
	 * hasn't yet been determined by a call to getStructure then method should
	 * do so before returning the next instance in the data set.
	 * 
	 * @param structure
	 *            the dataset header information, will get updated in case of
	 *            string or relational attributes
	 * @return the next instance in the data set as an Instance object or null
	 *         if there are no more instances to be read
	 * @throws IOException
	 *             if there is an error during parsing
	 */
	@Override
	public Instance getNextInstance(Instances structure) throws IOException {

		m_structure = structure;

		if (getRetrieval() == BATCH) {
			throw new IOException(
					"Cannot mix getting Instances in both incremental and batch modes");
		}
		setRetrieval(INCREMENTAL);

		Instance current = null;
		if (m_sourceReader != null) {
			current = m_ArffReader.readInstance(m_structure);
		}

		if ((m_sourceReader != null) && (current == null)) {
			try {
				// close the stream
				m_sourceReader.close();
				m_sourceReader = null;
				// reset();
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}
		return current;
	}

}
