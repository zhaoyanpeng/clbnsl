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
 * AbstractFileLoader.java
 * Copyright (C) 2006-2012 University of Waikato, Hamilton, New Zealand
 */

package edu.shanghaitech.ai.ml.clbnsl.weka.core.converters;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import edu.shanghaitech.ai.ml.clbnsl.weka.core.Instances;

/**
 * Abstract superclass for all file loaders.
 * 
 * @author fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 10203 $
 */
public abstract class AbstractFileLoader extends AbstractLoader {

	/** the file */
	protected String m_File = (new File(System.getProperty("user.dir")))
			.getAbsolutePath();

	/** Holds the determined structure (header) of the data set. */
	protected transient Instances m_structure = null;

	/** Holds the source of the data set. */
	protected File m_sourceFile = null;

	/**
	 * sets the source File
	 * 
	 * @param file
	 *            the source file
	 * @exception IOException
	 *                if an error occurs
	 */
	public void setFile(File file) throws IOException {
		m_structure = null;
		setRetrieval(NONE);

		// m_File = file.getAbsolutePath();
		setSource(file);
	}

	/**
	 * Resets the Loader object and sets the source of the data set to be the
	 * supplied File object.
	 * 
	 * @param file
	 *            the source file.
	 * @throws IOException
	 *             if an error occurs
	 */
	@Override
	public void setSource(File file) throws IOException {
		File original = file;
		m_structure = null;

		setRetrieval(NONE);

		if (file == null) {
			throw new IOException("Source file object is null!");
		}

		String fName = file.getPath();

		file = new File(fName);
		// set the source only if the file exists
		if (file.exists() && file.isFile()) {

			setSource(new FileInputStream(file));

		} else {

			// forward slashes are platform independent for loading from the
			// classpath...
			String fnameWithCorrectSeparators = fName.replace(
					File.separatorChar, '/');
			if (this.getClass().getClassLoader()
					.getResource(fnameWithCorrectSeparators) != null) {
				// System.out.println("Found resource in classpath...");
				setSource(this.getClass().getClassLoader()
						.getResourceAsStream(fnameWithCorrectSeparators));
			}
		}

		m_sourceFile = original;
		m_File = m_sourceFile.getPath();
	}
}
