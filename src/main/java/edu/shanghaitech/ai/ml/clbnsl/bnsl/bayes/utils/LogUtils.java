package edu.shanghaitech.ai.ml.clbnsl.bnsl.bayes.utils;

import org.apache.log4j.Logger;
import org.apache.log4j.xml.DOMConfigurator;

/**
 * @author Yanpeng Zhao
 * 4/7/2015
 */
public class LogUtils {
	
	private static LogUtils instance;
	
	private static Logger logFile = null;
	private static Logger logCons = null;
	
	
	private LogUtils() { 
		logFile = Logger.getLogger("CONSOLE");
		logCons = Logger.getLogger("FILE");
	}
	
	public static LogUtils getInstance() {
		if ( instance == null ) {
			instance = new LogUtils();
		}
		return instance;
	}
	
	public Logger getConsoleLogger(String xml) {
		// System.setProperty("log.name", "console");
		DOMConfigurator.configure(xml);
		// AUtils.printSystemProperty("log.name", false);
		return logFile;
	}
	
	public Logger getFileLogger(String xml, String log) {
		System.setProperty("log.name", log);
		DOMConfigurator.configure(xml);
		// AUtils.printSystemProperty("log.name", false);
		return logCons;
	}

}
