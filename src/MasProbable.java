import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/*Clasificador que devuelve siempre la clase mas probable en el dataset*/
public class MasProbable extends Classifier {

	private static final long serialVersionUID = 1L;

	// Vector con la frecuencia de cada clase
	double frecuencias[];
	
	public void buildClassifier(Instances data) throws Exception {		
        
		frecuencias = new double[data.numClasses()];

		// RELLENAR: Contabilizar las frecuencias de cada clase
		for (int i = 0; i < data.numInstances(); i++) {
			frecuencias[(int) data.instance(i).classValue()]++;			
		}	
		
	    for (int i=0; i<frecuencias.length; i++)
	    	frecuencias[i]/= data.numInstances();
	}

	public double classifyInstance(Instance instancia) {
		return Utils.maxIndex(frecuencias);
	}

	public double[] distributionForInstance(Instance instancia) {
		return Arrays.copyOf(frecuencias, frecuencias.length);			
		
	}

}
