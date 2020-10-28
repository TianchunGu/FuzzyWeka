

import weka.classifiers.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Clase que implementa el algoritmo kNN. MacQueen, J. (1967, June). 
 * Some methods for classification and analysis of multivariate observations.
 * In Proceedings of the fifth Berkeley symposium on mathematical statistics 
 * and probability (Vol. 1, No. 14, pp. 281-297).
 * 
 * @author Eva Gibaja
 */
public class CrispKNN extends Classifier {

	private static final long serialVersionUID = -7315265026073286088L;
	/** Instancias */
	protected Instances dataset;
	/** Tamaño del vecindario */
	protected int k;

	/**
	 * Constructor.
	 * 
	 * @param k
	 *            número de vecinos
	 */
	public CrispKNN(int k) {
		this.k = k;
	}

	@Override
	public void buildClassifier(Instances instancias) throws Exception {
		//Crea una copia del dataset
		dataset = new Instances(instancias);
	}

	@Override
	public double classifyInstance(Instance instancia) throws Exception {
		return (double) Utils.maxIndex(contarVecinos(instancia));
	}

	@Override
	public double[] distributionForInstance(Instance instancia) throws Exception {
			
		// Reserva memoria e inicializa a cero
		double predictions[] = new double[dataset.numClasses()];
		// LLama a la funcion contarVecinos
        int count[] = contarVecinos(instancia);
		
		predictions[Utils.maxIndex(count)] = 1;
		// RELLENAR: Actualizar la función para devolver la distribucion de probabilidad
		// teniendo en cuenta cuantos vecinos votan a cada clase
		/* 
		int nVotos = Utils.sum(count);
		for(int i=0; i<predictions.length; i++)
		   predictions[i]=(count[i]*1.0)/nVotos;		
		*/
        
		return predictions;
	}

	/**
	 * Cuenta cuantos vecinos de una determinada instancia votan a cada una de las clases
	 * 
	 * @param instancia
	 *            la instancia de la que se cuentan los vecinos
	 */
	private int[] contarVecinos(Instance instancia) throws Exception {

		// RELLENAR: Busqueda de los k vecinos mas cercanos
		LinearNNSearch S = new LinearNNSearch(dataset);
		S.setSkipIdentical(true);
		Instances kNN = S.kNearestNeighbours(instancia, k);

		// Registra en un vector cuántos vecinos votan a cada clase
		int count[] = new int[dataset.numClasses()];		
		for (int i = 0; i < kNN.numInstances(); i++) {
			int clase = (int) kNN.instance(i).classValue();
			// RELLENAR: Actualizar count[clase]
			count[clase]++;
		}
		return (count);

	}

}