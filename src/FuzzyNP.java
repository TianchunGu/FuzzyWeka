import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.Instance;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Clase que implementa el algoritmo Fuzzy Nearest Prototype de Keller. Keller,
 * J. M., Gray, M. R., & Givens, J. A. (1985). A fuzzy k-nearest neighbor
 * algorithm. IEEE transactions on systems, man, and cybernetics, (4), 580-585.
 * 
 * @author Eva Gibaja
 */
public class FuzzyNP extends Classifier {

	private static final long serialVersionUID = 2710434235579627719L;
	/** Prototipos de las clases del dataset */
	protected Instances prototipos;
	/** Exponente fuzzy */
	protected double m;
	/** Tolearancia para comparar flotantes */
	protected double e = 0.0000001;

	/**
	 * Constructor
	 * 
	 * @param m
	 *            Permite ponderar los vecinos más cercanos. Cuanto mas cercano es
	 *            este valor a 1, mas influencia tienen los vecinos mas cercanos.
	 *            Usualmente se suele considerar m=2.
	 */
	public FuzzyNP(double m) {
		this.m = m;
	}

	@Override
	public void buildClassifier(Instances dataset) throws Exception {

		int classIndex = dataset.classIndex();
		// Contabiliza cuantos patrones hay de cada clase
		int classCount[] = new int[dataset.numClasses()];

		// 1. RESERVA E INICIALIZA UN VECTOR DE PROTOTIPOS
		// Genera dataset vacio
		prototipos = new Instances(dataset, dataset.numClasses());
		// Indica cual será el classIndex
		prototipos.setClassIndex(classIndex);
		for (int i = 0; i < dataset.numClasses(); i++) {
			// Añade una instancia (un prototipo) por cada clase
			Instance instance = new Instance(dataset.numAttributes());
			// Inicializa los atributos de la instancia a 0
			for (int j = 0; j < dataset.numAttributes(); j++)
				if (j == classIndex)
					// Asigna el valor de clase
					instance.setValue(j, i);
				else
					// Inicializa el valor del atributo a 0
					instance.setValue(j, 0.0);
			prototipos.add(instance);

		}

		// 2. RECORRE LOS PROTOTIPOS Y CALCULA EL VALOR MEDIO
		for (int i = 0; i < dataset.numInstances(); i++) {
			// incrementa el contador de la clase
			int classValue = (int) dataset.instance(i).classValue();
			classCount[classValue]++;
			// actualiza la suma de valores para cada atributo
			for (int j = 0; j < dataset.numAttributes(); j++) {
				if (j != classIndex) {
					double valor1 = dataset.instance(i).value(j);
					double valor2 = prototipos.instance(classValue).value(j);
					prototipos.instance(classValue).setValue(j, valor1 + valor2);
				}
			}
		}

		// 3. CALCULA EL VALOR MEDIO
		for (int i = 0; i < prototipos.numInstances(); i++) {
			for (int j = 0; j < prototipos.numAttributes(); j++) {
				if (j != classIndex) {

					// RELLENAR: Calcular el valor medio
					double valorMedio = prototipos.instance(i).value(j) / (classCount[i] * 1.0);
					// RELLENAR: Actualizar el valor delatributo en el prototipo
					prototipos.instance(i).setValue(j, valorMedio);
				}
			}
		}

	}

	@Override
	public double classifyInstance(Instance instancia) throws Exception {
		double[] u = calcularu(instancia);

		// RELLENAR: Devolver la posicion de la clase con mayor grado de pertenencia
		return Utils.maxIndex(u);

	}

	@Override
	public double[] distributionForInstance(Instance instancia) throws Exception {
		return calcularu(instancia);
	}

	/**
	 * Calcula el vector de pertenencia difusa para una determinada instancia.
	 * 
	 * @param instancia
	 *            La instancia para la que se calculará el vector de pertenencia
	 */
	private double[] calcularu(Instance instancia) throws Exception {

		// Busqueda de vecinos para poder tener la distancia de la instancia a
		// cada prototipo
		LinearNNSearch S = new LinearNNSearch(prototipos);
		S.setSkipIdentical(true);
		Instances knn = S.kNearestNeighbours(instancia, prototipos.numInstances());
		double distancias[] = S.getDistances();

		// Calcula la suma de las distancias del punto a todos los prototipos
		double suma_den = 0.0;
		for (int i = 0; i < distancias.length; i++) {
			// RELLENAR: Actualizar suma_den utilizando distancias[i]
			suma_den += 1.0 / Math.pow(distancias[i], 2.0 / (m - 1));
		}

		// Crea y da valor al vector de pertenenecias
		double u[] = new double[prototipos.numInstances()];
		for (int i = 0; i < knn.numInstances(); i++) {
			// Al estar ordenados de acuerdo a la distancia, los class value
			// estan desordenados
			int classValue = (int) knn.instance(i).classValue();
			// CASO ESPECIAL: Si distancias[i]==0 u[i]=1 (la pertenencia es total)
			//if (distancias[i] < e) {
			//	u[classValue] = 1;
			//} else
			{
				// RELLENAR: Calcular el numerador
				double num = 1 / Math.pow(distancias[i], 2.0 / (m - 1));
				// RELLENAR: Actualizar u[classValue]
				u[classValue] = num / suma_den;
			}
		}

		return (u);
	}

}
