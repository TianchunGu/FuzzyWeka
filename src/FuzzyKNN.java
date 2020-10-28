import weka.classifiers.*;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;

/**
 * Clase que implementa el algoritmo fuzzy kNN de Keller. Keller, J. M., Gray,
 * M. R., & Givens, J. A. (1985). A fuzzy k-nearest neighbor algorithm. IEEE
 * transactions on systems, man, and cybernetics, (4), 580-585.
 * 
 * @author Eva Gibaja
 */
public class FuzzyKNN extends Classifier {

	private static final long serialVersionUID = 1L;

	/** Instancias */
	protected Instances dataset;
	/** Tamaño del vecindario */
	protected int k;
	/** Matriz de particion de num_clases x num_instances */
	protected double U[][];
	/** Tamaño del vecindario para la inicialización de la matriz U */
	protected int kini;
	/** Exponente fuzzy */
	protected double m;
	/** Tipos de inicializacion */
	protected int ini;
	/** Para hacer las busquedas de vecinos */
	LinearNNESearch S;
	/** Tolearancia para comparar flotantes */
	protected double e = 0.0000001;

	/**
	 * Constructor.
	 * 
	 * @param k
	 *            número de vecinos
	 * @param m
	 *            Permite ponderar los vecinos más cercanos. Cuanto mas cercano es
	 *            este valor a 1, mas influencia tienen los vecinos mas cercanos.
	 *            Usualmente se suele considerar m=2
	 * @param ini
	 *            Tipo de inicializcion de U: 1-crisp, 2-fuzzy
	 */
	public FuzzyKNN(int k, double m, int ini) {
		this.k = k;
		this.m = m;
		this.ini = ini;
		this.kini = k; // Por defecto kini = k
	}

	@Override
	public void buildClassifier(Instances instancias) throws Exception {
		dataset = new Instances(instancias);

		S = new LinearNNESearch(dataset);
		S.setSkipIdentical(true); //para evitar problemas porque la distancia se haga cero

		U = new double[dataset.numClasses()][dataset.numInstances()];
		if (ini == 1)
			inicializacionCrisp();
		else
			inicializacionFuzzy();
		// printU();
	}

	@Override
	public double classifyInstance(Instance instancia) throws Exception {
		double[] u = calcularu(instancia);
		return (double) Utils.maxIndex(u);
	}

	@Override
	public double[] distributionForInstance(Instance instancia) throws Exception {
		return calcularu(instancia);
	}

	/**
	 * Inicializa la matriz de pertenencia de modo que cada U[i][j] es 1 si la
	 * instancia j predice la clase.
	 */
	private void inicializacionCrisp() {
		// U[i][j] es 1 si la instancia j predice la clase
		for (int j = 0; j < dataset.numInstances(); j++) {
			U[(int) dataset.instance(j).classValue()][j] = 1.0;
		}
	}

	/**
	 * Inicializa la matriz de pertenencia con asignaciones difusas teniendo en
	 * cuenta la clase predicha por los k vecinos.
	 */
	private void inicializacionFuzzy() throws Exception {

		for (int j = 0; j < dataset.numInstances(); j++) {
			// Seleccionamos los kini vecinos de la instancia j
			Instances kNN = S.kNearestNeighbours(dataset.instance(j), kini);

			// Recorre los vecinos y calcular cuantos vecinos
			// pertenecen a cada clase
			int count[] = new int[dataset.numClasses()];
			for (int i = 0; i < kini; i++) {
				// RELLENAR: Incrementar count teniendo en cuenta el classValue del vecino i
				count[(int) kNN.instance(i).classValue()]++;
			}

			// Recorre los kini vecinos para dar el valor de la inicializacion de U[_][j]
			int clase = (int) dataset.instance(j).classValue(); 
			for (int i = 0; i < dataset.numClasses(); i++) {
				double valor = (count[i] / (kini * 1.0)) * 0.49;
				if (i == clase)
					// RELLENAR: Actualiar U[i][j]
					U[i][j] = 0.51 + valor;
				else
					// RELLENAR: Actualiar U[i][j]
					U[i][j] = valor;
			}
		}
	}

	public double[] calcularu(Instance instancia) throws Exception {

		int indices[] = S.kNearestNeighboursIndices(instancia, k);
		double distancias[] = S.getDistances();

		// Nos quedaremos con las k instancias mas cercanas
		// y calculamos el vector u
		double u[] = new double[dataset.numClasses()];
		for (int i = 0; i < dataset.numClasses(); i++) {
			double suma_num = 0.0;
			double suma_den = 0.0;
			// RELLENAR: Recorrer los k vecinos mas cercanos y calcular la distancia			
			for (int j = 0; j < k; j++) {
				//CASO ESPECIAL: Si distancias[j]==0 => 0^(2/(m-1))=0
				//if (distancias[j] >= e) 
				{
					// RELLENAR: Actualizar suma_num y suma_den
					double aux = 1.0 / Math.pow(distancias[j], 2.0 / (m - 1));
					suma_num += U[i][indices[j]] * aux;
					suma_den += aux;
				}
			}
			// RELLENAR: Actualizar u[i]
			u[i] = suma_num / suma_den;
		}
		return (u);
	}

	/** Imprimie la matriz de pertenencias U */
	public void printU() {
		System.out.println(
				"Printing U\t numClasses:" + dataset.numClasses() + "\tnumInstances:" + dataset.numInstances());
		for (int i = 0; i < dataset.numClasses(); i++) {
			for (int j = 0; j < dataset.numInstances(); j++)
				System.out.print(U[i][j] + " ");
			System.out.println();
		}
	}

	/** Imprime el vector de pertenencias u */
	public void printu(double u[]) {
		System.out.println("Printing u\t numClasses:" + dataset.numClasses());
		for (int i = 0; i < dataset.numClasses(); i++) {
			System.out.println(u[i] + " ");
		}
	}

}
