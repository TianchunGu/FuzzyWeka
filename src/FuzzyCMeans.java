import weka.clusterers.RandomizableClusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Clase que implementa el algoritmo Fuzzy C-Means de Bezdek. Bezdek, J. C.,
 * Ehrlich, R., & Full, W. (1984). FCM: The fuzzy c-means clustering algorithm.
 * Computers & Geosciences, 10(2 - 3), 191 - 203.
 * 
 * @author Eva Gibaja
 */
public class FuzzyCMeans extends RandomizableClusterer implements weka.clusterers.Clusterer {

  private static final long serialVersionUID = 3391001128125832810L;
  /** Exponente fuzzy */
  protected double m;
  /** Number of clusters */
  protected int c;
  /** Tolerancia */
  protected double epsilon;
  /** Numero maximo de iteraciones */
  protected int maxIteraciones = 100;
  /** Vector de prototipos. Tamaño numeroClusters x atributos */
  protected double V[][];
  /** Matriz de pertenencia. Tamaño numeroClusters x nInstancias */
  protected double U[][];
  /** Numero de instancias */
  protected int nInstancias;
  /** Numero de atributos */
  protected int nDimensiones;
  /** Dataset */
  protected Instances dataset;

  // 用于记录每个函数的总运行时间和调用次数
  private Map<String, Long> totalTimeMap = new HashMap<>();
  private Map<String, Integer> callCountMap = new HashMap<>();

  /**
   * Constructor
   * 
   * @param m
   *                Exponente fuzzy. Indica como de difusas son las particiones.
   *                Cuanto más cercano es este valor a 1, mas crisp son las
   *                particiones y cuanto mayor es este numero, las particiones son
   *                mas difusas. Son comunes valores entre [1,5 - 2.5], comunmente
   *                se utiliza m = 2.
   * @param c
   *                Numero de clusters.
   * @param epsilon
   *                Tolerancia. Condicion de parada que consiste en la diferencia
   *                entre las matrices de particion en dos iteraciones.
   */
  public FuzzyCMeans(double m, int c, double epsilon) {
    this.m = m;
    this.c = c;
    this.epsilon = epsilon;
  }

  @Override
  public void buildClusterer(Instances data) throws Exception {
    long startTime = System.nanoTime();
    this.nInstancias = data.numInstances();
    this.nDimensiones = data.numAttributes();
    this.dataset = data;
    double error;

    V = new double[c][nDimensiones]; // Construir vector de prototipos
    U = new double[c][nInstancias]; // Construir matriz de pertenencia

    inicializarV();
    actualizarU();

    int nIteraciones = 1;
    do {
      // Calcular los centros
      calcularV();

      // Guardar copia de U para poder calcular la norma
      double aux[][] = copia(U);

      // Actualizar U
      actualizarU();

      // CalcularNormaU
      error = NormaU(U, aux);

      nIteraciones++;
    } while (nIteraciones <= maxIteraciones && error > epsilon);
    long endTime = System.nanoTime();
    recordTime("buildClusterer", endTime - startTime);

    // 输出每个函数的平均运行时间
    for (Map.Entry<String, Long> entry : totalTimeMap.entrySet()) {
      String functionName = entry.getKey();
      long totalTime = entry.getValue();
      int callCount = callCountMap.get(functionName);
      double averageTime = (double) totalTime / (callCount * 1_000_000);
      System.out.println(functionName + " average time: " + averageTime + " ms");
    }
  }

  @Override
  public int clusterInstance(Instance instance) throws Exception {
    long startTime = System.nanoTime();
    double u[] = evaluarInstancia(instance);

    // RELLENAR: calcular la posicion del mayor grado de pertenencia y
    // devolver esa posicion
    int result = Utils.maxIndex(u);
    long endTime = System.nanoTime();
    recordTime("clusterInstance", endTime - startTime);
    return result;
  }

  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {
    long startTime = System.nanoTime();
    double[] result = evaluarInstancia(instance);
    long endTime = System.nanoTime();
    recordTime("distributionForInstance", endTime - startTime);
    return result;
  }

  @Override
  public int numberOfClusters() throws Exception {
    long startTime = System.nanoTime();
    int result = c;
    long endTime = System.nanoTime();
    recordTime("numberOfClusters", endTime - startTime);
    return result;
  }

  /**
   * Calcular el vector de pertencia de cada centro. Este vector tendra
   * dimensiones nCentros x nAtributos
   */
  protected void calcularV() {
    long startTime = System.nanoTime();
    int d, j, i;
    double numerador, denominador;

    // Recorrer todos los centros
    for (i = 0; i < c; i++) {
      // Recorremos todas las dimensiones (atributos)
      for (d = 0; d < nDimensiones; d++) {
        numerador = 0.0;
        denominador = 0.0;
        // Recorres todas las instancias
        for (j = 0; j < nInstancias; j++) {
          // si el atributo no es missing
          if (!dataset.instance(j).isMissing(d)) {
            numerador += Math.pow(U[i][j], m) * dataset.instance(j).value(d);
            denominador += Math.pow(U[i][j], m);
          }
        }
        V[i][d] = numerador / denominador;
      }
    }
    long endTime = System.nanoTime();
    recordTime("calcularV", endTime - startTime);
  }

  /** Calcula la distancia de una instancia a un cluster determinado */
  protected double distancia(int clusterIndex, Instance instancia) {
    long startTime = System.nanoTime();
    double suma = 0;
    for (int i = 0; i < nDimensiones; i++) {
      if (!instancia.isMissing(i)) {
        // RELLENAR: Actualizar suma
        suma += Math.pow(instancia.value(i) - V[clusterIndex][i], 2.0);
      }
    }
    double result = Math.sqrt(suma);
    long endTime = System.nanoTime();
    recordTime("distancia", endTime - startTime);
    return result;
  }

  /** Actualiza la matriz de pertenencias. */
  protected void actualizarU() {
    long startTime = System.nanoTime();
    {
      for (int j = 0; j < nInstancias; j++) {
        double u[] = evaluarInstancia(dataset.instance(j));
        for (int i = 0; i < u.length; i++)
          U[i][j] = u[i];
      }
    }
    long endTime = System.nanoTime();
    recordTime("actualizarU", endTime - startTime);
  }

  /**
   * Evalua una instancia devolviendo el grado de pertenencia a cada cluster.
   * 
   * @param instancia
   *                  La instancia a evaluar
   */
  protected double[] evaluarInstancia(Instance instancia) {
    long startTime = System.nanoTime();
    double[] u = new double[c];

    // Recorre los clusters
    for (int i = 0; i < c; i++) {
      double suma = 0;
      for (int j = 0; j < c; j++)
        if (distancia(j, instancia) != 0)
          suma += Math.pow(distancia(i, instancia) / distancia(j, instancia), 2.0 / (m - 1));
      u[i] = 1 / suma;
    }
    long endTime = System.nanoTime();
    recordTime("evaluarInstancia", endTime - startTime);
    return u;
  }

  /** Inicializa el vector de prototipos con instancias aleatorias del dataset. */
  protected void inicializarV() {
    long startTime = System.nanoTime();
    Random rand = new Random(getSeed());

    for (int i = 0; i < c; i++) {
      // Selecciona un patron aleatorio
      int index = rand.nextInt(nInstancias);
      for (int j = 0; j < nDimensiones; j++) {
        if (!dataset.instance(index).isMissing(j))
          V[i][j] = dataset.instance(index).value(j);
        // else
        // V[i][j] = 0.0;
      }
    }
    long endTime = System.nanoTime();
    recordTime("inicializarV", endTime - startTime);
  }

  /**
   * Calcula la diferencia elemento a elemento de dos matrices de la misma
   * dimension y devuelve la maxima diferencia
   */
  protected double NormaU(double U[][], double U_t_1[][]) {
    long startTime = System.nanoTime();
    double maxDiferencia = Math.abs(U[0][0] - U_t_1[0][0]);

    for (int i = 0; i < U.length; i++) {
      for (int j = 0; j < U[0].length; j++) {
        if (Math.abs(U[i][j] - U_t_1[i][j]) > maxDiferencia)
          maxDiferencia = Math.abs(U[i][j] - U_t_1[i][j]);
      }
    }
    long endTime = System.nanoTime();
    recordTime("NormaU", endTime - startTime);
    return maxDiferencia;
  }

  /** Hace una copia fisica de la matriz */
  protected double[][] copia(double U[][]) {
    long startTime = System.nanoTime();
    double aux[][] = new double[U.length][U[0].length];

    for (int i = 0; i < c; i++)
      for (int j = 0; j < nInstancias; j++)
        aux[i][j] = U[i][j];
    long endTime = System.nanoTime();
    recordTime("copia", endTime - startTime);
    return aux;
  }

  /** Imprime una matriz en pantalla */
  protected void imprimirMatriz(double M[][], int nfil, int ncol) {
    long startTime = System.nanoTime();
    for (int i = 0; i < nfil; i++) {
      for (int j = 0; j < ncol; j++) {
        System.out.print(M[i][j] + "  ");
      }
      System.out.println();
    }
    long endTime = System.nanoTime();
    recordTime("imprimirMatriz", endTime - startTime);
  }

  // 记录函数的运行时间和调用次数
  private void recordTime(String functionName, long time) {
    totalTimeMap.put(functionName, totalTimeMap.getOrDefault(functionName, 0L) + time);
    callCountMap.put(functionName, callCountMap.getOrDefault(functionName, 0) + 1);
  }
}
