import weka.clusterers.RandomizableClusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 实现贝兹代克的模糊C均值算法（Fuzzy C-Means, FCM）。该算法基于数据点的隶属度分配聚类，
 * 属于软聚类方法，与传统的K-means算法不同。
 * 
 * 贝兹代克，J.C., Ehrlich, R., & Full, W. (1984). FCM: The fuzzy c-means clustering
 * algorithm.
 * Computers & Geosciences, 10(2 - 3), 191 - 203.
 * 
 * @author Eva Gibaja
 */
public class FuzzyCMeans extends RandomizableClusterer implements weka.clusterers.Clusterer {

  private static final long serialVersionUID = 3391001128125832810L;

  /** 模糊指数（决定隶属度的模糊程度），通常设为 2.0 */
  protected double m;
  /** 聚类数量，需根据数据特性选择 */
  protected int c;
  /** 收敛阈值，当隶属度矩阵变化小于该值时终止迭代 */
  protected double epsilon;
  /** 最大迭代次数，防止算法长时间无法收敛 */
  protected int maxIteraciones = 200;

  // ================== 数据结构 ==================
  /** 聚类中心矩阵 [c][nDimensiones]，存储每个簇中心点的各维度值 */
  protected double V[][];
  /** 隶属度矩阵 [c][nInstancias]，U[i][j]表示实例j属于簇i的隶属度 */
  protected double U[][];
  /** 实例数量 */
  protected int nInstancias;
  /** 属性维度数量 */
  protected int nDimensiones;
  /** 数据集实例 */
  protected Instances dataset;

  /** 性能监控：记录各方法累计耗时（单位：纳秒） */
  private Map<String, Long> totalTimeMap = new HashMap<>();
  private Map<String, Integer> callCountMap = new HashMap<>();

  /**
   * 构造函数
   * 
   * @param m       模糊指数，建议值为 2.0
   * @param c       聚类数量，需要根据数据的特性来选择
   * @param epsilon 收敛阈值，通常设为 0.001
   */
  public FuzzyCMeans(double m, int c, double epsilon) {
    this.m = m;
    this.c = c;
    this.epsilon = epsilon;
  }

  /**
   * 核心聚类方法（Weka框架入口）
   * 该方法通过以下步骤来进行聚类：
   * 1. 初始化阶段：随机选择初始聚类中心
   * 2. 迭代优化阶段：交替更新聚类中心和隶属度矩阵
   * 3. 终止条件：达到最大迭代次数或隶属度变化小于指定阈值
   * 
   * @param data 输入数据集（包含所有实例特征）
   */
  @Override
  public void buildClusterer(Instances data) throws Exception {
    // [初始化阶段]
    long startTime = System.nanoTime();
    this.nInstancias = data.numInstances(); // 获取实例数量
    this.nDimensiones = data.numAttributes(); // 获取属性数量
    this.dataset = data;
    double error;

    V = new double[c][nDimensiones]; // 初始化聚类中心矩阵
    U = new double[c][nInstancias]; // 初始化隶属度矩阵

    inicializarV(); // 随机初始化聚类中心
    actualizarU(); // 计算初始隶属度
    // 利用函数imprimirMatriz分别打印V、U矩阵
    // 打印聚类中心矩阵 V
    // System.out.println("聚类中心矩阵 V:");
    // imprimirMatriz(V, c, nDimensiones);

    // // 打印隶属度矩阵 U
    // System.out.println("隶属度矩阵 U:");
    // imprimirMatriz(U, c, nInstancias);

    // [迭代优化阶段]
    int nIteraciones = 1;
    do {
      // 根据当前隶属度更新聚类中心
      calcularV();

      // 备份当前隶属度矩阵，用于计算迭代过程中的变化
      double aux[][] = copia(U);

      // 根据新的聚类中心更新隶属度
      actualizarU();

      // 计算两次迭代的隶属度差异
      // error值为0？？？？？？为什么？？？？
      error = NormaU(U, aux);

      // 新增误差输出
      System.out.printf("迭代 %d | 当前误差: %.6f\n", nIteraciones, error);
      // System.out.printf("迭代 %d | V[0][0]=%.4f | U[0][0]=%.4f | 误差: %.6f\n", 
      //     nIteraciones, V[0][0], U[0][0], error);
      
      nIteraciones++;
    } while (nIteraciones <= maxIteraciones && error > epsilon); // 收敛条件
    long endTime = System.nanoTime();

    // [性能分析阶段]
    recordTime("buildClusterer", endTime - startTime);

    // 输出每个函数的平均运行时间
    for (Map.Entry<String, Long> entry : totalTimeMap.entrySet()) {
      String functionName = entry.getKey();
      long totalTime = entry.getValue();
      int callCount = callCountMap.get(functionName);
      double averageTime = (double) totalTime / (callCount * 1_000_000); // 纳秒转为毫秒
      // 修改后的输出逻辑
      if (functionName == "buildClusterer") {
        double averageIN = averageTime / (nIteraciones-1);
        System.out.printf("运行总耗时：%.4fms，迭代%d次，平均迭代耗时：%.4fms\n", averageTime, nIteraciones-1, averageIN);
      } else {
        System.out.printf("%-20s | 调用次数: %-6d | 平均耗时: %.4f ms\n",
            functionName, callCount, averageTime);
      }
    }
  }

  /**
   * 对一个实例进行聚类并返回其所属的簇
   * 
   * @param instance 待聚类的实例
   * @return 该实例所属的簇的索引
   */
  @Override
  public int clusterInstance(Instance instance) throws Exception {
    long startTime = System.nanoTime();
    double u[] = evaluarInstancia(instance); // 获取该实例的隶属度

    // 返回隶属度最大的簇索引
    int result = Utils.maxIndex(u);
    long endTime = System.nanoTime();
    recordTime("clusterInstance", endTime - startTime);
    return result;
  }

  /**
   * 返回给定实例的各簇的隶属度分布
   * 
   * @param instance 待评估的实例
   * @return 隶属度分布
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {
    long startTime = System.nanoTime();
    double[] result = evaluarInstancia(instance); // 计算隶属度分布
    long endTime = System.nanoTime();
    recordTime("distributionForInstance", endTime - startTime);
    return result;
  }

  /**
   * 返回聚类的数量
   * 
   * @return 聚类数
   */
  @Override
  public int numberOfClusters() throws Exception {
    long startTime = System.nanoTime();
    int result = c;
    long endTime = System.nanoTime();
    recordTime("numberOfClusters", endTime - startTime);
    return result;
  }

  /**
   * 计算新的聚类中心V[i][d]，即每个簇在每个维度上的中心值
   */
  protected void calcularV() {
    long startTime = System.nanoTime();
    int d, j, i;
    double numerador, denominador;

    // 遍历每个簇
    for (i = 0; i < c; i++) {
      // 遍历每个属性
      for (d = 0; d < nDimensiones; d++) {
        numerador = 0.0;
        denominador = 0.0;
        // 遍历每个实例
        for (j = 0; j < nInstancias; j++) {
          if (!dataset.instance(j).isMissing(d)) { // 如果属性值不是缺失值
            numerador += Math.pow(U[i][j], m) * dataset.instance(j).value(d); // 更新分子
            denominador += Math.pow(U[i][j], m); // 更新分母
          }
        }
        V[i][d] = numerador / denominador; // 计算新的聚类中心
      }
    }
    long endTime = System.nanoTime();
    recordTime("calcularV", endTime - startTime);
  }

  /**
   * 计算实例到指定簇的距离
   * 
   * @param clusterIndex 簇的索引
   * @param instancia    待计算的实例
   * @return 实例到指定簇的距离
   */
  protected double distancia(int clusterIndex, Instance instancia) {
    long startTime = System.nanoTime();
    double suma = 0;
    for (int i = 0; i < nDimensiones; i++) {
      if (!instancia.isMissing(i)) {
        suma += Math.pow(instancia.value(i) - V[clusterIndex][i], 2.0); // 计算欧氏距离
      }
    }
    double result = Math.sqrt(suma); // 距离 = sqrt(总和)
    long endTime = System.nanoTime();
    recordTime("distancia", endTime - startTime);
    return result;
  }

  /**
   * 更新隶属度矩阵U
   */
  protected void actualizarU() {
    long startTime = System.nanoTime();
    for (int j = 0; j < nInstancias; j++) {
      double u[] = evaluarInstancia(dataset.instance(j)); // 计算每个实例的隶属度
      for (int i = 0; i < u.length; i++)
        U[i][j] = u[i]; // 更新隶属度矩阵
    }
    long endTime = System.nanoTime();
    recordTime("actualizarU", endTime - startTime);
  }

  /**
   * 计算实例的隶属度分布
   * 
   * @param instancia 待评估的实例
   * @return 隶属度分布
   */
  protected double[] evaluarInstancia(Instance instancia) {
      long startTime = System.nanoTime();
      double[] u = new double[c];
      
      // 预计算所有距离并添加极小值保护
      double[] d = new double[c];
      for(int k=0; k<c; k++){
          d[k] = Math.max(distancia(k, instancia), 1e-10);
      }
  
      for (int i = 0; i < c; i++) {
          double suma = 0;
          for (int j = 0; j < c; j++) {
              // 使用预计算的距离值，确保分母不为零
              suma += Math.pow(d[i] / d[j], 2.0 / (m - 1.0));
          }
          u[i] = 1.0 / suma;
      }
      
      long endTime = System.nanoTime();
      recordTime("evaluarInstancia", endTime - startTime);
      return u;
  }

  /**
   * 随机初始化聚类中心
   */
  protected void inicializarV() {
    // 初始化计时器，用于性能监控
    long startTime = System.nanoTime();
    // 使用Weka基类的随机种子初始化随机数生成器
    Random rand = new Random(getSeed());

    // 遍历每个聚类中心（共c个）
    for (int i = 0; i < c; i++) {
      // 随机选择一个数据实例作为初始聚类中心
      int index = rand.nextInt(nInstancias);
      // 遍历所有数据维度（属性）
      for (int j = 0; j < nDimensiones; j++) {
        // 跳过缺失值（Weka数据集特性处理）
        if (!dataset.instance(index).isMissing(j))
          // 将选定实例的属性值赋给聚类中心
          V[i][j] = dataset.instance(index).value(j);
      }
    }
    // 记录方法执行时间
    long endTime = System.nanoTime();
    recordTime("inicializarV", endTime - startTime);
  }

  /**
   * 计算两次隶属度矩阵的差异
   * 
   * @param U     当前隶属度矩阵
   * @param U_t_1 上一轮隶属度矩阵
   * @return 最大差异
   */
  protected double NormaU(double U[][], double U_t_1[][]) {
    long startTime = System.nanoTime();
    double maxDiferencia = Math.abs(U[0][0] - U_t_1[0][0]);

    for (int i = 0; i < U.length; i++) {
      for (int j = 0; j < U[0].length; j++) {
        if (Math.abs(U[i][j] - U_t_1[i][j]) > maxDiferencia)
          maxDiferencia = Math.abs(U[i][j] - U_t_1[i][j]); // 计算最大差异
      }
    }
    long endTime = System.nanoTime();
    recordTime("NormaU", endTime - startTime);
    return maxDiferencia;
  }

  /**
   * 复制一个矩阵
   * 
   * @param U 原矩阵
   * @return 复制的矩阵
   */
  protected double[][] copia(double U[][]) {
    long startTime = System.nanoTime();
    double aux[][] = new double[U.length][U[0].length];

    for (int i = 0; i < c; i++)
      for (int j = 0; j < nInstancias; j++)
        aux[i][j] = U[i][j]; // 复制矩阵
    long endTime = System.nanoTime();
    recordTime("copia", endTime - startTime);
    return aux;
  }

  /**
   * 打印矩阵
   * 
   * @param M    矩阵
   * @param nfil 矩阵的行数
   * @param ncol 矩阵的列数
   */
  protected void imprimirMatriz(double M[][], int nfil, int ncol) {
    long startTime = System.nanoTime();
    for (int i = 0; i < nfil; i++) {
      for (int j = 0; j < ncol; j++) {
        System.out.print(M[i][j] + "  "); // 打印矩阵元素
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
