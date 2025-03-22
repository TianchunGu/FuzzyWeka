import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.io.IOException;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.NumericToNominal; // 新增过滤器
import weka.core.Attribute; // 新增Attribute类导入

public class TestFuzzyCMeansOne {

  public static void main(String[] args) {
    try {
      // 加载样本数据（原有代码保持不变）
      CSVLoader loader = new CSVLoader();
      loader.setSource(new File("data/S1.csv"));
      Instances data = loader.getDataSet();
      // 输出读取的data样本数据
      for(int i=0;i<data.numInstances();i++){
        System.out.println("data[" + (i + 1) + "]: " + data.instance(i));
      }
      // 数据读取无误
      System.out.println("数据加载成功: " + data.numInstances() + " 实例, " + data.numAttributes() + " 属性");
      // 执行模糊C均值聚类函数
      double m = 2;
      double e = 0.0001;
      int c = 15;
      FuzzyCMeans fcm = new FuzzyCMeans(m, c, e);
      fcm.setSeed(10);
      fcm.buildClusterer(data);
    } catch (Exception e) {
      System.err.println("数据加载失败: " + e.getMessage());
      e.printStackTrace();
    }
  }
}
