import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.io.IOException;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.NumericToNominal; // 新增过滤器
import weka.core.Attribute; // 新增Attribute类导入

public class TestFuzzyCMeans {
  public static void main(String[] args) {
    try {
      // 加载样本数据
      CSVLoader loader = new CSVLoader();
      loader.setSource(new File("data/S1.csv"));
      Instances data = loader.getDataSet();

      // 加载标签数据
      CSVLoader labelLoader = new CSVLoader();
      labelLoader.setSource(new File("data/s1-label.csv")); 
      Instances labels = labelLoader.getDataSet();

      // 加载标签数据后添加类型转换
      NumericToNominal convertFilter = new NumericToNominal();
      convertFilter.setOptions(new String[]{"-R","1"});
      convertFilter.setInputFormat(labels);
      labels = Filter.useFilter(labels, convertFilter);
      
      // 删除无效的setName和typeToString调用（原32-33行）
      // 验证标签类型
      System.out.println("转换后标签类型: " + (labels.attribute(0).isNominal() ? "NOMINAL" : "NUMERIC"));

      // 将标签添加为类别属性
      data.insertAttributeAt(labels.attribute(0), data.numAttributes());
      data.setClassIndex(data.numAttributes() - 1);

      // 移除类别属性用于聚类
      Remove removeFilter = new Remove();
      removeFilter.setAttributeIndices("" + (data.classIndex() + 1));
      removeFilter.setInputFormat(data);
      Instances dataCluster = Filter.useFilter(data, removeFilter);

      // 将标准化过滤器移动到这里（原错误位置）
      weka.filters.unsupervised.attribute.Standardize stdFilter = new weka.filters.unsupervised.attribute.Standardize();
      stdFilter.setInputFormat(dataCluster);
      dataCluster = Filter.useFilter(dataCluster, stdFilter);

      // 删除重复的stdFilter声明（原56-58行）
      stdFilter.setInputFormat(dataCluster);
      dataCluster = Filter.useFilter(dataCluster, stdFilter);
      
      // 创建并构建聚类模型（参数保持原样）
      FuzzyCMeans fcm = new FuzzyCMeans(2.0, 15, 0.001);
      fcm.buildClusterer(dataCluster);

      // 使用完整数据（含标签）进行评估
      ClusterEvaluation eval = new ClusterEvaluation();
      eval.setClusterer(fcm);
      eval.evaluateClusterer(data); // 自动关联真实标签

      System.out.println(eval.clusterResultsToString());
    } catch (Exception e) { 
      System.err.println("聚类过程中出错: " + e.getMessage());
    }
  }
}
