import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.filters.Filter;

public class EjecutarFuzzyCMeans {

  public static void main(String[] args) throws Exception {

    ClusterEvaluation eval;
    SimpleKMeans kmeans;
    int c = 3;

    // DataSource source = new DataSource("data/diabetes.arff");
    DataSource source = new DataSource("/home/gtc/Desktop/GitHub/FuzzyWeka/data/ionosphere.arff");
    Instances data = source.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);

    // Quitar la informacion de clase para entrenar el modelo
    weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
    filter.setAttributeIndices("" + (data.classIndex() + 1));
    filter.setInputFormat(data);
    Instances dataClusterer = Filter.useFilter(data, filter);

    // Entrenar el modelo
    kmeans = new SimpleKMeans();
    kmeans.setSeed(10);
    // This is the important parameter to set
    kmeans.setPreserveInstancesOrder(true);
    kmeans.setNumClusters(c);
    kmeans.buildClusterer(dataClusterer);
    eval = new ClusterEvaluation();
    eval.setClusterer(kmeans);
    eval.evaluateClusterer(new Instances(data));
    System.out.println("RESULTADOS DE CLUSTERING CON KMEANS");
    System.out.println("# of clusters: " + eval.getNumClusters());
    System.out.println(eval.clusterResultsToString());

    // Repetimos lo mismo para FCM
    double m = 1.75;
    double e = 0.001;
    FuzzyCMeans fcm = new FuzzyCMeans(m, c, e);
    fcm.setSeed(10);
    fcm.buildClusterer(dataClusterer);
    eval = new ClusterEvaluation();
    eval.setClusterer(fcm);
    eval.evaluateClusterer(new Instances(data));
    System.out.println("RESULTADOS DE CLUSTERING CON FUZZY C MEANS");
    System.out.println("# of clusters: " + eval.getNumClusters());
    System.out.println(eval.clusterResultsToString());
  }
}
