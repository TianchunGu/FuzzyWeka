import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import java.util.Random;

public class EjecutarKNN {

   public static void main(String[] args) throws Exception {
		
   FuzzyKNN F1= new FuzzyKNN(3, 1.5, 1);   
   FuzzyKNN F2= new FuzzyKNN(3, 1.05, 2);
   CrispKNN C= new CrispKNN(3);   
   FuzzyNP NP= new FuzzyNP(3);
   
   String filename = new String("data/breast-cancer.arff");
   //String filename = new String("data/ionosphere.arff");
   //String filename = new String("data/diabetes.arff");
   //String filename = new String("data/iris.arff");
   
   //Cargamos el dataset en memoria
   DataSource source = new DataSource(filename);
   Instances instances = source.getDataSet();
   
   //Indicamos cual es la clase objetivo
   instances.setClassIndex(instances.numAttributes() - 1);
   
   Evaluation eval;
   
   //Cross validation para CrispKNN
   eval = new Evaluation(instances);
   eval.crossValidateModel(C, instances, 10, new Random(1));
   System.out.println("RESULTADOS CRISP KNN CON DATASET "+filename);
   System.out.println(eval.toSummaryString());
   System.out.println(eval.toClassDetailsString());
   System.out.println(eval.toMatrixString());
   
   //Cross validation para FuzzyKNN 
   eval = new Evaluation(instances);
   eval.crossValidateModel(F1, instances, 10, new Random(1));
   System.out.println("RESULTADOS FUZZY KNN (inicializacion crisp) CON DATASET "+filename);
   System.out.println(eval.toSummaryString());
   System.out.println(eval.toClassDetailsString());
   System.out.println(eval.toMatrixString());
   
   //Cross validation para FuzzyKNN 
   eval = new Evaluation(instances);
   eval.crossValidateModel(F2, instances, 10, new Random(2));
   System.out.println("RESULTADOS FUZZY KNN (inicializacion fuzzy) CON DATASET "+filename);
   System.out.println(eval.toSummaryString());
   System.out.println(eval.toClassDetailsString());
   System.out.println(eval.toMatrixString());
   
  
   //Cross validation para Fuzzy Nearest Prototype
   //RELLENAR: Incluir el codigo para evaluar fuzzy nearest prototype
   eval = new Evaluation(instances);
   eval.crossValidateModel(NP, instances, 10, new Random(1));   
   System.out.println("RESULTADOS FUZZY NEAREST PROTOTYPE CON DATASET "+ filename);
   System.out.println(eval.toSummaryString());
   System.out.println(eval.toClassDetailsString());
   System.out.println(eval.toMatrixString());
   
  }

}

