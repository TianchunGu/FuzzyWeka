import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class EjecutarMasProbable {
	public static void main(String[] args) throws Exception {
		
		   MasProbable C;
		   
		   C= new MasProbable();
		   
		   //Cargamos el dataset en memoria. Indicamos cuál es la clase objetivo
		   DataSource source = new DataSource("data/breast-cancer.arff");
		   Instances instances = source.getDataSet();
		   instances.setClassIndex(instances.numAttributes() - 1);
		     
		   //Cross validation
		   Evaluation eval = new Evaluation(instances);
		   eval.crossValidateModel(C, instances, 5, new Random(1));
		   System.out.println("RESULTADOS CLASIFICADOR MAS PROBABLE");
		   
		   //Imprime resultados generales
		   System.out.println(eval.toSummaryString());
		   
		   //Imprime resultados detallados por clase
		   System.out.println(eval.toClassDetailsString());
		   
		   //Imprime la matriz de confusión
		   System.out.println(eval.toMatrixString());
}
}