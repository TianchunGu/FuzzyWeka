
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;


public class EjecutarFoo {

	public static void main(String[] args) throws Exception {
		
		   Foo C;
		   
		    C= new Foo();
		   
		   //Cargamos el dataset en memoria. Indicamos cuál es la clase objetivo
		   DataSource source = new DataSource("data/iris.arff");
		   Instances instances = source.getDataSet();
		   instances.setClassIndex(instances.numAttributes() - 1);
		     
		   //Cross validation
		   Evaluation eval = new Evaluation(instances);
		   eval.crossValidateModel(C, instances, 5, new Random(1));
		   System.out.println("RESULTADOS CLASIFICADOR FOO");
		   System.out.println(eval.toSummaryString());
		   System.out.println(eval.toClassDetailsString());
		   System.out.println(eval.toMatrixString());
		   
		   
		   
	}

}
