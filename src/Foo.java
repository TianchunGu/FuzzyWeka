import weka.classifiers.*;
import weka.core.*;

/**
Clasificador que alcula del conjunto de entrenamiento la media
de los valores de todos sus datos, y compara con la media
de la instancia a clasificar. Si la media de la instancia es 
menor que la media del conjutno de entrenamiento clasifica con la
clase cuyo índice valor sea 0.0 si no clasifica con la clase
cuyo índice valor es 1.0.
*/

public class Foo extends Classifier {
    
	private static final long serialVersionUID = -273566323458498981L;
	private double media_entrenamiento = 0; //Media del conjunto de entrenamiento

    public void buildClassifier(Instances instancias) throws Exception {
    	// Recorremos todas las instancias
        for (int i=0; i<instancias.numInstances(); i++) {
           // Recorremos todos sus atributos
           for (int j=0; j < instancias.numAttributes(); j++) 
           {  //Si se trata del atributo de clase no se contabiliza
        	   if(j!=instancias.classIndex())
        	      media_entrenamiento += instancias.instance(i).value(j);
           }
        }
        media_entrenamiento=media_entrenamiento/(instancias.numInstances()*instancias.numAttributes());
    } // buildClassifier

    public double classifyInstance(Instance instancia){
        float media_instancia = 0;

         // Recorremos todos sus atributos
         // El atributo de clase no se contabilizará
         for (int j=0; j < instancia.numAttributes(); j++)
         {
        	 if(j!=instancia.classIndex()){
                      media_instancia += instancia.value(j);                     
        	 }
         }
         media_instancia = media_instancia/instancia.numAttributes();
         if (media_instancia < media_entrenamiento)
           {return 0.0;}
         else 
           {return 1.0;}
              
   } // classifyInstance
   
 } // Fin clase

