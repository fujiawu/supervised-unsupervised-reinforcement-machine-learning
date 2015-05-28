package opt.example;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;


/**
 * A evaluation function for k-color with a graph
 * @author Fujia Wu fujiawu@gatech.edu
 */
public class KColorEvaluationFunction implements EvaluationFunction {
    
    /**
     * adjacent matrix of the graph
     */
    private int[][] adj;
   
    /**
     * Make a new k-color evaluation function
     */
    public KColorEvaluationFunction(int[][] a) {
       adj = a;
    }

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        Vector data = d.getData();
        double value = 0;
        for(int i = 0; i < adj.length; i++) {
            if (adj[i] == null) continue;
            for(int j = 0; j < adj[i].length; j++) {
                 if (data.get(i) == data.get(adj[i][j]))
                    value++;
            }
        }
        value = 1000/(value+1);  
        return value;
    }
}
