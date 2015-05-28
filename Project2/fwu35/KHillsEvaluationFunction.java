package opt.example;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;


/**
 * A evaluation function for k-hills, that
 * is the sum of given k n-dimensional Gaussian functions
 * y = sum_k(h_k*Exp[-sum_n((x_n-mean(k,n))/(2*(std(k,n))^2))])
 * for k=1, n=1, y = h*Exp[-(x-mean)^2/(2*std^2)]
 * @author Fujia Wu fujiawu@gatech.edu
 */
public class KHillsEvaluationFunction implements EvaluationFunction {
    
    /**
     * The means for the k n-dim Gaussian functions
     */
    private double[][] mean;
    
    /**
     * The standard deviations for the k n-dim Gaussian functions
     */
    private double[][] std;
    
    /**
     * The heights for the k n-dim Gaussian functions
     */
    private double[] height;
    
    /**
     * Make a new k-hills evaluation function
     * @param m the k n-dim means m[k][n]
     * @param s the k n-dim standard deviations s[k][n]
     * @param h the k n-dim heights h[k]
     */
    public KHillsEvaluationFunction(double[][] m, double[][] s, double[] h) {
        mean = m;
        std = s;
        height = h;
    }

    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        Vector data = d.getData();
        double value = 0;
        for (int k = 0; k < height.length; k++) {   
          double arg = 0;
          for (int i = 0; i < data.size(); i++) {
             double temp = (data.get(i)-mean[k][i]) * (data.get(i)-mean[k][i]);
             temp = temp / (2*std[k][i]*std[k][i]);
             arg += temp;
          }
          value += height[k]*Math.exp(-arg);              
        }
        return value;
    }

}
