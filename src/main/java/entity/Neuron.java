package entity;

import java.util.Random;

public class Neuron {
    private double threshold;
    private double[] weights;
    private double ponderedSum;

    Neuron(int entriesCount, Random random){
        threshold = random.nextDouble();
        weights = new double[entriesCount];

        for(int i = 0; i < weights.length; i++){
            weights[i] = random.nextDouble();
        }
    }

    public double activation(double[] entries){
        ponderedSum = threshold;

        for(int i = 0; i < entries.length; i++){
            ponderedSum += entries[i] * weights[i];
        }

        return utils.fuctions.sigmoid(ponderedSum);
    }

    public double[] getWeights() {
        return weights;
    }

    public double getPonderedSum() {
        return ponderedSum;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold){
        this.threshold = threshold;
    }
}
