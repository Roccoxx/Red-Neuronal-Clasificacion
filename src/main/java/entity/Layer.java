package entity;

import java.util.ArrayList;
import java.util.Random;

public class Layer {
    private ArrayList<Neuron> neuronList;
    private double[] outputs;

    public Layer(int entriesNum, int neuronsNum, Random random) {
        neuronList = new ArrayList<Neuron>();

        for (int i = 0; i < neuronsNum; i++) {
            neuronList.add(new Neuron(entriesNum, random));
        }
    }

    public double[] activation(double[] entries) {
        outputs = new double[neuronList.size()];

        for (int i = 0; i < neuronList.size(); i++) {
            outputs[i] = neuronList.get(i).activation(entries);
        }

        return outputs;
    }

    public ArrayList<Neuron> getNeuronList() {
        return neuronList;
    }

    public double[] getOutputs() {
        return outputs;
    }
}
