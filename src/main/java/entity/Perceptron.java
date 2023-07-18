package entity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Perceptron {
    private ArrayList<Layer> layerList;

    /*
    Permite ir calculando las derivadas parciales de forma dinámica capa por capa
    y así cuando se este calculando las derivadas de las neuronas de la capa i
    ya previamente se han calculado las derivadas de la capa i+1,
    esto es así porque el proceso de Back propagation se hace desde la capa de salida hasta la capa de entrada.
     */
    private ArrayList<double[]> sigmaList;
    private ArrayList<double[][]> deltaList; // almacenamos las derivadas parciales de la funcion de error respecto a los pesos

    public Perceptron(int[] neuronsPerLayer) {
        layerList = new ArrayList<Layer>();
        Random random = new Random();

        for (int i = 0; i < neuronsPerLayer.length; i++) {
            if (i == 0) {
                layerList.add(new Layer(neuronsPerLayer[i], neuronsPerLayer[i], random));
            } else {
                layerList.add(new Layer(neuronsPerLayer[i - 1], neuronsPerLayer[i], random));
            }
        }
    }

    public double[] activation(double[] entries){
        double[] outputs = new double[0];

        for (Layer layer : layerList) {
            outputs = layer.activation(entries);
            entries = outputs;
        }

        return outputs;
    }

    public double calculateError(double[] realOutput, double[] correctOutput){
        double err = 0;
        for(int i = 0; i < realOutput.length; i++){
            err += 0.5 * Math.pow(realOutput[i] - correctOutput[i], 2);
        }

        return err;
    }

    public double calculateTotalError(ArrayList<double[]> entries, ArrayList<double[]> correctOutput){
        double err = 0;
        for(int i = 0; i < entries.size(); i++){
            err += calculateError(activation(entries.get(i)), correctOutput.get(i));
        }

        return err;
    }

    public void initializeDeltas(){
        this.deltaList = new ArrayList<double[][]>();

        for (int i = 0; i < layerList.size(); i++) {
            int neuronsNum = this.layerList.get(i).getNeuronList().size();
            int weightsNum = this.layerList.get(i).getNeuronList().get(0).getWeights().length;

            // la matriz de deltas esta compuesta por una matriz por cada neurona de la capa
            // cada matriz tiene tantas filas como neuronas en la capa y tantas columnas como pesos tenga cada neurona
            // obviamente que cada neurona de la capa va a tener la misma cantidad de pesos por eso obtenemos el primer indice
            this.deltaList.add(new double[neuronsNum][weightsNum]);

            // a cada matriz de deltas le asignamos 0 en cada una de sus celdas
            for (int j = 0; j < neuronsNum; j++) {
                Arrays.fill(this.deltaList.get(i)[j], 0);
            }
        }
    }

    public void calculateSigmas(double[] correctOutput){
        this.sigmaList = new ArrayList<double[]>();

        for (Layer layer : layerList) {
            this.sigmaList.add(new double[layer.getNeuronList().size()]);
        }

        for (int i = this.layerList.size() - 1; i >= 0; i--) {
            for(int j = 0; j < this.layerList.get(i).getNeuronList().size(); j++) {
                // si es la ultima capa
                if (i == this.layerList.size() - 1) {
                    double[] outputs = this.layerList.get(i).getOutputs();
                    double y = outputs[j];
                    this.sigmaList.get(i)[j] = (y - correctOutput[j]) * utils.fuctions.sigmoidDerivative(y);
                }
                else {
                    double sum = 0;
                    for (int k = 0; k < this.layerList.get(i + 1).getNeuronList().size(); k++) {
                        double[] weights = this.layerList.get(i + 1).getNeuronList().get(k).getWeights();
                        sum += weights[j] * this.sigmaList.get(i + 1)[k];
                    }
                    this.sigmaList.get(i)[j] =  utils.fuctions.sigmoidDerivative(this.layerList.get(i).getNeuronList().get(j).getPonderedSum()) * sum;
                }
            }
        }
    }

    public void calculateDeltas() {
        for (int i = 1; i < this.layerList.size(); i++) {
            for (int j = 0; j < this.layerList.get(i).getNeuronList().size(); j++) {
                for (int k = 0; k < this.layerList.get(i).getNeuronList().get(j).getWeights().length; k++) {
                    double[] outputs = this.layerList.get(i - 1).getOutputs();
                    double sigma = this.sigmaList.get(i)[j];
                    this.deltaList.get(i)[j][k] += sigma * outputs[k];
                }
            }
        }
    }

    public void updateWeights(double learningRate) {
        for (int i = 0; i < this.layerList.size(); i++){
            for (int j = 0; j < this.layerList.get(i).getNeuronList().size(); j++) {
                for(int k = 0; k < this.layerList.get(i).getNeuronList().get(j).getWeights().length; k++){
                    this.layerList.get(i).getNeuronList().get(j).getWeights()[k] -= learningRate * this.deltaList.get(i)[j][k];
                }
            }
        }
    }

    public void updateThresholds(double learningRate) {
        for (int i = 0; i < this.layerList.size(); i++) {
            for (int j = 0; j < this.layerList.get(i).getNeuronList().size(); j++) {
                Neuron neuron = this.layerList.get(i).getNeuronList().get(j);
                double newThreshold = neuron.getThreshold() - learningRate * this.sigmaList.get(i)[j];
                neuron.setThreshold(newThreshold);
            }
        }
    }

    /*
    Para cada conjunto de entrada posible calculamos la salida de la red y propagamos el error hacia atrás, o lo que es lo mismo,
    calculamos las derivadas de todos los pesos y los umbrales con respecto al error.
    Luego actualizamos los pesos y los umbrales según la fórmula del Descenso del gradiente.
    Note que para actualizar los pesos hay que esperar haber calculado todas las derivadas asociadas a una entrada.
     */

    public void backPropagation(ArrayList<double[]> entries, ArrayList<double[]> correctOutput, double learningRate){
        this.initializeDeltas();
        for(int i = 0; i < entries.size(); i++){
            this.activation(entries.get(i));
            this.calculateSigmas(correctOutput.get(i));
            this.calculateDeltas();
            this.updateThresholds(learningRate);
        }

        this.updateWeights(learningRate);
    }

    public void train(ArrayList<double[]> trainEntries, ArrayList<double[]> trainOutputs, double learningRate, double maxError){
        double err = 99999999;

        while(err > maxError){
            this.backPropagation(trainEntries, trainOutputs, learningRate);
            err = this.calculateTotalError(trainEntries, trainOutputs);
            System.out.println(err);
        }
    }
}
