import entity.Perceptron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

/*
Ejemplo donde nuestra red tiene que determinar cuando la salida es 0 o 1 (CLASIFICACION).
*/
public class Main {
    public static void main(String[] args) {
        ArrayList<double[]> entradas = new ArrayList<double[]>();
        ArrayList<double[]> salidas = new ArrayList<double[]>();

        for(int i = 0; i < 4; i++){
            entradas.add(new double[2]);
            salidas.add(new double[1]);
        }

        entradas.get(0)[0] = 0; entradas.get(0)[1] = 0; salidas.get(0)[0] = 1;
        entradas.get(1)[0] = 0; entradas.get(1)[1] = 1; salidas.get(1)[0] = 0;
        entradas.get(2)[0] = 1; entradas.get(2)[1] = 0; salidas.get(2)[0] = 0;
        entradas.get(3)[0] = 1; entradas.get(3)[1] = 1; salidas.get(3)[0] = 0;

        Perceptron p = new Perceptron(new int[]{entradas.get(0).length, 3, salidas.get(0).length});

        p.train(entradas, salidas, 0.5, 0.01);

        Random random = new Random();

        System.out.println("-------------------------------------");
        System.out.println("Test pijudo");

        for (int i = 0; i < 10; i++) {
            double a = random.nextDouble();
            double b = random.nextDouble();

            double[] outputs = p.activation(new double[]{a, b});
            System.out.println(Arrays.toString(outputs));
        }
    }
}