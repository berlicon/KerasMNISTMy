using Keras;
using Keras.Layers;
using Keras.Models;
using Numpy;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Keras.Datasets;
using Keras.Utils;
using Keras.Optimizers;

namespace KerasMNISTMy
{
    class Program
    {
        //Скачать файлы для работы нейросети можно тут: https://www.kaggle.com/oddrationale/mnist-in-csv
        const int INPUT_LAYER_SIZE = 784;       //each image 28*28 pixels = 784 px
        const int ASSOCIATIONS_LAYER_SIZE = 20; //число нейронов на среднем слое, можно любое число, но лучше > 10
        const int RESULT_LAYER_SIZE = 10;       //analyse 10 images - numbers 0..9

        const int TRAIN_ROWS_COUNT = 1000;      //first rows to train;
        const int TEST_ROWS_COUNT = 100;       //other rows to test

        //Прошлая реализация (самописный алгоритм) RealNeuralNetworkMNIST
        //39% 1.900+100
        //80% 5.000+5.000
        //97% 60.000+100
        //91% 60.000+10.000

        //Текущая реализация:
        //batchSize/epochs: 1/1; 128:10; 1/3
        //1000+100 - 14% --- 43%
        //1900+100 - 11% 26% 61%
        //5000+100 - 17% 17% 90%
        //10000+100 - 14% -- 92%
        //5000+5000 - 12% -- 83%
        //5000+1000 - -- -- 85%
        //60000+100 - 15% -- 99%
        //60000+10000 - -- -- 96.02%

        const string FILE_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_train.csv";
        const string TEST_PATH = @"C:\Users\3208080\Downloads\mnist-in-csv\mnist_test.csv";

        private static int batchSize = 1;
        private static int epochs = 3;

        private static float learningRate = 0.5f;
        private static long correctResults = 0;
        private static Sequential model;
        private static bool trainModel = true;  //false если загружаем сохраненную обученную модель (экономия времени)

        static void Main(string[] args)
        {
            if (trainModel)
            {
                createModel();
                train();
            }
            test();

            Console.WriteLine("Правильно распознано {0}% вариантов. {1} из {2}",
                (float)100 * correctResults / TEST_ROWS_COUNT, correctResults, TEST_ROWS_COUNT);
        }

        private static void createModel()
        {
            model = new Sequential();
            model.Add(new Dense(INPUT_LAYER_SIZE, activation: "sigmoid"/*, input_dim: 1*/));//relu - better
            model.Add(new Dense(ASSOCIATIONS_LAYER_SIZE /* *5 better */, activation: "sigmoid"));// relu - better
            model.Add(new Dense(RESULT_LAYER_SIZE, activation: "sigmoid"));
            model.Compile(loss: "mean_squared_error"/*binary_crossentropy - better*/, optimizer: new SGD(lr: learningRate), metrics: new string[] { "accuracy" });
        }

        private static void train()
        {
            try
            {
                Console.WriteLine("Начало тренировки нейросети");
                var rows = File.ReadAllLines(FILE_PATH).Skip(1).Take(TRAIN_ROWS_COUNT).ToList();

                Console.WriteLine("Заполняем датасет данными");
                float[,] inputArray = new float[rows.Count, INPUT_LAYER_SIZE];
                float[,] outputArray = new float[rows.Count, RESULT_LAYER_SIZE];
                for (int i = 0; i < rows.Count; i++)
                {
                    Console.WriteLine("Итерация {0} из {1}", i + 1, TRAIN_ROWS_COUNT);
                    var values = rows[i].Split(',');
                    var correctNumber = byte.Parse(values[0]);

                    byte[] inputValues = values.Skip(1).Select(x => byte.Parse(x)).ToArray();
                    for (int j = 0; j < inputValues.Length; j++) { inputArray[i, j] = inputValues[j]; }

                    outputArray[i, correctNumber] = 1;
                }

                var input = new NDarray(inputArray);
                input = input.astype(np.float32);
                input /= 255;
                var output = new NDarray(outputArray);

                Console.WriteLine("Запускаем обучение");
                model.Fit(input, output, batch_size: batchSize, epochs: epochs, verbose: 2);

                Console.WriteLine("Сохраняем модель");
                File.WriteAllText("model.json", model.ToJson());
                model.SaveWeight("model.h5");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                throw;
            }
        }

        private static void test()
        {
            Console.WriteLine("Начало тестирования нейросети");
            var index = 1;
            var rows = File.ReadAllLines(TEST_PATH).Skip(1).Take(TEST_ROWS_COUNT).ToList();

            BaseModel loadedModel;
            if (model == null)
            {
                //model = (Sequential)BaseModel.ModelFromJson(File.ReadAllText("model.json"));  //не может преобразовать BaseModel в Sequential
                loadedModel = Sequential.ModelFromJson(File.ReadAllText("model.json"));
                loadedModel.LoadWeight("model.h5");
                loadedModel.Compile(loss: "mean_squared_error", optimizer: new SGD(lr: learningRate), metrics: new string[] { "accuracy" });
            }
            else
            {
                loadedModel = model;
            }

            foreach (var row in rows)
            {
                Console.WriteLine("Итерация {0} из {1}", index++, TEST_ROWS_COUNT);
                var values = row.Split(',');
                var correctNumber = byte.Parse(values[0]);

                float[,] inputArray = new float[1, INPUT_LAYER_SIZE];
                byte[] inputValues = values.Skip(1).Select(x => byte.Parse(x)).ToArray();
                for (int i = 0; i < inputValues.Length; i++) { inputArray[0, i] = inputValues[i]; }
                var input = new NDarray(inputArray);
                input = input.astype(np.float32);
                input /= 255;

                float[,] outputArray = new float[1, RESULT_LAYER_SIZE];
                outputArray[0, correctNumber] = 1;
                var output = new NDarray(outputArray);

                var score = loadedModel.Evaluate(input, output, verbose: 0);
                Console.WriteLine($"Test loss: {score[0]}");
                Console.WriteLine($"Test accuracy: {score[1]}");

                var outputActual = loadedModel.Predict(input, verbose: 0);
                var x = outputActual.argmax();

                calculateStatistics(correctNumber, int.Parse(x.str));
            }
        }

        private static void calculateStatistics(int correctNumber, int proposalNumber)
        {
            Console.WriteLine("Число {0} определено как {1} {2}", correctNumber, proposalNumber,
                proposalNumber == correctNumber ? "УСПЕХ" : "НЕУДАЧА");
            if (proposalNumber == correctNumber) correctResults++;
        }
    }
}
