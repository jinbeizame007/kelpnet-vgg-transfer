using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using CaffemodelLoader;
using KelpNet.Loss;
using KelpNet.Common;
using KelpNet.Common.Loss;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;
using KelpNet.Optimizers;
using TestDataManager;

class VGGTransfer
{
    private const string DOWNLOAD_URL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel";
    private const string MODEL_FILE = "VGG_ILSVRC_16_layers.caffemodel";
    private const int TRAIN_DATA_LENGTH = 1000;
    private const int TEST_DATA_LENGTH = 100;
    private const int BATCH_SIZE = 50;
    private const int TRAIN_DATA_COUNT = 20;

    public static void Main()
    {

        // platformIdは、OpenCL・GPUの導入の記事に書いてある方法でご確認ください
        // https://jinbeizame.hateblo.jp/entry/kelpnet_opencl_gpu
        Weaver.Initialize(ComputeDeviceTypes.Gpu, platformId: 1, deviceIndex: 0);

        // ネットからVGGの学習済みモデルをダウンロード
        string modelFilePath = InternetFileDownloader.Donwload(DOWNLOAD_URL, MODEL_FILE);
        // 学習済みモデルをFunctionのリストとして保存
        List<Function> vgg16Net = CaffemodelDataLoader.ModelLoad(modelFilePath);

        // VGGの出力層とその活性化関数を削除
        vgg16Net.RemoveAt(vgg16Net.Count() - 1);
        vgg16Net.RemoveAt(vgg16Net.Count() - 1);

        // VGGの各FunctionのgpuEnableをtrueに
        for (int i = 0; i < vgg16Net.Count - 1; i++)
        {
            // GPUに対応している層であれば、GPU対応へ
            if (vgg16Net[i] is Convolution2D || vgg16Net[i] is Linear || vgg16Net[i] is MaxPooling)
            {
                ((IParallelizable)vgg16Net[i]).SetGpuEnable(true);
            }
        }

        // VGGをリストからFunctionStackに変換
        FunctionStack vgg = new FunctionStack(vgg16Net.ToArray());

        // 層を圧縮
        vgg.Compress();

        // 新しく出力層とその活性化関数を用意
        FunctionStack nn = new FunctionStack(
            new Linear(4096, 1, gpuEnable: true),
            new Sigmoid()
        );

        // 最適化手法としてAdamをセット
        nn.SetOptimizer(new Adam());

        Console.WriteLine("DataSet Loading...");

        // 訓練・テストデータ用のNdArrayを用意
        // データセットは以下のURLからダウンロードを行い、
        // VGGTransfer /bin/Debug/Data にtrainフォルダを置いてください。
        // https://www.kaggle.com/c/dogs-vs-cats/data
        NdArray[] trainData = new NdArray[TRAIN_DATA_LENGTH * 2];
        NdArray[] trainLabel = new NdArray[TRAIN_DATA_LENGTH * 2];
        NdArray[] testData = new NdArray[TEST_DATA_LENGTH * 2];
        NdArray[] testLabel = new NdArray[TEST_DATA_LENGTH * 2];

        for (int i = 0; i < TRAIN_DATA_LENGTH + TEST_DATA_LENGTH; i++)
        {
            // 犬・猫の画像読み込み
            Bitmap baseCatImage = new Bitmap("Data/train/cat." + i + ".jpg");
            Bitmap baseDogImage = new Bitmap("Data/train/dog." + i + ".jpg");
            // 変換後の画像を格納するBitmapを定義
            Bitmap catImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
            Bitmap dogImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
            // Graphicsオブジェクトに変換
            Graphics gCat = Graphics.FromImage(catImage);
            Graphics gDog = Graphics.FromImage(dogImage);
            // Graphicsオブジェクト（の中のcatImageに）baseImageを変換して描画
            gCat.DrawImage(baseCatImage, 0, 0, 224, 224);
            gDog.DrawImage(baseDogImage, 0, 0, 224, 224);
            // Graphicsオブジェクトを破棄し、メモリを解放
            gCat.Dispose();
            gDog.Dispose();

            // 訓練・テストデータにデータを格納
            // 先にテストデータの枚数分テストデータに保存し、その後訓練データを保存する
            // 画素値の値域は0 ~ 255のため、255で割ることで0 ~ 1に正規化
            if (i < TEST_DATA_LENGTH)
            {
                // ImageをNdArrayに変換したものをvggに入力し、出力した特徴量を入力データとして保存
                testData[i * 2] = vgg.Predict(NdArrayConverter.Image2NdArray(catImage, false, true) / 255.0)[0];
                testLabel[i * 2] = new NdArray(new Real[] { 0 });
                testData[i * 2 + 1] = vgg.Predict(NdArrayConverter.Image2NdArray(dogImage, false, true) / 255.0)[0];
                testLabel[i * 2 + 1] = new NdArray(new Real[] { 1 });
            }
            else
            {
                trainData[(i - TEST_DATA_LENGTH) * 2] = vgg.Predict(NdArrayConverter.Image2NdArray(catImage, false, true) / 255.0)[0];
                trainLabel[(i - TEST_DATA_LENGTH) * 2] = new NdArray(new Real[] { 0 });//new Real [] { 0 };
                trainData[(i - TEST_DATA_LENGTH) * 2] = vgg.Predict(NdArrayConverter.Image2NdArray(dogImage, false, true) / 255.0)[0];
                trainLabel[(i - TEST_DATA_LENGTH) * 2] = new NdArray(new Real[] { 1 });// = new Real [] { 1 };
            }
        }

        Console.WriteLine("Training Start...");

        // ミニバッチ用のNdArrayを定義
        NdArray batchData = new NdArray(new[] { 4096 }, BATCH_SIZE);
        NdArray batchLabel = new NdArray(new[] { 1 }, BATCH_SIZE);

        // 誤差関数を定義（今回は二値分類なので二乗誤差関数(MSE)）
        LossFunction lossFunction = new MeanSquaredError();

        // エポックを回す
        for (int epoch = 0; epoch < 10; epoch++)
        {
            // １エポックで訓練データ // バッチサイズ の回数分学習
            for (int step = 0; step < TRAIN_DATA_COUNT; step++)
            {

                // ミニバッチを用意
                for (int i = 0; i < BATCH_SIZE; i++)
                {
                    // 0 ~ 訓練データサイズ-1 の中からランダムで整数を取得
                    int index = Mother.Dice.Next(trainData.Length);
                    // trainData(NdArray[])を、batchData(NdArray)の形にコピー
                    Array.Copy(trainData[index].Data, 0, batchData.Data, i * batchData.Length, batchData.Length);
                    batchLabel.Data[i] = trainLabel[index].Data[0];
                }

                // 学習（順伝播、誤差の計算、逆伝播、更新）
                NdArray[] output = nn.Forward(batchData);
                Real loss = lossFunction.Evaluate(output, batchLabel);
                nn.Backward(output);
                nn.Update();
            }

            // 認識率（accuracy）の計算
            // テストデータの回数データを回す
            Real accuracy = 0;
            for (int i = 0; i < TEST_DATA_LENGTH * 2; i++)
            {
                NdArray[] output = nn.Predict(testData[i]);
                // 出力outputと正解の誤差が0.5以下（正解が0のときにoutput<0.5、正解が1のときにoutput>0.5）
                // の際に正確に認識したとする
                if (Math.Abs(output[0].Data[0] - trainLabel[i].Data[0]) < 0.5)
                {
                    accuracy += 1;
                }
                accuracy /= TEST_DATA_LENGTH * 2.0;
                Console.WriteLine("Epoch:" + epoch + "accuracy:" + accuracy);
            }
        }
    }
}