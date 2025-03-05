import * as tf from '@tensorflow/tfjs';
import * as brain from 'brain.js';
import * as ml from 'ml-regression';

export class MLPredictor {
  private tfModel: tf.Sequential | null = null;
  private brainModel: brain.NeuralNetwork | null = null;
  private regressionModel: ml.SimpleLinearRegression | null = null;
  private isTraining = false;

  constructor() {
    this.initializeModels();
  }

  private initializeModels() {
    // Initialize TensorFlow.js model
    this.tfModel = tf.sequential();
    this.tfModel.add(tf.layers.dense({ inputShape: [2], units: 8, activation: 'relu' }));
    this.tfModel.add(tf.layers.dense({ units: 4, activation: 'relu' }));
    this.tfModel.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    this.tfModel.compile({
      optimizer: tf.train.adam(),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    // Initialize Brain.js model
    this.brainModel = new brain.NeuralNetwork();

    // ML.js model will be initialized during training
  }

  private normalizeData(data: number[]): number[] {
    const min = Math.min(...data);
    const max = Math.max(...data);
    return data.map(v => (v - min) / (max - min));
  }

  async trainModels(employees: Array<{ workload: number; satisfaction: number; hasResigned: boolean }>) {
    if (this.isTraining) return;
    this.isTraining = true;

    try {
      console.log("Training ML models...");

      // Prepare data
      const workload = employees.map(e => e.workload);
      const satisfaction = employees.map(e => e.satisfaction);
      const labels = employees.map(e => e.hasResigned ? 1 : 0);

      const normWorkload = this.normalizeData(workload);
      const normSatisfaction = this.normalizeData(satisfaction);

      // Train TensorFlow.js model
      const tfData = tf.tensor2d(
        normWorkload.map((w, i) => [w, normSatisfaction[i]])
      );
      const tfLabels = tf.tensor2d(labels, [labels.length, 1]);

      await this.tfModel!.fit(tfData, tfLabels, {
        epochs: 200,
        batchSize: 2,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (epoch % 50 === 0) {
              console.log(`TF Epoch ${epoch}: Loss = ${logs?.loss.toFixed(4)}, Accuracy = ${logs?.acc.toFixed(4)}`);
            }
          }
        }
      });

      // Train Brain.js model
      const brainData = employees.map(e => ({
        input: [e.workload / 100, e.satisfaction / 100],
        output: [e.hasResigned ? 1 : 0]
      }));
      await this.brainModel!.trainAsync(brainData, {
        iterations: 2000,
        errorThresh: 0.005
      });

      // Train ML.js model
      this.regressionModel = new ml.SimpleLinearRegression(workload, labels);

      console.log("All models trained successfully!");

      // Cleanup
      tfData.dispose();
      tfLabels.dispose();
    } catch (error) {
      console.error("Error training models:", error);
    } finally {
      this.isTraining = false;
    }
  }

  async predict(workload: number, satisfaction: number): Promise<{
    tfPrediction: number;
    brainPrediction: number;
    mlPrediction: number;
    consensusPrediction: number;
  }> {
    try {
      // TensorFlow.js prediction
      const tfInput = tf.tensor2d([[workload / 100, satisfaction / 100]]);
      const tfPrediction = (await this.tfModel!.predict(tfInput).data())[0];
      tfInput.dispose();

      // Brain.js prediction
      const brainPrediction = (this.brainModel!.run([workload / 100, satisfaction / 100]))[0];

      // ML.js prediction
      const mlPrediction = this.regressionModel!.predict(workload);

      // Calculate consensus prediction (weighted average)
      const consensusPrediction = (
        tfPrediction * 0.4 +    // TensorFlow.js (40% weight)
        brainPrediction * 0.4 + // Brain.js (40% weight)
        mlPrediction * 0.2      // ML.js (20% weight)
      );

      return {
        tfPrediction,
        brainPrediction,
        mlPrediction,
        consensusPrediction
      };
    } catch (error) {
      console.error("Error making predictions:", error);
      return {
        tfPrediction: 0,
        brainPrediction: 0,
        mlPrediction: 0,
        consensusPrediction: 0
      };
    }
  }

  async initializeWithSampleData() {
    const sampleData = [
      { workload: 40, satisfaction: 85, hasResigned: false },
      { workload: 50, satisfaction: 60, hasResigned: true },
      { workload: 30, satisfaction: 90, hasResigned: false },
      { workload: 55, satisfaction: 50, hasResigned: true },
      { workload: 45, satisfaction: 75, hasResigned: false },
      { workload: 60, satisfaction: 40, hasResigned: true },
      { workload: 35, satisfaction: 80, hasResigned: false },
      { workload: 48, satisfaction: 65, hasResigned: true },
      { workload: 52, satisfaction: 55, hasResigned: true },
      { workload: 38, satisfaction: 88, hasResigned: false }
    ];

    await this.trainModels(sampleData);
  }
}

export const mlPredictor = new MLPredictor();