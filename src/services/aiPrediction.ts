import * as tf from '@tensorflow/tfjs';

class AIPredictor {
  private model: tf.Sequential | null = null;
  private isTraining = false;

  constructor() {
    this.initializeModel();
  }

  private initializeModel() {
    // Build the Neural Network Model with enhanced architecture
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: [2], units: 8, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    this.model.compile({
      optimizer: tf.train.adam(),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
  }

  async trainModel(employees: Array<{ workload: number; satisfaction: number; hasResigned?: boolean }>) {
    if (this.isTraining || !this.model) return;

    this.isTraining = true;
    console.log("Training started...");

    try {
      // Prepare training data
      const trainingData = tf.tensor2d(
        employees.map(emp => [emp.workload, emp.satisfaction])
      );

      const outputData = tf.tensor2d(
        employees.map(emp => [emp.hasResigned ? 1 : 0]),
        [employees.length, 1]
      );

      // Train the model with enhanced parameters
      await this.model.fit(trainingData, outputData, {
        epochs: 100,
        batchSize: 2,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (epoch % 10 === 0) {
              console.log(`Epoch ${epoch + 1}: loss = ${logs?.loss}, accuracy = ${logs?.acc}`);
            }
          }
        }
      });

      console.log("Training Complete!");

      // Clean up tensors
      trainingData.dispose();
      outputData.dispose();
    } catch (error) {
      console.error("Error training model:", error);
    } finally {
      this.isTraining = false;
    }
  }

  async predictResignationRisk(workload: number, satisfaction: number): Promise<number> {
    if (!this.model) return 0;

    try {
      const testEmployee = tf.tensor2d([[workload, satisfaction]]);
      const prediction = await this.model.predict(testEmployee) as tf.Tensor;
      const result = await prediction.data();
      
      // Clean up tensors
      testEmployee.dispose();
      prediction.dispose();

      return result[0];
    } catch (error) {
      console.error("Error making prediction:", error);
      return 0;
    }
  }

  // Enhanced sample data for initial training
  async initializeWithSampleData() {
    const sampleData = [
      { workload: 40, satisfaction: 85, hasResigned: false },  // Balanced workload, high performance
      { workload: 50, satisfaction: 60, hasResigned: true },   // Overworked, lower performance
      { workload: 30, satisfaction: 90, hasResigned: false },  // Low workload, high performance
      { workload: 55, satisfaction: 50, hasResigned: true },   // Heavy workload, low performance
      { workload: 45, satisfaction: 75, hasResigned: false },  // Moderate workload, medium performance
      { workload: 60, satisfaction: 40, hasResigned: true },   // Extreme workload, poor performance
      { workload: 35, satisfaction: 80, hasResigned: false },  // Comfortable workload, good performance
      { workload: 52, satisfaction: 55, hasResigned: true },   // High workload, mediocre performance
      { workload: 42, satisfaction: 88, hasResigned: false },  // Optimal workload, excellent performance
      { workload: 58, satisfaction: 45, hasResigned: true },   // Overloaded, struggling performance
    ];

    await this.trainModel(sampleData);
  }
}

export const aiPredictor = new AIPredictor();