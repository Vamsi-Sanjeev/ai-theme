import React from 'react';
import { motion } from 'framer-motion';
import { Brain, TrendingUp, TrendingDown, DollarSign, Users, Building } from 'lucide-react';

interface PredictionsProps {
  darkMode: boolean;
  setSelectedRecommendation: (rec: any) => void;
}

export const Predictions: React.FC<PredictionsProps> = ({ darkMode, setSelectedRecommendation }) => {
  const predictions = [
    {
      title: 'Revenue Forecast',
      icon: DollarSign,
      prediction: 'Expected 15% growth in Q2 2024',
      confidence: 89,
      trend: 'up',
    },
    {
      title: 'Market Expansion',
      icon: Building,
      prediction: 'High potential in Southeast region',
      confidence: 78,
      trend: 'up',
    },
    {
      title: 'Employee Turnover',
      icon: Users,
      prediction: 'Risk of 3 key departures in next 2 months',
      confidence: 72,
      trend: 'down',
    },
  ];

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <Brain className="w-8 h-8 text-blue-500" />
        <h1 className="text-3xl font-bold">AI Predictions</h1>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {predictions.map((prediction, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`
              p-6 rounded-xl border
              ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
            `}
          >
            <div className="flex items-center gap-3 mb-4">
              <prediction.icon className="w-6 h-6 text-blue-500" />
              <h3 className="text-xl font-semibold">{prediction.title}</h3>
            </div>

            <p className={`
              text-lg mb-4
              ${darkMode ? 'text-gray-300' : 'text-gray-700'}
            `}>
              {prediction.prediction}
            </p>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">Confidence:</span>
                <span className="font-semibold text-blue-500">{prediction.confidence}%</span>
              </div>
              {prediction.trend === 'up' ? (
                <TrendingUp className="w-5 h-5 text-green-500" />
              ) : (
                <TrendingDown className="w-5 h-5 text-red-500" />
              )}
            </div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setSelectedRecommendation({
                id: `${index}`,
                category: 'expansion',
                title: `AI Recommendations for ${prediction.title}`,
                steps: [
                  'Analyze historical data trends',
                  'Identify key growth factors',
                  'Develop action plan',
                  'Monitor implementation progress'
                ],
                impact: 'high'
              })}
              className={`
                w-full mt-4 py-2 rounded-lg
                bg-blue-500 hover:bg-blue-600
                text-white font-medium
                transition-colors duration-200
              `}
            >
              View Detailed Analysis
            </motion.button>
          </motion.div>
        ))}
      </div>

      <div className={`
        p-6 rounded-xl border mt-8
        ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
      `}>
        <h2 className="text-2xl font-bold mb-6">Long-term Predictions</h2>
        <div className="space-y-4">
          {[
            { year: '2024', prediction: 'Market expansion to 3 new regions', probability: 85 },
            { year: '2025', prediction: 'Launch of AI-driven customer service', probability: 72 },
            { year: '2026', prediction: 'Integration of blockchain technology', probability: 64 },
          ].map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`
                p-4 rounded-lg border
                ${darkMode ? 'border-gray-700' : 'border-gray-200'}
              `}
            >
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-sm text-blue-500 font-semibold">{item.year}</span>
                  <p className="mt-1">{item.prediction}</p>
                </div>
                <div className="text-right">
                  <span className="text-sm text-gray-500">Probability</span>
                  <p className="font-semibold">{item.probability}%</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};