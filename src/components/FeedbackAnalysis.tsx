import React from 'react';
import { motion } from 'framer-motion';
import { Smile, Meh, Frown } from 'lucide-react';
import type { CustomerFeedback } from '../types';

interface FeedbackAnalysisProps {
  feedback: CustomerFeedback[];
  darkMode: boolean;
}

export const FeedbackAnalysis: React.FC<FeedbackAnalysisProps> = ({ feedback, darkMode }) => {
  const sentimentCounts = feedback.reduce((acc, item) => {
    acc[item.sentiment]++;
    return acc;
  }, { positive: 0, neutral: 0, negative: 0 });

  const total = feedback.length;
  const getPercentage = (count: number) => ((count / total) * 100).toFixed(1);

  return (
    <div className={`
      p-6 rounded-xl border
      ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
    `}>
      <h2 className="text-xl font-bold mb-6">Customer Sentiment Analysis</h2>

      <div className="space-y-6">
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Smile className="text-green-500" />
              <span>Positive</span>
            </div>
            <span>{getPercentage(sentimentCounts.positive)}%</span>
          </div>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${getPercentage(sentimentCounts.positive)}%` }}
            className="h-2 bg-green-500 rounded-full"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Meh className="text-yellow-500" />
              <span>Neutral</span>
            </div>
            <span>{getPercentage(sentimentCounts.neutral)}%</span>
          </div>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${getPercentage(sentimentCounts.neutral)}%` }}
            className="h-2 bg-yellow-500 rounded-full"
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Frown className="text-red-500" />
              <span>Negative</span>
            </div>
            <span>{getPercentage(sentimentCounts.negative)}%</span>
          </div>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${getPercentage(sentimentCounts.negative)}%` }}
            className="h-2 bg-red-500 rounded-full"
          />
        </div>
      </div>

      <div className="mt-8">
        <h3 className="font-semibold mb-4">Recent Feedback</h3>
        <div className="space-y-3">
          {feedback.slice(0, 3).map((item) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className={`
                p-3 rounded-lg border
                ${darkMode ? 'border-gray-700' : 'border-gray-200'}
              `}
            >
              <div className="flex items-center gap-2 mb-2">
                {item.sentiment === 'positive' && <Smile className="text-green-500" size={16} />}
                {item.sentiment === 'neutral' && <Meh className="text-yellow-500" size={16} />}
                {item.sentiment === 'negative' && <Frown className="text-red-500" size={16} />}
                <span className="text-sm text-gray-500">{item.source}</span>
              </div>
              <p className="text-sm">{item.text}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};