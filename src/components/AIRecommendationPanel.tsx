import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ChevronRight } from 'lucide-react';
import type { AIRecommendation } from '../types';

interface AIRecommendationPanelProps {
  recommendation: AIRecommendation | null;
  onClose: () => void;
  darkMode: boolean;
}

export const AIRecommendationPanel: React.FC<AIRecommendationPanelProps> = ({
  recommendation,
  onClose,
  darkMode,
}) => {
  return (
    <AnimatePresence>
      {recommendation && (
        <motion.div
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 25, stiffness: 200 }}
          className={`
            fixed top-0 right-0 w-96 h-full
            ${darkMode ? 'bg-gray-800' : 'bg-white'}
            border-l ${darkMode ? 'border-gray-700' : 'border-gray-200'}
            p-6 shadow-xl z-50
          `}
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold">AI Recommendation</h2>
            <button
              onClick={onClose}
              className={`
                p-2 rounded-full
                ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}
              `}
            >
              <X size={20} />
            </button>
          </div>

          <div className={`
            inline-block px-3 py-1 rounded-full text-sm mb-4
            ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}
          `}>
            {recommendation.category.charAt(0).toUpperCase() + recommendation.category.slice(1)}
          </div>

          <h3 className="text-lg font-semibold mb-4">{recommendation.title}</h3>

          <div className="space-y-4">
            {recommendation.steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-start gap-3"
              >
                <div className={`
                  mt-1 p-1 rounded-full
                  ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}
                `}>
                  <ChevronRight size={16} />
                </div>
                <p className="text-sm">{step}</p>
              </motion.div>
            ))}
          </div>

          <div className="mt-8">
            <button className={`
              w-full py-2 rounded-lg font-medium
              bg-blue-500 hover:bg-blue-600 text-white
              transition-colors duration-200
            `}>
              Implement Recommendation
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};