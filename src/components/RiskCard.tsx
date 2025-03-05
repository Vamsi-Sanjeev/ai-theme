import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, TrendingUp } from 'lucide-react';
import type { Risk } from '../types';

interface RiskCardProps {
  risk: Risk;
  darkMode: boolean;
}

export const RiskCard: React.FC<RiskCardProps> = ({ risk, darkMode }) => {
  const isRisk = risk.type === 'risk';
  const Icon = isRisk ? AlertTriangle : TrendingUp;
  const colorClass = isRisk ? 'bg-red-500' : 'bg-green-500';

  return (
    <motion.div
      initial={{ rotateX: 0 }}
      whileHover={{ rotateX: 180 }}
      className="relative preserve-3d cursor-pointer h-48"
    >
      {/* Front */}
      <div className={`
        absolute w-full h-full backface-hidden rounded-xl p-4
        ${darkMode ? 'bg-gray-800' : 'bg-white'}
        ${darkMode ? 'border-gray-700' : 'border-gray-200'}
        border
      `}>
        <div className="flex items-center gap-2 mb-3">
          <Icon className={isRisk ? 'text-red-500' : 'text-green-500'} size={24} />
          <h3 className="text-lg font-semibold">{risk.title}</h3>
        </div>
        <div className={`
          inline-block px-2 py-1 rounded-full text-white text-sm mb-3
          ${colorClass}
        `}>
          {risk.severity.toUpperCase()}
        </div>
        <p className={`
          text-sm
          ${darkMode ? 'text-gray-300' : 'text-gray-600'}
        `}>
          {risk.description}
        </p>
      </div>

      {/* Back */}
      <div className={`
        absolute w-full h-full backface-hidden rounded-xl p-4 rotate-y-180
        ${darkMode ? 'bg-gray-800' : 'bg-white'}
        ${darkMode ? 'border-gray-700' : 'border-gray-200'}
        border
      `}>
        <h4 className="text-lg font-semibold mb-3">Recommended Actions</h4>
        <ul className="space-y-2">
          <li className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            <span className="text-sm">Review historical data</span>
          </li>
          <li className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            <span className="text-sm">Consult team leaders</span>
          </li>
          <li className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            <span className="text-sm">Implement mitigation plan</span>
          </li>
        </ul>
      </div>
    </motion.div>
  );
};