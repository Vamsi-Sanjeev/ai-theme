import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertCircle, Search, Filter, Plus, AlertTriangle, TrendingUp } from 'lucide-react';
import { RiskCard } from '../components/RiskCard';
import { AddRiskModal } from '../components/AddRiskModal';
import { useStore } from '../store';
import type { AIRecommendation } from '../types';

interface RisksProps {
  darkMode: boolean;
  setSelectedRecommendation: (rec: AIRecommendation | null) => void;
}

export const Risks: React.FC<RisksProps> = ({ darkMode, setSelectedRecommendation }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBy, setFilterBy] = useState<'all' | 'risks' | 'opportunities'>('all');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  const risks = useStore((state) => state.risks);

  const filteredRisks = risks.filter(risk => {
    const matchesSearch = risk.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         risk.description.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (filterBy === 'risks') {
      return matchesSearch && risk.type === 'risk';
    } else if (filterBy === 'opportunities') {
      return matchesSearch && risk.type === 'opportunity';
    }
    
    return matchesSearch;
  });

  const riskStats = {
    high: filteredRisks.filter(r => r.type === 'risk' && r.severity === 'high').length,
    medium: filteredRisks.filter(r => r.type === 'risk' && r.severity === 'medium').length,
    low: filteredRisks.filter(r => r.type === 'risk' && r.severity === 'low').length,
  };

  const opportunityStats = {
    high: filteredRisks.filter(r => r.type === 'opportunity' && r.severity === 'high').length,
    medium: filteredRisks.filter(r => r.type === 'opportunity' && r.severity === 'medium').length,
    low: filteredRisks.filter(r => r.type === 'opportunity' && r.severity === 'low').length,
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <AlertCircle className="w-8 h-8 text-blue-500" />
          <h1 className="text-3xl font-bold">Risk Management</h1>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsAddModalOpen(true)}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg
            bg-blue-500 hover:bg-blue-600 text-white
            transition-colors duration-200
          `}
        >
          <Plus size={20} />
          <span>Add Risk/Opportunity</span>
        </motion.button>
      </div>

      <div className="flex flex-col md:flex-row gap-4">
        <div className={`
          flex-1 relative
          ${darkMode ? 'text-white' : 'text-gray-900'}
        `}>
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search risks and opportunities..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={`
              w-full pl-10 pr-4 py-2 rounded-lg
              ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
              border focus:outline-none focus:ring-2 focus:ring-blue-500
            `}
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="text-gray-400" />
          <select
            value={filterBy}
            onChange={(e) => setFilterBy(e.target.value as 'all' | 'risks' | 'opportunities')}
            className={`
              px-4 py-2 rounded-lg
              ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
              border focus:outline-none focus:ring-2 focus:ring-blue-500
            `}
          >
            <option value="all">All Items</option>
            <option value="risks">Risks Only</option>
            <option value="opportunities">Opportunities Only</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredRisks.map((risk) => (
          <RiskCard
            key={risk.id}
            risk={risk}
            darkMode={darkMode}
          />
        ))}
      </div>

      {filteredRisks.length === 0 && (
        <div className={`
          text-center py-8
          ${darkMode ? 'text-gray-400' : 'text-gray-600'}
        `}>
          No risks or opportunities found matching your search criteria.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
        <div className={`
          p-6 rounded-xl border
          ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
        `}>
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <AlertTriangle className="text-red-500" />
            Risk Summary
          </h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>High Severity</span>
              <span className="text-red-500 font-bold">{riskStats.high}</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Medium Severity</span>
              <span className="text-yellow-500 font-bold">{riskStats.medium}</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Low Severity</span>
              <span className="text-green-500 font-bold">{riskStats.low}</span>
            </div>
          </div>
        </div>

        <div className={`
          p-6 rounded-xl border
          ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
        `}>
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="text-green-500" />
            Opportunity Summary
          </h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>High Impact</span>
              <span className="text-green-500 font-bold">{opportunityStats.high}</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Medium Impact</span>
              <span className="text-blue-500 font-bold">{opportunityStats.medium}</span>
            </div>
            <div className="flex justify-between items-center">
              <span>Low Impact</span>
              <span className="text-gray-500 font-bold">{opportunityStats.low}</span>
            </div>
          </div>
        </div>
      </div>

      <AddRiskModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        darkMode={darkMode}
      />
    </div>
  );
};