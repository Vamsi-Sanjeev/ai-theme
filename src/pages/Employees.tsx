import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Users, Search, Filter, Plus } from 'lucide-react';
import { EmployeeCard } from '../components/EmployeeCard';
import { AddEmployeeModal } from '../components/AddEmployeeModal';
import { useStore } from '../store';
import { mlPredictor } from '../services/mlPredictor';
import type { AIRecommendation } from '../types';

interface EmployeesProps {
  darkMode: boolean;
  setSelectedRecommendation: (rec: AIRecommendation | null) => void;
}

export const Employees: React.FC<EmployeesProps> = ({ darkMode, setSelectedRecommendation }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBy, setFilterBy] = useState<'all' | 'high-risk' | 'overworked'>('all');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isMLInitialized, setIsMLInitialized] = useState(false);

  const employees = useStore((state) => state.employees);

  useEffect(() => {
    const initializeML = async () => {
      if (!isMLInitialized) {
        await mlPredictor.initializeWithSampleData();
        setIsMLInitialized(true);
      }
    };

    initializeML();
  }, [isMLInitialized]);

  const filteredEmployees = employees.filter(employee => {
    const matchesSearch = employee.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         employee.role.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (filterBy === 'high-risk') {
      return matchesSearch && employee.burnoutRisk >= 60;
    } else if (filterBy === 'overworked') {
      return matchesSearch && employee.workload >= 80;
    }
    
    return matchesSearch;
  });

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Users className="w-8 h-8 text-blue-500" />
          <h1 className="text-3xl font-bold">Employee Management</h1>
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
          <span>Add Employee</span>
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
            placeholder="Search employees..."
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
            onChange={(e) => setFilterBy(e.target.value as 'all' | 'high-risk' | 'overworked')}
            className={`
              px-4 py-2 rounded-lg
              ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
              border focus:outline-none focus:ring-2 focus:ring-blue-500
            `}
          >
            <option value="all">All Employees</option>
            <option value="high-risk">High Risk</option>
            <option value="overworked">Overworked</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredEmployees.map((employee) => (
          <EmployeeCard
            key={employee.id}
            employee={employee}
            darkMode={darkMode}
          />
        ))}
      </div>

      {filteredEmployees.length === 0 && (
        <div className={`
          text-center py-8
          ${darkMode ? 'text-gray-400' : 'text-gray-600'}
        `}>
          No employees found matching your search criteria.
        </div>
      )}

      <AddEmployeeModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        darkMode={darkMode}
      />
    </div>
  );
};