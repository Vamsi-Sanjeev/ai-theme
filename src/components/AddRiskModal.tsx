import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { useStore } from '../store';
import type { Risk } from '../types';

interface AddRiskModalProps {
  isOpen: boolean;
  onClose: () => void;
  darkMode: boolean;
}

export const AddRiskModal: React.FC<AddRiskModalProps> = ({
  isOpen,
  onClose,
  darkMode,
}) => {
  const addRisk = useStore((state) => state.addRisk);
  const [formData, setFormData] = useState<Partial<Risk>>({
    type: 'risk',
    title: '',
    description: '',
    severity: 'medium',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newRisk: Risk = {
      id: crypto.randomUUID(),
      type: formData.type as 'risk' | 'opportunity',
      title: formData.title || '',
      description: formData.description || '',
      severity: formData.severity as 'low' | 'medium' | 'high',
      timestamp: new Date().toISOString(),
    };
    addRisk(newRisk);
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className={`
              w-full max-w-md p-6 rounded-xl
              ${darkMode ? 'bg-gray-800' : 'bg-white'}
              shadow-xl
            `}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold">Add New Risk/Opportunity</h2>
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

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Type</label>
                <select
                  value={formData.type}
                  onChange={(e) => setFormData({ ...formData, type: e.target.value as 'risk' | 'opportunity' })}
                  className={`
                    w-full px-3 py-2 rounded-lg
                    ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}
                    border focus:outline-none focus:ring-2 focus:ring-blue-500
                  `}
                  required
                >
                  <option value="risk">Risk</option>
                  <option value="opportunity">Opportunity</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Title</label>
                <input
                  type="text"
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  className={`
                    w-full px-3 py-2 rounded-lg
                    ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}
                    border focus:outline-none focus:ring-2 focus:ring-blue-500
                  `}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Description</label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className={`
                    w-full px-3 py-2 rounded-lg
                    ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}
                    border focus:outline-none focus:ring-2 focus:ring-blue-500
                  `}
                  rows={3}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Severity</label>
                <select
                  value={formData.severity}
                  onChange={(e) => setFormData({ ...formData, severity: e.target.value as 'low' | 'medium' | 'high' })}
                  className={`
                    w-full px-3 py-2 rounded-lg
                    ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}
                    border focus:outline-none focus:ring-2 focus:ring-blue-500
                  `}
                  required
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium"
              >
                Add {formData.type === 'risk' ? 'Risk' : 'Opportunity'}
              </motion.button>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};