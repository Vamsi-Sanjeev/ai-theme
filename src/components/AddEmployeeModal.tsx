import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { useStore } from '../store';

interface AddEmployeeModalProps {
  isOpen: boolean;
  onClose: () => void;
  darkMode: boolean;
}

export const AddEmployeeModal: React.FC<AddEmployeeModalProps> = ({
  isOpen,
  onClose,
  darkMode,
}) => {
  const addEmployee = useStore((state) => state.addEmployee);
  const [formData, setFormData] = useState({
    name: '',
    role: '',
    workload: 50,
    satisfaction: 75,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newEmployee = {
      id: crypto.randomUUID(),
      ...formData,
      burnoutRisk: Math.round((formData.workload - formData.satisfaction + 100) / 3),
    };
    addEmployee(newEmployee);
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
              <h2 className="text-xl font-bold">Add New Employee</h2>
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
                <label className="block text-sm font-medium mb-1">Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className={`
                    w-full px-3 py-2 rounded-lg
                    ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}
                    border focus:outline-none focus:ring-2 focus:ring-blue-500
                  `}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Role</label>
                <input
                  type="text"
                  value={formData.role}
                  onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                  className={`
                    w-full px-3 py-2 rounded-lg
                    ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}
                    border focus:outline-none focus:ring-2 focus:ring-blue-500
                  `}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Workload (%) - {formData.workload}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={formData.workload}
                  onChange={(e) => setFormData({ ...formData, workload: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Satisfaction (%) - {formData.satisfaction}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={formData.satisfaction}
                  onChange={(e) => setFormData({ ...formData, satisfaction: parseInt(e.target.value) })}
                  className="w-full"
                />
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium"
              >
                Add Employee
              </motion.button>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};