import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  Users, 
  TrendingUp, 
  AlertCircle, 
  MessageSquare,
  Brain,
} from 'lucide-react';

interface SidebarProps {
  darkMode: boolean;
  isOpen: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ darkMode, isOpen }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    { icon: BarChart3, label: 'Dashboard', path: '/dashboard' },
    { icon: Brain, label: 'Predictions', path: '/predictions' },
    { icon: Users, label: 'Employees', path: '/employees' },
    { icon: AlertCircle, label: 'Risks', path: '/risks' },
    { icon: MessageSquare, label: 'Feedback', path: '/feedback' },
  ];

  return (
    <motion.aside
      initial={false}
      animate={{
        x: isOpen ? 0 : -256,
        opacity: isOpen ? 1 : 0,
      }}
      transition={{ duration: 0.3 }}
      className={`
        fixed top-0 left-0 h-full w-64
        ${darkMode ? 'bg-gray-800' : 'bg-white'} 
        border-r ${darkMode ? 'border-gray-700' : 'border-gray-200'}
        z-20
      `}
    >
      <div className="p-4">
        <motion.h1 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}
        >
          AI Business Suite
        </motion.h1>
      </div>
      
      <nav className="mt-8">
        {menuItems.map((item, index) => {
          const isActive = location.pathname === item.path;
          return (
            <motion.button
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 * (index + 1) }}
              onClick={() => navigate(item.path)}
              className={`
                w-full flex items-center gap-3 px-4 py-3
                ${darkMode ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-600 hover:bg-gray-100'}
                ${isActive ? (darkMode ? 'bg-gray-700' : 'bg-gray-100') : ''}
                transition-colors duration-200
              `}
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </motion.button>
          );
        })}
      </nav>
    </motion.aside>
  );
};