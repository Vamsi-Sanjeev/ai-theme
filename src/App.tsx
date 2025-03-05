import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Sun, Moon, Menu } from 'lucide-react';
import { Sidebar } from './components/Sidebar';
import { AIRecommendationPanel } from './components/AIRecommendationPanel';
import { Dashboard } from './pages/Dashboard';
import { Predictions } from './pages/Predictions';
import { Employees } from './pages/Employees';
import { Risks } from './pages/Risks';
import { Feedback } from './pages/Feedback';
import type { AIRecommendation } from './types';
import './styles/animations.css';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [selectedRecommendation, setSelectedRecommendation] = useState<AIRecommendation | null>(null);

  const toggleDarkMode = () => setDarkMode(!darkMode);
  const toggleSidebar = () => setSidebarOpen(!isSidebarOpen);

  return (
    <Router>
      <div className={`min-h-screen ${darkMode ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
        <Sidebar 
          darkMode={darkMode} 
          isOpen={isSidebarOpen} 
        />
        <AIRecommendationPanel
          recommendation={selectedRecommendation}
          onClose={() => setSelectedRecommendation(null)}
          darkMode={darkMode}
        />

        <div className={`
          ${isSidebarOpen ? 'ml-64' : 'ml-0'}
          transition-all duration-300
        `}>
          <motion.header 
            initial={false}
            animate={{
              marginLeft: isSidebarOpen ? '16rem' : '0',
            }}
            transition={{ duration: 0.3 }}
            className={`
              fixed top-0 right-0 left-0
              ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} 
              border-b px-4 py-3 z-10
            `}
          >
            <div className="flex items-center justify-between">
              <motion.button 
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={toggleSidebar}
                className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
              >
                <Menu size={24} className={darkMode ? 'text-white' : 'text-gray-600'} />
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                onClick={toggleDarkMode}
                className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
              >
                {darkMode ? (
                  <Sun size={24} className="text-white" />
                ) : (
                  <Moon size={24} className="text-gray-600" />
                )}
              </motion.button>
            </div>
          </motion.header>

          <main className={`pt-16 p-6 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route 
                path="/dashboard" 
                element={
                  <Dashboard 
                    darkMode={darkMode}
                    setSelectedRecommendation={setSelectedRecommendation}
                  />
                } 
              />
              <Route 
                path="/predictions" 
                element={
                  <Predictions 
                    darkMode={darkMode}
                    setSelectedRecommendation={setSelectedRecommendation}
                  />
                } 
              />
              <Route 
                path="/employees" 
                element={
                  <Employees 
                    darkMode={darkMode}
                    setSelectedRecommendation={setSelectedRecommendation}
                  />
                } 
              />
              <Route 
                path="/risks" 
                element={
                  <Risks 
                    darkMode={darkMode}
                    setSelectedRecommendation={setSelectedRecommendation}
                  />
                } 
              />
              <Route 
                path="/feedback" 
                element={
                  <Feedback 
                    darkMode={darkMode}
                    setSelectedRecommendation={setSelectedRecommendation}
                  />
                } 
              />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;