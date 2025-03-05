import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Search, Filter, Smile, Meh, Frown } from 'lucide-react';
import { FeedbackAnalysis } from '../components/FeedbackAnalysis';
import type { CustomerFeedback } from '../types';

interface FeedbackProps {
  darkMode: boolean;
  setSelectedRecommendation: (rec: any) => void;
}

const mockFeedback: CustomerFeedback[] = [
  {
    id: '1',
    sentiment: 'negative',
    text: 'Service was slow during peak hours',
    source: 'Google Reviews',
    timestamp: new Date().toISOString(),
  },
  {
    id: '2',
    sentiment: 'positive',
    text: 'Great product quality and helpful staff',
    source: 'Facebook',
    timestamp: new Date().toISOString(),
  },
  {
    id: '3',
    sentiment: 'neutral',
    text: 'Average experience, nothing special',
    source: 'Yelp',
    timestamp: new Date().toISOString(),
  },
  {
    id: '4',
    sentiment: 'positive',
    text: 'The new mobile app is fantastic!',
    source: 'App Store',
    timestamp: new Date().toISOString(),
  },
  {
    id: '5',
    sentiment: 'negative',
    text: 'Website checkout process is confusing',
    source: 'Twitter',
    timestamp: new Date().toISOString(),
  },
  {
    id: '6',
    sentiment: 'positive',
    text: 'Customer support resolved my issue quickly',
    source: 'Email',
    timestamp: new Date().toISOString(),
  },
];

export const Feedback: React.FC<FeedbackProps> = ({ darkMode, setSelectedRecommendation }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBy, setFilterBy] = useState<'all' | 'positive' | 'negative' | 'neutral'>('all');

  const filteredFeedback = mockFeedback.filter(feedback => {
    const matchesSearch = feedback.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         feedback.source.toLowerCase().includes(searchTerm.toLowerCase());
    
    if (filterBy !== 'all') {
      return matchesSearch && feedback.sentiment === filterBy;
    }
    
    return matchesSearch;
  });

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <Smile className="text-green-500" size={20} />;
      case 'neutral':
        return <Meh className="text-yellow-500" size={20} />;
      case 'negative':
        return <Frown className="text-red-500" size={20} />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-3">
        <MessageSquare className="w-8 h-8 text-blue-500" />
        <h1 className="text-3xl font-bold">Customer Feedback</h1>
      </div>

      <div className="flex flex-col md:flex-row gap-4">
        <div className={`
          flex-1 relative
          ${darkMode ? 'text-white' : 'text-gray-900'}
        `}>
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search feedback..."
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
            onChange={(e) => setFilterBy(e.target.value as 'all' | 'positive' | 'negative' | 'neutral')}
            className={`
              px-4 py-2 rounded-lg
              ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
              border focus:outline-none focus:ring-2 focus:ring-blue-500
            `}
          >
            <option value="all">All Feedback</option>
            <option value="positive">Positive Only</option>
            <option value="neutral">Neutral Only</option>
            <option value="negative">Negative Only</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeedbackAnalysis feedback={filteredFeedback} darkMode={darkMode} />
        
        <div className={`
          p-6 rounded-xl border
          ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
        `}>
          <h2 className="text-xl font-bold mb-6">Recent Feedback</h2>
          <div className="space-y-4">
            {filteredFeedback.map((item) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`
                  p-4 rounded-lg border
                  ${darkMode ? 'border-gray-700' : 'border-gray-200'}
                `}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getSentimentIcon(item.sentiment)}
                    <span className="text-sm font-medium">{item.source}</span>
                  </div>
                  <span className="text-sm text-gray-500">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </span>
                </div>
                <p className={`
                  text-sm
                  ${darkMode ? 'text-gray-300' : 'text-gray-600'}
                `}>
                  {item.text}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};