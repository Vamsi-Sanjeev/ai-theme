import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

const revenueData = [
  { month: 'Jan', value: 65000 },
  { month: 'Feb', value: 75000 },
  { month: 'Mar', value: 85000 },
  { month: 'Apr', value: 95000 },
  { month: 'May', value: 105000 },
  { month: 'Jun', value: 124500 },
];

const satisfactionData = [
  { month: 'Jan', satisfaction: 88 },
  { month: 'Feb', satisfaction: 90 },
  { month: 'Mar', satisfaction: 92 },
  { month: 'Apr', satisfaction: 91 },
  { month: 'May', satisfaction: 93 },
  { month: 'Jun', satisfaction: 94 },
];

export const RevenueChart: React.FC<{ darkMode: boolean }> = ({ darkMode }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={revenueData}>
        <defs>
          <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8}/>
            <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
        <XAxis 
          dataKey="month" 
          stroke={darkMode ? '#9ca3af' : '#4b5563'}
        />
        <YAxis 
          stroke={darkMode ? '#9ca3af' : '#4b5563'}
          tickFormatter={(value) => `$${value.toLocaleString()}`}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: darkMode ? '#1f2937' : '#ffffff',
            borderColor: darkMode ? '#374151' : '#e5e7eb',
            color: darkMode ? '#ffffff' : '#000000'
          }}
        />
        <Area 
          type="monotone" 
          dataKey="value" 
          stroke="#6366f1" 
          fillOpacity={1} 
          fill="url(#colorRevenue)" 
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};

export const SatisfactionChart: React.FC<{ darkMode: boolean }> = ({ darkMode }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={satisfactionData}>
        <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
        <XAxis 
          dataKey="month" 
          stroke={darkMode ? '#9ca3af' : '#4b5563'}
        />
        <YAxis 
          stroke={darkMode ? '#9ca3af' : '#4b5563'}
          tickFormatter={(value) => `${value}%`}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: darkMode ? '#1f2937' : '#ffffff',
            borderColor: darkMode ? '#374151' : '#e5e7eb',
            color: darkMode ? '#ffffff' : '#000000'
          }}
        />
        <Bar dataKey="satisfaction" fill="#10b981" />
      </BarChart>
    </ResponsiveContainer>
  );
};