import React from 'react';
import { RiskCard } from '../components/RiskCard';
import { EmployeeCard } from '../components/EmployeeCard';
import { FeedbackAnalysis } from '../components/FeedbackAnalysis';
import { RevenueChart, SatisfactionChart } from '../components/Charts';
import type { Risk, Employee, CustomerFeedback, AIRecommendation } from '../types';

// Mock data
const mockRisks: Risk[] = [
  {
    id: '1',
    type: 'risk',
    title: 'Low Stock Alert',
    description: 'Trending sneakers inventory below threshold. Restock needed within 7 days.',
    severity: 'high',
    timestamp: new Date().toISOString(),
  },
  {
    id: '2',
    type: 'opportunity',
    title: 'Market Expansion',
    description: 'High demand detected in neighboring cities. Consider new store locations.',
    severity: 'medium',
    timestamp: new Date().toISOString(),
  },
];

const mockEmployees: Employee[] = [
  {
    id: '1',
    name: 'Sarah Johnson',
    role: 'Senior Developer',
    workload: 85,
    satisfaction: 75,
    burnoutRisk: 65,
  },
  {
    id: '2',
    name: 'Mike Chen',
    role: 'Product Manager',
    workload: 70,
    satisfaction: 90,
    burnoutRisk: 30,
  },
];

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
];

interface DashboardProps {
  darkMode: boolean;
  setSelectedRecommendation: (rec: AIRecommendation | null) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ darkMode, setSelectedRecommendation }) => {
  return (
    <>
      {/* Risks & Opportunities */}
      <section className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Risks & Opportunities</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {mockRisks.map((risk) => (
            <RiskCard key={risk.id} risk={risk} darkMode={darkMode} />
          ))}
        </div>
      </section>

      {/* Employee Monitoring */}
      <section className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Employee Monitoring</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {mockEmployees.map((employee) => (
            <EmployeeCard key={employee.id} employee={employee} darkMode={darkMode} />
          ))}
        </div>
      </section>

      {/* Customer Feedback & Analytics */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FeedbackAnalysis feedback={mockFeedback} darkMode={darkMode} />
        <div className={`
          p-6 rounded-xl border
          ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}
        `}>
          <h2 className="text-xl font-bold mb-4">Performance Metrics</h2>
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">Revenue Trend</h3>
              <RevenueChart darkMode={darkMode} />
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3">Customer Satisfaction</h3>
              <SatisfactionChart darkMode={darkMode} />
            </div>
          </div>
        </div>
      </section>
    </>
  );
};