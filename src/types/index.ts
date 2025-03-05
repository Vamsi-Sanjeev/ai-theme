export interface Risk {
  id: string;
  type: 'risk' | 'opportunity';
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
  timestamp: string;
}

export interface Employee {
  id: string;
  name: string;
  role: string;
  workload: number;
  satisfaction: number;
  burnoutRisk: number;
}

export interface CustomerFeedback {
  id: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  text: string;
  source: string;
  timestamp: string;
}

export interface AIRecommendation {
  id: string;
  category: 'hiring' | 'marketing' | 'budgeting' | 'expansion';
  title: string;
  steps: string[];
  impact: 'low' | 'medium' | 'high';
}