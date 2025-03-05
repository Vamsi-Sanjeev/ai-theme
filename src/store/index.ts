import { create } from 'zustand';
import type { Employee, Risk, CustomerFeedback, AIRecommendation } from '../types';

interface AppState {
  employees: Employee[];
  risks: Risk[];
  feedback: CustomerFeedback[];
  recommendations: AIRecommendation[];
  addEmployee: (employee: Employee) => void;
  addRisk: (risk: Risk) => void;
  addFeedback: (feedback: CustomerFeedback) => void;
  addRecommendation: (recommendation: AIRecommendation) => void;
  updateEmployee: (id: string, updates: Partial<Employee>) => void;
  removeEmployee: (id: string) => void;
  removeRisk: (id: string) => void;
}

export const useStore = create<AppState>((set) => ({
  employees: [],
  risks: [],
  feedback: [],
  recommendations: [],
  
  addEmployee: (employee) => 
    set((state) => ({ employees: [...state.employees, employee] })),
  
  addRisk: (risk) => 
    set((state) => ({ risks: [...state.risks, risk] })),
  
  addFeedback: (feedback) => 
    set((state) => ({ feedback: [...state.feedback, feedback] })),
  
  addRecommendation: (recommendation) => 
    set((state) => ({ recommendations: [...state.recommendations, recommendation] })),
  
  updateEmployee: (id, updates) =>
    set((state) => ({
      employees: state.employees.map((emp) =>
        emp.id === id ? { ...emp, ...updates } : emp
      ),
    })),
  
  removeEmployee: (id) =>
    set((state) => ({
      employees: state.employees.filter((emp) => emp.id !== id),
    })),
  
  removeRisk: (id) =>
    set((state) => ({
      risks: state.risks.filter((risk) => risk.id !== id),
    })),
}));