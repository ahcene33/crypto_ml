# src/investment_scheduler.py
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class MonthlyInvestment:
    date: datetime
    amount: float
    cumulative: float
    portfolio_value_after: float = 0.0

class InvestmentTracker:
    def __init__(self, start_date: datetime = datetime(2026, 1, 22), 
                 monthly_amount: float = 50.0):
        self.start_date = start_date
        self.monthly_amount = monthly_amount
        self.investments: List[MonthlyInvestment] = []
        self.cumulative_total = 0.0
        
    def generate_schedule(self, end_date: datetime = None) -> List[MonthlyInvestment]:
        """Génère le calendrier d'investissement mensuel"""
        if end_date is None:
            end_date = datetime.now()
            
        current_date = self.start_date
        cumulative = 0.0
        
        while current_date <= end_date:
            cumulative = self.cumulative_total + self.monthly_amount
            investment = MonthlyInvestment(
                date=current_date,
                amount=self.monthly_amount,
                cumulative=cumulative
            )
            self.investments.append(investment)
            self.cumulative_total = cumulative
            
            # Passer au mois suivant, jour 22
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return self.investments
    
    def calculate_portfolio_with_investments(self, 
                                           base_growth_rate: float = 0.001,
                                           volatility: float = 0.02) -> pd.DataFrame:
        """Simule la croissance du portefeuille avec investissements"""
        if not self.investments:
            self.generate_schedule()
        
        # Créer une série temporelle quotidienne
        start = self.investments[0].date
        end = datetime.now()
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        portfolio_values = []
        cash_invested = []
        current_value = 0.0
        current_invested = 0.0
        
        # Simulation de croissance avec volatilité
        daily_returns = np.random.normal(base_growth_rate, volatility, len(date_range))
        
        for i, date in enumerate(date_range):
            # Ajouter les investissements mensuels
            for inv in self.investments:
                if inv.date.date() == date.date():
                    current_invested += inv.amount
                    current_value += inv.amount
            
            # Appliquer la croissance quotidienne
            if i > 0:
                current_value *= (1 + daily_returns[i])
            
            portfolio_values.append(current_value)
            cash_invested.append(current_invested)
        
        return pd.DataFrame({
            'date': date_range,
            'portfolio_value': portfolio_values,
            'cash_invested': cash_invested,
            'unrealized_gain': np.array(portfolio_values) - np.array(cash_invested)
        })
