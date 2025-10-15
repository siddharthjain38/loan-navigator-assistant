"""
Pydantic models for structured outputs across all agents.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict, Any, List
from enum import Enum


class AgentType(str, Enum):
    """Valid agent types for routing"""
    SQL_AGENT = "SQL_AGENT"
    WHAT_IF_CALCULATOR = "WHAT_IF_CALCULATOR" 
    POLICY_GURU = "POLICY_GURU"


class RoutingDecision(BaseModel):
    """Pydantic model for comprehensive routing decisions"""
    agent: AgentType = Field(..., description="The agent to route the query to")
    reasoning: str = Field(..., description="Brief explanation of why this agent was chosen")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
    )
    needs_customer_data: bool = Field(..., description="Whether this query needs customer-specific data")


class SQLQueryResult(BaseModel):
    """Structured output for SQL Agent queries"""
    query: str = Field(..., description="The SQL query generated")
    explanation: str = Field(..., description="Explanation of what the query does")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the query")
    estimated_rows: Optional[int] = Field(None, description="Estimated number of rows returned")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip().upper().startswith(('SELECT', 'WITH')):
            raise ValueError('Query must start with SELECT or WITH')
        return v.strip()


class EMICalculation(BaseModel):
    """Structured output for What-If Calculator"""
    loan_amount: float = Field(..., gt=0, description="Principal loan amount")
    interest_rate: float = Field(..., ge=0, le=100, description="Annual interest rate percentage")
    tenure_months: int = Field(..., gt=0, le=600, description="Loan tenure in months")
    monthly_emi: float = Field(..., gt=0, description="Calculated monthly EMI")
    total_interest: float = Field(..., ge=0, description="Total interest payable")
    total_amount: float = Field(..., gt=0, description="Total amount payable")

class EMIScenarios(BaseModel):
    """Multiple EMI scenarios for different tenures"""
    scenarios: List[EMICalculation] = Field(..., description="List of EMI calculations for different tenures")
    recommendation: str = Field(..., description="Brief recommendation based on calculations")
    summary: str = Field(..., description="Summary of all scenarios")
    
    @validator('scenarios')
    def validate_scenarios(cls, v):
        if len(v) < 1:
            raise ValueError('At least one scenario is required')
        return v


class PolicyResponse(BaseModel):
    """Simplified policy response model"""
    answer: str = Field(..., description="The policy answer or guidance")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the answer")
    
    @validator('answer')
    def validate_answer(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Answer must be at least 10 characters')
        return v.strip()