from typing import Dict, Any, Union, List
from .base_agent import BaseAgent, AgentResponse
from core.routing_models import EMICalculation, EMIScenarios
from pydantic import ValidationError

class WhatIfCalculator(BaseAgent):
    """Agent for calculating loan EMIs with different scenarios."""
    
    def __init__(self):
        """Initialize WhatIfCalculator with all resources."""
        super().__init__('what_if_calculator', temperature=0.2)
        
        # Create structured LLM for EMI calculations
        self.structured_llm = self.llm.with_structured_output(EMIScenarios)
            
    def process(self, query: Union[str, Dict[Any, Any]]) -> AgentResponse:
        """
        Process the loan calculation request with flexible input handling.
        
        Args:
            query: String query, dict with query/customer_data, or legacy dict formats
            
        Returns:
            AgentResponse: EMI calculations for different tenures
        """
        try:
            # Handle new workflow engine input format
            if isinstance(query, dict) and "query" in query:
                return self._process_workflow_input(query)
            
            # Handle legacy input formats
            elif isinstance(query, dict):
                return self._process_legacy_dict(query)
            
            # Handle direct string queries
            else:
                return self._process_string_query(query)
                
        except Exception as e:
            return self.handle_error(f"What-if calculation error: {str(e)}")

    def _process_workflow_input(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process input from WorkflowEngine with query and optional customer_data list"""
        query = input_data.get("query", "")
        customer_data = input_data.get("customer_data", [])  # Always a list now
        
        # Enhanced prompt - let LLM handle 0, 1, or multiple customers intelligently
        calculation_prompt = f"""
        Process this EMI calculation request:
        
        Query: {query}
        
        Customer Data: {customer_data}
        
        Instructions:
        - If customer_data is empty ([]): Process as general EMI calculation
        - If customer_data has 1 record: Process for that specific customer with their current loan details
        - If customer_data has multiple records: Process for all customers and provide comparison
        - Use current loan amounts and interest rates from customer data when available
        - If user wants to "decrease EMI", show options with longer tenures
        - If user wants to "increase loan", show scenarios with higher amounts
        - Handle any missing or invalid customer data gracefully
        - Format response clearly for the number of customers involved
        - Provide practical recommendations
        
        Calculate EMI scenarios for 5, 10, 15, 20, and 30 year tenures.
        Always be helpful and format the response clearly.
        """
        
        # Use structured LLM to get EMI scenarios - LLM handles everything!
        try:
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(calculation_prompt)
            return self._format_emi_response(emi_scenarios, customer_data, query)
        except ValidationError as e:
            return self.handle_error(f"EMI calculation validation error: {str(e)}")

    def _format_emi_response(self, emi_scenarios: EMIScenarios, customer_data: List[Dict] = None, original_query: str = "") -> AgentResponse:
        """Format EMI response with optional customer context - handles multiple customers"""
        answer = ""
        
        # Add customer context if available - LLM-friendly approach
        if customer_data:
            if len(customer_data) == 1:
                # Single customer
                customer = customer_data[0]
                answer += f"""Customer: {customer.get('customer_id')}
Current Loan: ₹{customer.get('loan_amount', 0):,.0f} at {customer.get('interest_rate', 0)}%
Current EMI: ₹{customer.get('monthly_emi', 0):,.0f}

"""
                # Add context about the request
                if "decrease" in original_query.lower() or "reduce" in original_query.lower():
                    answer += "🎯 Request: Decrease current EMI\n\n"
                elif "increase" in original_query.lower():
                    answer += "🎯 Request: Increase loan amount\n\n"
                
            elif len(customer_data) > 1:
                # Multiple customers
                answer += "👥 Multiple Customer Analysis:\n"
                for customer in customer_data:
                    answer += f"• {customer.get('customer_id')}: ₹{customer.get('loan_amount', 0):,.0f} at {customer.get('interest_rate', 0)}%\n"
                answer += "\n"
        
        # Add EMI scenarios
        answer += f"EMI Calculation Results:\n\n"
        
        for i, scenario in enumerate(emi_scenarios.scenarios, 1):
            years = scenario.tenure_months // 12
            answer += f"{i}. {years} Year(s) ({scenario.tenure_months} months):\n"
            answer += f"   • Monthly EMI: ₹{scenario.monthly_emi:,.0f}\n"
            answer += f"   • Total Interest: ₹{scenario.total_interest:,.0f}\n"
            answer += f"   • Total Amount: ₹{scenario.total_amount:,.0f}\n\n"
        
        answer += f"Summary: {emi_scenarios.summary}\n"
        answer += f"Recommendation: {emi_scenarios.recommendation}"
        
        return AgentResponse(
            answer=answer,
            metadata={
                "scenarios": [scenario.dict() for scenario in emi_scenarios.scenarios],
                "recommendation": emi_scenarios.recommendation,
                "customer_count": len(customer_data) if customer_data else 0,
                "customer_ids": [c.get('customer_id') for c in customer_data] if customer_data else []
            }
        )

    def _process_legacy_dict(self, query: Dict[Any, Any]) -> AgentResponse:
        """Handle legacy dictionary formats for backward compatibility"""
        # Check if it's routing data from supervisor
        if "parameters" in query and "confidence" in query:
            parameters = query["parameters"]
            loan_amount = parameters.get("loan_amount")
            interest_rate = parameters.get("interest_rate")
            
            if not loan_amount or not interest_rate:
                return self.handle_missing_parameters(parameters, query.get("confidence"))
                
        # Direct parameter format
        elif "loan_amount" in query and "interest_rate" in query:
            loan_amount = query.get("loan_amount")
            interest_rate = query.get("interest_rate")
        else:
            return self.handle_error("Invalid legacy input format")
        
        # Process with structured parameters
        return self._calculate_emi_scenarios(loan_amount, interest_rate)

    def _process_string_query(self, query: str) -> AgentResponse:
        """Process natural language string queries"""
        prompt = f"""
        Natural Language Query: {query}
        
        Extract loan amount and interest rate from this query.
        If not clearly specified, use reasonable defaults (₹10,00,000 and 10% interest).
        
        Calculate EMI scenarios for 5, 10, 15, 20, and 30 year tenures.
        Clearly state what parameters you used for the calculation.
        """
        
        try:
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(prompt)
            return self._format_emi_response(emi_scenarios, [], query)  # Pass empty list for no customer data
        except ValidationError as e:
            return self.handle_error(f"Natural language EMI calculation error: {str(e)}")

    def _calculate_emi_scenarios(self, loan_amount: float, interest_rate: float) -> AgentResponse:
        """Calculate EMI scenarios for given loan amount and interest rate"""
        system_prompt = self.prompts['system']
        user_prompt = (
            f"Calculate EMI scenarios for a loan amount of ₹{loan_amount:,.0f} "
            f"with an interest rate of {interest_rate}% per annum. "
            f"Provide structured calculations for tenures of 5, 10, 15, 20, and 30 years."
        )
        
        # Call structured LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Get structured EMI scenarios
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(messages)
            
            # Format structured response
            answer = f"EMI Calculation Results for ₹{loan_amount:,.0f} at {interest_rate}% annual interest:\n\n"
            
            for i, scenario in enumerate(emi_scenarios.scenarios, 1):
                years = scenario.tenure_months // 12
                answer += f"{i}. {years} Year(s) ({scenario.tenure_months} months):\n"
                answer += f"   • Monthly EMI: ₹{scenario.monthly_emi:,.0f}\n"
                answer += f"   • Total Interest: ₹{scenario.total_interest:,.0f}\n"
                answer += f"   • Total Amount: ₹{scenario.total_amount:,.0f}\n\n"
            
            answer += f"Summary: {emi_scenarios.summary}\n"
            answer += f"Recommendation: {emi_scenarios.recommendation}"
            
            return AgentResponse(
                answer=answer,
                metadata={
                    "scenarios": [scenario.dict() for scenario in emi_scenarios.scenarios],
                    "recommendation": emi_scenarios.recommendation,
                    "structured_output": True
                }
            )
            
        except ValidationError as e:
            return self.handle_error(f"Invalid EMI calculation structure: {e}")
        except Exception as e:
            return self.handle_error(f"EMI calculation failed: {e}")
    
    def handle_missing_parameters(self, parameters: Dict[str, Any], confidence: float) -> AgentResponse:
        """Handle cases where required parameters are missing."""
        error_response = self.handle_error("Missing parameters")
        return error_response