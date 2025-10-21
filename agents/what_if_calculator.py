from typing import Dict, Any, Union, List, Optional
from .base_agent import BaseAgent, AgentResponse
from core.routing_models import EMICalculation, EMIScenarios
from pydantic import ValidationError
import json


class WhatIfCalculator(BaseAgent):
    """Agent for calculating loan EMIs with different scenarios and fallback mechanisms."""

    def __init__(self):
        """Initialize WhatIfCalculator with all resources."""
        super().__init__("what_if_calculator", temperature=0.2)

        # Create structured LLM for EMI calculations
        self.structured_llm = self.llm.with_structured_output(EMIScenarios)

        # Validation constraints
        self.min_loan_amount = 1000  # Minimum loan amount
        self.max_loan_amount = 100000000  # 10 crore maximum
        self.min_tenure_months = 6  # 6 months minimum
        self.max_tenure_months = 360  # 30 years maximum
        self.min_interest_rate = 0.1  # 0.1% minimum
        self.max_interest_rate = 50  # 50% maximum (extremely high but possible)

    def _calculate_emi(
        self, loan_amount: float, interest_rate: float, tenure_months: int
    ) -> float:
        """
        Calculate EMI using the standard formula.

        Formula: EMI = P × r × (1 + r)^n / ((1 + r)^n - 1)
        Where:
            P = Principal loan amount
            r = Monthly interest rate (annual rate / 12 / 100)
            n = Number of months

        Returns:
            Monthly EMI amount
        """
        if interest_rate == 0:
            # For 0% interest, EMI is simply loan_amount / tenure
            return loan_amount / tenure_months

        # Calculate monthly interest rate
        monthly_rate = interest_rate / (12 * 100)

        # EMI formula
        emi = (
            loan_amount
            * monthly_rate
            * pow(1 + monthly_rate, tenure_months)
            / (pow(1 + monthly_rate, tenure_months) - 1)
        )

        return round(emi, 2)

    def _validate_inputs(
        self,
        loan_amount: float,
        interest_rate: float,
        tenure_months: Optional[int] = None,
        prepayment: Optional[float] = None,
        outstanding_balance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validate calculation inputs and return validation result.

        Returns:
            Dict with is_valid flag, errors list, and suggestions
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        # Validate loan amount
        if loan_amount <= 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Loan amount must be positive")
            validation_result["suggestions"].append(
                "Please provide a loan amount greater than ₹0"
            )
        elif loan_amount < self.min_loan_amount:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Loan amount too small (minimum: ₹{self.min_loan_amount:,})"
            )
        elif loan_amount > self.max_loan_amount:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Loan amount exceeds maximum (₹{self.max_loan_amount:,})"
            )
            validation_result["suggestions"].append(
                "Consider breaking into multiple loans or contact for special approval"
            )

        # Validate interest rate
        if interest_rate <= 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Interest rate must be positive")
        elif interest_rate < self.min_interest_rate:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Interest rate too low (minimum: {self.min_interest_rate}%)"
            )
        elif interest_rate > self.max_interest_rate:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Interest rate too high (maximum: {self.max_interest_rate}%)"
            )
            validation_result["warnings"].append(
                "This interest rate seems unusually high. Please verify."
            )

        # Validate tenure if provided
        if tenure_months is not None:
            if tenure_months <= 0:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Tenure must be positive")
                validation_result["suggestions"].append(
                    "Please specify a tenure between 6 months and 30 years"
                )
            elif tenure_months < self.min_tenure_months:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Tenure too short (minimum: {self.min_tenure_months} months)"
                )
            elif tenure_months > self.max_tenure_months:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Tenure too long (maximum: {self.max_tenure_months} months / 30 years)"
                )

        # Validate prepayment if provided
        if prepayment is not None:
            if prepayment < 0:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    "Prepayment amount cannot be negative"
                )
                validation_result["suggestions"].append(
                    "Please provide a prepayment amount of ₹0 or greater"
                )

            # Check if prepayment exceeds outstanding balance
            if outstanding_balance is not None and prepayment > outstanding_balance:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Prepayment (₹{prepayment:,.0f}) exceeds outstanding balance (₹{outstanding_balance:,.0f})"
                )
                validation_result["suggestions"].append(
                    f"Would you like to simulate a complete foreclosure with ₹{outstanding_balance:,.0f}?"
                )
            elif prepayment > loan_amount:
                # If no outstanding balance given, compare with loan amount
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"Prepayment (₹{prepayment:,.0f}) exceeds loan amount (₹{loan_amount:,.0f})"
                )
                validation_result["suggestions"].append(
                    "For complete loan closure, would you like to see foreclosure options?"
                )

        return validation_result

    def _generate_error_response(
        self, validation_result: Dict[str, Any], original_query: str = ""
    ) -> Dict[str, Any]:
        """Generate error JSON response with helpful suggestions."""
        error_response = {
            "status": "error",
            "message": "Input validation failed",
            "errors": validation_result["errors"],
            "warnings": validation_result.get("warnings", []),
            "suggestions": validation_result.get("suggestions", []),
            "original_query": original_query,
            "help": {
                "loan_amount_range": f"₹{self.min_loan_amount:,} to ₹{self.max_loan_amount:,}",
                "interest_rate_range": f"{self.min_interest_rate}% to {self.max_interest_rate}%",
                "tenure_range": f"{self.min_tenure_months} to {self.max_tenure_months} months ({self.min_tenure_months//12} to {self.max_tenure_months//12} years)",
                "examples": [
                    "Calculate EMI for ₹50,00,000 at 8.5% interest",
                    "What if I prepay ₹1,00,000 on my current loan?",
                    "Show EMI for ₹25 lakhs over 20 years at 9% interest",
                ],
            },
        }

        return error_response

    def process(
        self, query: Union[str, Dict[Any, Any]], retry_count: int = 0
    ) -> AgentResponse:
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

        # Extract parameters for validation if available
        loan_amount = None
        interest_rate = None
        outstanding_balance = None
        prepayment_amount = None

        # Get loan details from customer data if available
        if customer_data and len(customer_data) > 0:
            customer = customer_data[0]  # Use first customer for validation
            loan_amount = customer.get("loan_amount")
            interest_rate = customer.get("interest_rate")

            # Calculate outstanding balance if available
            if "loan_amount" in customer and "amount_paid" in customer:
                outstanding_balance = customer.get("loan_amount") - customer.get(
                    "amount_paid", 0
                )

        # Try to extract prepayment from query
        import re

        prepay_match = re.search(
            r"prepay(?:ment)?\s+(?:of\s+)?₹?(\d+(?:,\d+)*(?:\.\d+)?)",
            query,
            re.IGNORECASE,
        )
        if prepay_match:
            prepayment_str = prepay_match.group(1).replace(",", "")
            try:
                prepayment_amount = float(prepayment_str)
            except ValueError:
                pass

        # Validate inputs if we have them
        if loan_amount and interest_rate:
            validation = self._validate_inputs(
                loan_amount=loan_amount,
                interest_rate=interest_rate,
                prepayment=prepayment_amount,
                outstanding_balance=outstanding_balance,
            )

            if not validation["is_valid"]:
                print(
                    f"❌ What-If Calculator validation failed: {validation['errors']}"
                )

                # Generate error response
                error_json = self._generate_error_response(validation, query)

                # Format user-friendly error message
                error_message = "⚠️ **Input Validation Error**\n\n"
                error_message += "**Issues Found:**\n"
                for error in validation["errors"]:
                    error_message += f"• {error}\n"

                if validation.get("suggestions"):
                    error_message += "\n**Suggestions:**\n"
                    for suggestion in validation["suggestions"]:
                        error_message += f"• {suggestion}\n"

                error_message += f"\n**Valid Ranges:**\n"
                error_message += f"• Loan Amount: ₹{self.min_loan_amount:,} to ₹{self.max_loan_amount:,}\n"
                error_message += f"• Interest Rate: {self.min_interest_rate}% to {self.max_interest_rate}%\n"
                error_message += f"• Tenure: {self.min_tenure_months//12} to {self.max_tenure_months//12} years\n"

                return AgentResponse(
                    answer=error_message,
                    metadata={},
                )

        # Enhanced prompt - let LLM handle 0, 1, or multiple customers intelligently
        calculation_prompt = self.prompts["customer_data_calculation"].format(
            query=query,
            customer_data=customer_data
        )

        # Use structured LLM to get EMI scenarios - LLM handles everything!
        try:
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(calculation_prompt)
            return self._format_emi_response(emi_scenarios, customer_data, query)
        except ValidationError as e:
            return self.handle_error(f"EMI calculation validation error: {str(e)}")

    def _format_emi_response(
        self,
        emi_scenarios: EMIScenarios,
        customer_data: List[Dict] = None,
        original_query: str = "",
    ) -> AgentResponse:
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
                if (
                    "decrease" in original_query.lower()
                    or "reduce" in original_query.lower()
                ):
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
            },
        )

    def _process_legacy_dict(self, query: Dict[Any, Any]) -> AgentResponse:
        """Handle legacy dictionary formats for backward compatibility"""
        # Check if it's routing data from supervisor
        if "parameters" in query and "confidence" in query:
            parameters = query["parameters"]
            loan_amount = parameters.get("loan_amount")
            interest_rate = parameters.get("interest_rate")

            if not loan_amount or not interest_rate:
                return self.handle_missing_parameters(
                    parameters, query.get("confidence")
                )

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
        prompt = self.prompts["natural_language_calculation"].format(query=query)

        try:
            emi_scenarios: EMIScenarios = self.structured_llm.invoke(prompt)
            return self._format_emi_response(
                emi_scenarios, [], query
            )  # Pass empty list for no customer data
        except ValidationError as e:
            return self.handle_error(
                f"Natural language EMI calculation error: {str(e)}"
            )

    def _calculate_emi_scenarios(
        self, loan_amount: float, interest_rate: float
    ) -> AgentResponse:
        """Calculate EMI scenarios for given loan amount and interest rate"""
        system_prompt = self.prompts["system"]
        user_prompt = (
            f"Calculate EMI scenarios for a loan amount of ₹{loan_amount:,.0f} "
            f"with an interest rate of {interest_rate}% per annum. "
            f"Provide structured calculations for tenures of 5, 10, 15, 20, and 30 years."
        )

        # Call structured LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
                    "scenarios": [
                        scenario.dict() for scenario in emi_scenarios.scenarios
                    ],
                },
            )

        except ValidationError as e:
            return self.handle_error(f"Invalid EMI calculation structure: {e}")
        except Exception as e:
            return self.handle_error(f"EMI calculation failed: {e}")

    def handle_missing_parameters(
        self, parameters: Dict[str, Any], confidence: float
    ) -> AgentResponse:
        """Handle cases where required parameters are missing."""
        error_response = self.handle_error("Missing parameters")
        return error_response
