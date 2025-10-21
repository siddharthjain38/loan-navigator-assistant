"""
Simple SQL Agent for converting natural language queries to SQL and fetching loan data.
"""
from typing import Dict, Any, List, Union, Optional
import sqlite3
import json
import re

from .base_agent import BaseAgent, AgentResponse
from core.routing_models import SQLQueryResult
from core.constants import LOAN_DB_PATH
from pydantic import ValidationError

class SQLAgent(BaseAgent):
    """Simple agent for converting NLP to SQL and fetching loan data."""
    
    def __init__(self):
        """Initialize SQLAgent."""
        super().__init__('sql_agent', temperature=0.1)
        self.db_path = LOAN_DB_PATH
        
        # Create structured LLM for SQL query generation
        self.structured_llm = self.llm.with_structured_output(SQLQueryResult)
        
        # Known valid table and column names for validation
        self.valid_tables = ['loan_data']
        self.valid_columns = [
            'loan_id', 'customer_id', 'loan_amount', 'interest_rate', 
            'tenure_months', 'start_date', 'monthly_emi', 'amount_paid',
            'next_due_date', 'status', 'topup_eligible', 'prepayment_limit'
        ]
    
    def _generate_sql(self, query: str) -> SQLQueryResult:
        """Generate SQL query from natural language with structured output."""
        try:
            # Use self.prompts for structured query generation
            prompt = self.prompts['sql_prompt'].format(user_query=query)
            
            system_prompt = self.prompts['system_prompt']
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Get structured SQL query response
            sql_result: SQLQueryResult = self.structured_llm.invoke(messages)
            
            return sql_result
            
        except ValidationError as e:
            raise ValueError(f"Invalid SQL query structure: {e}")
    
    def _generate_sql_query(self, query: str) -> str:
        """Generate SQL query string from natural language (for testing)."""
        sql_result = self._generate_sql(query)
        return sql_result.query  # Correct attribute name
    
    def _execute_sql_with_params(self, sql: str, params: tuple) -> List[Dict[str, Any]]:
        """Execute parameterized SQL query (for testing)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = [dict(row) for row in rows]
            conn.close()
            
            return results
        except Exception as e:
            if conn:
                conn.close()
            raise e
        except Exception as e:
            raise RuntimeError(f"SQL generation failed: {e}")
    
    def _validate_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query before execution.
        Returns validation result with is_valid flag and issues list.
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }
        
        sql_upper = sql_query.upper().strip()
        
        # 1. Security check: only allow SELECT statements
        if not sql_upper.startswith(('SELECT', 'WITH')):
            validation_result["is_valid"] = False
            validation_result["issues"].append("Only SELECT and WITH queries are allowed")
            return validation_result
        
        # 2. Check for dangerous SQL operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'EXEC']
        for keyword in dangerous_keywords:
            if re.search(rf'\b{keyword}\b', sql_upper):
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Dangerous SQL keyword detected: {keyword}")
        
        # 3. Validate table names
        sql_lower = sql_query.lower()
        for table in self.valid_tables:
            if table in sql_lower:
                break
        else:
            # No valid table found
            validation_result["warnings"].append("Query might reference invalid table name")
        
        # 4. Check for basic SQL syntax patterns
        if 'SELECT' in sql_upper and 'FROM' not in sql_upper:
            validation_result["is_valid"] = False
            validation_result["issues"].append("SELECT query must have FROM clause")
        
        # 5. Check for SQL injection patterns
        injection_patterns = [r"';", r"--", r"/\*", r"\*/", r"xp_", r"sp_"]
        for pattern in injection_patterns:
            if re.search(pattern, sql_query):
                validation_result["warnings"].append(f"Potential SQL injection pattern detected: {pattern}")
        
        return validation_result
    
    def _execute_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results after validation."""
        # Validate SQL before execution
        validation = self._validate_sql(sql_query)
        
        if not validation["is_valid"]:
            raise ValueError(f"SQL validation failed: {', '.join(validation['issues'])}")
        
        # Log warnings if any
        if validation["warnings"]:
            print(f"⚠️ SQL Warnings: {', '.join(validation['warnings'])}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def _format_results_with_llm(self, query: str, results: List[Dict[str, Any]], sql_explanation: str) -> str:
        """Use LLM to format SQL results in a natural, user-friendly way."""
        if not results:
            return "No data found matching your criteria."
        
        # Prepare data summary
        data_json = json.dumps(results, indent=2, default=str)
        
        # Use prompt from YAML configuration
        format_prompt = self.prompts['format_prompt'].format(
            user_query=query,
            sql_explanation=sql_explanation,
            num_records=len(results),
            data_json=data_json
        )
        
        response = self.llm.invoke(format_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def _generate_clarification_prompt(self, original_query: str, sql_query: str, 
                                       error_type: str, context: Optional[str] = None) -> str:
        """Generate guided clarification prompt for failed queries."""
        clarification_prompts = {
            "zero_results": f"""
I couldn't find any data matching your query: "{original_query}"

This might be because:
• The customer ID doesn't exist in our database
• The date range specified has no records
• The status or filter criteria don't match any loans

Could you please:
1. Verify the customer ID is correct
2. Check if you meant a different date range
3. Confirm the loan status you're looking for (active, closed, etc.)

Available loan statuses: Active, Closed, Pending
""",
            "validation_failed": f"""
I had trouble understanding your query: "{original_query}"

Could you please rephrase it more specifically? For example:
• "Show me loan details for customer 1900"
• "What is the current EMI for loan ID 2001?"
• "List all active loans"
• "Show payment history for customer 1900"
""",
            "sql_error": f"""
I encountered an issue processing your query: "{original_query}"

Please try:
• Using specific customer IDs or loan IDs
• Simplifying your query
• Asking about one aspect at a time (e.g., EMI, loan amount, status)

Example queries that work well:
• "What is my current loan amount?" (if customer ID is known)
• "Show details for loan 2001"
• "List all loans for customer 1900"
"""
        }
        
        return clarification_prompts.get(error_type, clarification_prompts["sql_error"])
    
    def process(self, query: Union[str, Dict[str, Any]], retry_count: int = 0) -> AgentResponse:
        """Process NLP query to fetch loan data with fallback mechanisms."""
        try:
            # Handle new workflow engine input format
            if isinstance(query, dict) and "query" in query:
                nlp_query = query.get("query", "")
                customer_data = query.get("customer_data", [])  # Always a list now
                context = query.get("context", "")
                
                # Add customer context to query if available
                if customer_data:
                    customer_ids = [str(c.get('customer_id', '')) for c in customer_data if c.get('customer_id') is not None]
                    if customer_ids:
                        nlp_query = f"{nlp_query} [Customer IDs: {', '.join(customer_ids)}]"
            else:
                # Handle legacy input formats
                if isinstance(query, dict):
                    nlp_query = query.get("query", str(query))
                else:
                    nlp_query = query
                customer_data = []
                context = ""
            
            if not nlp_query:
                return self.handle_error("Empty query provided")
            
            # Generate structured SQL query
            sql_result = self._generate_sql(nlp_query)
            
            # Validate SQL before execution
            validation = self._validate_sql(sql_result.query)
            
            if not validation["is_valid"]:
                # SQL validation failed - signal supervisor for clarification
                print(f"❌ SQL Validation Failed: {validation['issues']}")
                clarification = self._generate_clarification_prompt(
                    nlp_query, sql_result.query, "validation_failed", context
                )
                
                if retry_count == 0:
                    # First attempt - signal supervisor
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "sql_fallback_flag": True,
                            "needs_clarification": True,
                            "reason": "SQL validation failed",
                            "validation_issues": validation["issues"],
                            "original_query": nlp_query,
                            "retry_count": retry_count
                        }
                    )
                else:
                    # Retry also failed - return clarification directly
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "is_fallback": True,
                            "retry_count": retry_count,
                            "reason": "SQL validation failed after retry"
                        }
                    )
            
            # Execute validated SQL
            try:
                results = self._execute_sql(sql_result.query)
            except Exception as exec_error:
                print(f"❌ SQL Execution Error: {exec_error}")
                clarification = self._generate_clarification_prompt(
                    nlp_query, sql_result.query, "sql_error", context
                )
                
                if retry_count == 0:
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "sql_fallback_flag": True,
                            "needs_clarification": True,
                            "reason": f"SQL execution error: {str(exec_error)}",
                            "original_query": nlp_query,
                            "retry_count": retry_count
                        }
                    )
                else:
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "is_fallback": True,
                            "retry_count": retry_count,
                            "reason": "SQL execution failed after retry"
                        }
                    )
            
            # Zero-result detection
            if len(results) == 0:
                print(f"⚠️ Zero Results Returned for query: {nlp_query}")
                clarification = self._generate_clarification_prompt(
                    nlp_query, sql_result.query, "zero_results", context
                )
                
                if retry_count == 0 and not customer_data:
                    # No customer data and no results - signal supervisor for clarification
                    return AgentResponse(
                        answer=clarification,
                        metadata={
                            "sql_fallback_flag": True,
                            "needs_clarification": True,
                            "reason": "Zero results returned, customer ID may be needed",
                            "original_query": nlp_query,
                            "retry_count": retry_count,
                            "sql_executed": sql_result.query
                        }
                    )
                else:
                    # Return friendly zero-results message
                    answer = f"No data found matching your criteria.\n\n{clarification}"
                    return AgentResponse(
                        answer=answer,
                        metadata={
                            "zero_results": True,
                            "retry_count": retry_count,
                            "sql_executed": sql_result.query
                        }
                    )
            
            # Format response using LLM for natural presentation
            answer = self._format_results_with_llm(
                query=nlp_query if isinstance(query, str) else query.get("query", ""),
                results=results,
                sql_explanation=sql_result.explanation
            )
            
            return AgentResponse(
                answer=answer,
                sources=[{
                    "type": "database",
                    "sql": sql_result.query,
                    "explanation": sql_result.explanation,
                    "confidence": sql_result.confidence,
                    "count": len(results)
                }],
                metadata={
                    "sql_confidence": sql_result.confidence,
                    "estimated_rows": sql_result.estimated_rows,
                    "actual_rows": len(results),
                    "retry_count": retry_count
                }
            )
            
        except Exception as e:
            print(f"❌ Unexpected error in SQL Agent: {e}")
            return self.handle_error(f"SQL error: {str(e)}")
    
    def get_data(self, query: str) -> List[Dict[str, Any]]:
        """Simple method to get raw data for other agents."""
        try:
            sql_query = self._generate_sql(query)
            return self._execute_sql(sql_query)
        except Exception:
            return []