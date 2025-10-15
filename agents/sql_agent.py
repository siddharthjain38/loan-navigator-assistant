"""
Simple SQL Agent for converting natural language queries to SQL and fetching loan data.
"""
from typing import Dict, Any, List, Union
import sqlite3
import json
from pathlib import Path

from .base_agent import BaseAgent, AgentResponse
from core.routing_models import SQLQueryResult
from pydantic import ValidationError

class SQLAgent(BaseAgent):
    """Simple agent for converting NLP to SQL and fetching loan data."""
    
    def __init__(self):
        """Initialize SQLAgent."""
        super().__init__('sql_agent', temperature=0.1)
        self.db_path = Path(__file__).parent.parent / "database" / "loan_data" / "loan_data.db"
        
        # Create structured LLM for SQL query generation
        self.structured_llm = self.llm.with_structured_output(SQLQueryResult)
    
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
        except Exception as e:
            raise RuntimeError(f"SQL generation failed: {e}")
    
    def _execute_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results."""
        # Security: only allow SELECT statements  
        if not sql_query.strip().upper().startswith(('SELECT', 'WITH')):
            raise Exception("Only SELECT and WITH queries are allowed")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def process(self, query: Union[str, Dict[str, Any]]) -> AgentResponse:
        """Process NLP query to fetch loan data with flexible input handling."""
        try:
            # Handle new workflow engine input format
            if isinstance(query, dict) and "query" in query:
                nlp_query = query.get("query", "")
                customer_data = query.get("customer_data", [])  # Always a list now
                
                # Enhance query with customer context if available - let LLM handle multiple customers
                if customer_data:
                    enhanced_query = f"""
                    Data Query: {nlp_query}
                    
                    Existing Customer Data: {customer_data}
                    
                    Instructions:
                    - If single customer: Use their context for targeted queries
                    - If multiple customers: Generate queries that work for all customers
                    - Consider customer IDs, loan amounts, interest rates in query generation
                    - Handle missing customer data gracefully
                    
                    Generate appropriate SQL query considering the customer context.
                    """
                    nlp_query = enhanced_query
            else:
                # Handle legacy input formats
                if isinstance(query, dict):
                    nlp_query = query.get("query", str(query))
                else:
                    nlp_query = query
                customer_data = []
            
            if not nlp_query:
                return self.handle_error("Empty query provided")
            
            # Generate structured SQL query
            sql_result = self._generate_sql(nlp_query)
            results = self._execute_sql(sql_result.query)
            
            # Format response with structured information
            if not results:
                answer = f"No data found matching your criteria.\nQuery: {sql_result.explanation}"
            else:
                # Add customer context if available
                if customer_data:
                    if len(customer_data) == 1:
                        answer = f"Customer: {customer_data[0].get('customer_id')}\n"
                    elif len(customer_data) > 1:
                        customer_ids = [c.get('customer_id') for c in customer_data]
                        answer = f"Customers: {', '.join(customer_ids)}\n"
                    else:
                        answer = ""
                    answer += f"Found {len(results)} records.\n"
                else:
                    answer = f"Found {len(results)} records.\n"
                    
                answer += f"Query explanation: {sql_result.explanation}\n"
                answer += f"Confidence: {sql_result.confidence:.1%}\n\n"
                answer += json.dumps(results[:3], indent=2, default=str)
                if len(results) > 3:
                    answer += f"\n... and {len(results) - 3} more records."
            
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
                    "actual_rows": len(results)
                }
            )
            
        except Exception as e:
            return self.handle_error(f"SQL error: {str(e)}")
    
    def get_data(self, query: str) -> List[Dict[str, Any]]:
        """Simple method to get raw data for other agents."""
        try:
            sql_query = self._generate_sql(query)
            return self._execute_sql(sql_query)
        except Exception:
            return []