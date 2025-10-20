"""
Simple SQL Agent for converting natural language queries to SQL and fetching loan data.
"""
from typing import Dict, Any, List, Union
import sqlite3
import json

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
    
    def process(self, query: Union[str, Dict[str, Any]]) -> AgentResponse:
        """Process NLP query to fetch loan data with flexible input handling."""
        try:
            # Handle new workflow engine input format
            if isinstance(query, dict) and "query" in query:
                nlp_query = query.get("query", "")
                customer_data = query.get("customer_data", [])  # Always a list now
                
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
            
            if not nlp_query:
                return self.handle_error("Empty query provided")
            
            # Generate structured SQL query
            sql_result = self._generate_sql(nlp_query)
            results = self._execute_sql(sql_result.query)
            
            # Format response using LLM for natural presentation
            if not results:
                answer = f"No data found matching your criteria.\n\nQuery explanation: {sql_result.explanation}"
            else:
                # Use LLM to format the complete results
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