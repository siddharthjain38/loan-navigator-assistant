"""
Simple workflow engine following KISS, DRY, SOLID principles.
ONE function handles all agent interactions.
"""

from typing import Optional, List, Dict, Any
import re
import sqlite3
from agents.base_agent import AgentResponse
from core.factory import get_sql_agent, get_what_if_calculator, get_policy_guru
from core.routing_models import AgentType
from core.constants import LOAN_DB_PATH


class WorkflowEngine:
    """Handles ALL agent interactions - ONE simple function"""

    def __init__(self):
        self.sql_agent = get_sql_agent()
        self.what_if_calculator = get_what_if_calculator()
        self.policy_guru = get_policy_guru()

    def execute_workflow(
        self, query: str, agent_type: AgentType, needs_customer_data: bool
    ) -> AgentResponse:
        """Single function - get customer data if needed, let agents handle the rest"""
        try:
            print(
                f"🔄 WorkflowEngine: {agent_type.value} (needs_data: {needs_customer_data})"
            )

            # Get customer data if needed - always returns list (0, 1, or multiple)
            customer_data = (
                self._get_customer_data(query) if needs_customer_data else []
            )

            # Simple input - let LLM figure out what to do
            agent_input = {
                "query": query,
                "customer_data": customer_data,  # Always a list: [], [single], [multiple]
            }

            print(f"📊 Found {len(customer_data)} customer(s)")

            # Simple delegation - same pattern for all agents
            if agent_type == AgentType.SQL_AGENT:
                return self.sql_agent.process(agent_input)
            elif agent_type == AgentType.WHAT_IF_CALCULATOR:
                return self.what_if_calculator.process(agent_input)
            elif agent_type == AgentType.POLICY_GURU:
                return self.policy_guru.process(agent_input)
            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")

        except Exception as e:
            return AgentResponse(answer=f"Workflow error: {str(e)}. Please try again.")

    def _get_customer_data(self, query: str) -> List[Dict[str, Any]]:
        """Extract ALL customer IDs and get their data - simple SQL IN query"""
        try:
            # Extract ALL customer IDs - supports CUST1234 or just 1234
            customer_id_pattern = r"(?:customer|cust|id)[_\s]*(?:id|is|:)?\s*([A-Z]*\d+)"
            matches = re.findall(customer_id_pattern, query, re.IGNORECASE)

            if not matches:
                return []  # Empty list instead of None

            # Convert to integers (database stores as INTEGER)
            customer_ids = []
            for match in set(matches):
                try:
                    # Extract numeric part only
                    numeric_id = ''.join(filter(str.isdigit, match))
                    if numeric_id:
                        customer_ids.append(int(numeric_id))
                except ValueError:
                    continue
            
            if not customer_ids:
                return []
                
            print(f"🔍 Customer IDs: {customer_ids}")

            # Single SQL query with IN clause - let database handle multiple IDs
            with sqlite3.connect(LOAN_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Single query for all customer IDs
                placeholders = ",".join("?" for _ in customer_ids)
                cursor.execute(
                    f"""
                    SELECT customer_id, loan_amount, interest_rate, tenure_months, 
                           monthly_emi, status, loan_id, next_due_date, amount_paid
                    FROM loan_data 
                    WHERE customer_id IN ({placeholders})
                """,
                    customer_ids,
                )

                # Convert all rows to list of dictionaries
                rows = cursor.fetchall()
                return [dict(row) for row in rows]  # Simple list comprehension

        except Exception as e:
            print(f"❌ Database error: {e}")
            return []  # Return empty list on error
