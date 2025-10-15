"""
Simple Supervisor Agent following KISS, DRY, SOLID principles.
Pure routing with workflow delegation - no complex multi-step logic here.
"""

from typing import Optional
from agents.base_agent import BaseAgent, AgentResponse
from core.factory import get_workflow_engine
from core.routing_models import RoutingDecision, AgentType


class Supervisor(BaseAgent):
    """Simple supervisor - pure routing with workflow delegation following KISS principle."""

    def __init__(self):
        super().__init__("supervisor", temperature=0.1)
        self.workflow_engine = get_workflow_engine()
        self.structured_llm = self.llm.with_structured_output(RoutingDecision)

    def process(self, query: str, context: Optional[str] = None) -> AgentResponse:
        """Simple routing with comprehensive decision-making - KISS principle"""
        try:
            # Single LLM call for complete routing decision
            routing_decision = self._intelligent_route(query, context)

            print(
                f"🧠 Supervisor: {routing_decision.agent.value} (confidence: {routing_decision.confidence:.2f})"
            )
            print(f"🎯 Reasoning: {routing_decision.reasoning}")
            print(f"📊 Customer data needed: {routing_decision.needs_customer_data}")

            # Handle low confidence with direct response
            if routing_decision.confidence < 0.7:
                return self._generate_direct_response(query, routing_decision)

            # Clean delegation with comprehensive routing info
            return self.workflow_engine.execute_workflow(
                query, routing_decision.agent, routing_decision.needs_customer_data
            )

        except Exception as e:
            return self.handle_error(f"Supervisor error: {str(e)}")

    def _intelligent_route(
        self, query: str, context: Optional[str] = None
    ) -> RoutingDecision:
        """Single LLM call for comprehensive routing decision - DRY principle"""
        routing_prompt = self.prompts["intelligent_routing_prompt"]
        full_prompt = f"{routing_prompt}\n\nUser Query: {query}\n{f'Available context: {context}' if context else 'No additional context available.'}"
        return self.structured_llm.invoke(full_prompt)

    def _generate_direct_response(
        self, query: str, routing_decision: RoutingDecision
    ) -> AgentResponse:
        """Generate direct response for low confidence queries"""
        prompt_template = self.prompts["out_of_scope_response_prompt"]
        formatted_prompt = prompt_template.format(
            query=query, confidence=routing_decision.confidence
        )
        return AgentResponse(answer=self.llm.invoke(formatted_prompt).content)
