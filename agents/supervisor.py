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

    def process(self, query: str, context: Optional[str] = None, session_id: Optional[str] = None) -> AgentResponse:
        """Simple routing with comprehensive decision-making and retry logic - KISS principle"""
        try:
            # Single LLM call for complete routing decision
            routing_decision = self._intelligent_route(query, context)

            print(
                f"🧠 Supervisor: {routing_decision.agent.value} (confidence: {routing_decision.confidence:.2f})"
            )
            print(f"🎯 Reasoning: {routing_decision.reasoning}")
            print(f"📊 Customer data needed: {routing_decision.needs_customer_data}")

            # FALLBACK 1: Detect ambiguous queries (confidence < 0.5)
            if routing_decision.confidence < 0.5:
                print("⚠️ Ambiguous query detected - requesting clarification")
                return self._handle_ambiguous_query(query, routing_decision, context)
            
            # FALLBACK 2: Handle low confidence (0.5 <= confidence < 0.7)
            if routing_decision.confidence < 0.7:
                print("⚠️ Low confidence routing - checking for multi-domain query")
                multi_domain_result = self._detect_multi_domain_query(query, context)
                if multi_domain_result["is_multi_domain"]:
                    print(f"🔀 Multi-domain query detected: {multi_domain_result['domains']}")
                    return self._handle_multi_domain_query(query, multi_domain_result, context, session_id)
                return self._generate_direct_response(query, routing_decision)

            # Clean delegation with comprehensive routing info - pass context and session_id
            response = self.workflow_engine.execute_workflow(
                query, routing_decision.agent, routing_decision.needs_customer_data,
                context, session_id
            )
            
            # Check if Policy Guru needs query enhancement (fallback mechanism)
            if (routing_decision.agent == AgentType.POLICY_GURU and 
                response.metadata and 
                response.metadata.get("needs_query_enhancement")):
                
                print("🔄 Policy Guru needs query enhancement - retrying with enriched context")
                
                # Enhance query with additional context
                enhanced_query = self._enhance_query_with_context(query, context, session_id)
                
                # Retry with enhanced query
                retry_response = self.workflow_engine.execute_workflow(
                    enhanced_query, 
                    AgentType.POLICY_GURU, 
                    routing_decision.needs_customer_data,
                    context, 
                    session_id,
                    retry_count=1
                )
                
                print(f"✅ Retry complete - fallback: {retry_response.metadata.get('is_fallback', False)}")
                return retry_response
            
            # Check if SQL Agent needs clarification (fallback mechanism)
            if (routing_decision.agent == AgentType.SQL_AGENT and 
                response.metadata and 
                response.metadata.get("sql_fallback_flag")):
                
                print("🔄 SQL Agent needs clarification - attempting to provide guidance")
                
                # Check if we should retry or just return clarification
                retry_count = response.metadata.get("retry_count", 0)
                
                if retry_count == 0:
                    print("📝 Returning clarification prompt to user")
                    # First attempt - return clarification directly
                    # User needs to rephrase or provide more info
                    return response
                else:
                    # Retry already happened - return as-is
                    print("✅ Clarification provided after retry")
                    return response
            
            return response

        except Exception as e:
            return self.handle_error(f"Supervisor error: {str(e)}")
    
    def _enhance_query_with_context(self, query: str, context: Optional[str], session_id: Optional[str]) -> str:
        """Enhance query with additional context for retry attempts."""
        enhancement_prompt = f"""
        Original query needs more context to find relevant policies.
        
        Original Query: {query}
        
        Conversation Context: {context if context else "No conversation history"}
        
        Please reformulate this query to be more specific and include:
        - Loan type (if mentioned in context)
        - Customer profile details (if available)
        - Specific policy area being asked about
        - Any additional context that would help retrieve relevant policies
        
        Return ONLY the reformulated query, nothing else.
        """
        
        try:
            enhanced = self.llm.invoke(enhancement_prompt).content.strip()
            print(f"📝 Enhanced query: {enhanced[:100]}...")
            return enhanced
        except:
            # If enhancement fails, return original with context appended
            if context:
                return f"{query}\n\nAdditional Context: {context}"
            return query

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
    
    def _handle_ambiguous_query(
        self, query: str, routing_decision: RoutingDecision, context: Optional[str]
    ) -> AgentResponse:
        """Handle ambiguous queries by requesting clarification from user"""
        clarification_prompt = self.prompts.get("ambiguous_query_clarification", 
            """The query "{query}" is ambiguous. 
            
            I can help you with:
            1. **Loan Data Queries**: View your loan details, EMI, outstanding balance
            2. **EMI Calculations**: Calculate EMI for different loan amounts and tenures
            3. **Loan Policies**: Eligibility, documentation, prepayment rules
            
            Could you please clarify what you're looking for?""")
        
        formatted_prompt = clarification_prompt.format(
            query=query,
            reasoning=routing_decision.reasoning,
            confidence=routing_decision.confidence
        )
        
        response = self.llm.invoke(formatted_prompt).content
        
        return AgentResponse(
            answer=response,
            metadata={
                "requires_clarification": True,
                "confidence": routing_decision.confidence,
                "suggested_agent": routing_decision.agent.value,
                "clarification_type": "ambiguous_query"
            }
        )
    
    def _detect_multi_domain_query(self, query: str, context: Optional[str]) -> dict:
        """Detect if query spans multiple domains (SQL + Calculator + Policy)"""
        detection_prompt = self.prompts.get("multi_domain_detection",
            """Analyze if this query requires multiple agents:
            
            Query: {query}
            Context: {context}
            
            Agents available:
            - SQL_AGENT: Customer data lookup
            - WHAT_IF_CALCULATOR: EMI calculations
            - POLICY_GURU: Policy information
            
            Does this query need multiple agents? List which ones and why.
            
            Examples of multi-domain:
            - "Show my loan and calculate what if I prepay 1 lakh" (SQL + Calculator)
            - "What's my eligibility and current loan status" (Policy + SQL)
            - "Calculate EMI and tell me prepayment rules" (Calculator + Policy)
            
            Return JSON: {{"is_multi_domain": true/false, "domains": ["agent1", "agent2"], "reasoning": "..."}}
            """)
        
        formatted_prompt = detection_prompt.format(
            query=query,
            context=context if context else "No context"
        )
        
        try:
            response = self.llm.invoke(formatted_prompt).content
            # Try to parse JSON response
            import json
            if "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
                result = json.loads(json_str)
                return result
        except:
            pass
        
        # Default: not multi-domain
        return {"is_multi_domain": False, "domains": [], "reasoning": "Single domain query"}
    
    def _handle_multi_domain_query(
        self, query: str, multi_domain_result: dict, 
        context: Optional[str], session_id: Optional[str]
    ) -> AgentResponse:
        """Handle queries that span multiple domains by coordinating agents"""
        domains = multi_domain_result.get("domains", [])
        
        print(f"🔀 Processing multi-domain query across: {', '.join(domains)}")
        
        # Execute each domain in sequence and combine results
        results = []
        
        for domain in domains:
            try:
                agent_type = AgentType(domain)
                print(f"  → Executing {domain}...")
                
                response = self.workflow_engine.execute_workflow(
                    query, agent_type, True, context, session_id
                )
                
                results.append({
                    "agent": domain,
                    "answer": response.answer,
                    "metadata": response.metadata
                })
                
            except Exception as e:
                print(f"  ❌ Error executing {domain}: {str(e)}")
                continue
        
        # Combine results into coherent response
        combined_answer = self._combine_multi_domain_results(query, results)
        
        return AgentResponse(
            answer=combined_answer,
            metadata={
                "is_multi_domain": True,
                "domains_executed": domains,
                "individual_results": results
            }
        )
    
    def _combine_multi_domain_results(self, query: str, results: list) -> str:
        """Combine multiple agent results into a coherent response"""
        combination_prompt = self.prompts.get("multi_domain_combination",
            """User asked: {query}
            
            We gathered information from multiple systems:
            
            {results}
            
            Please combine these results into a single, coherent response that:
            1. Directly answers the user's question
            2. Integrates information from all sources
            3. Is clear and well-organized
            4. Maintains professional tone
            
            Do not mention "multiple agents" or technical details - just provide the integrated answer.
            """)
        
        results_text = "\n\n".join([
            f"**{r['agent']}**:\n{r['answer']}"
            for r in results
        ])
        
        formatted_prompt = combination_prompt.format(
            query=query,
            results=results_text
        )
        
        try:
            combined = self.llm.invoke(formatted_prompt).content
            return combined
        except:
            # Fallback: just concatenate results
            return "\n\n---\n\n".join([r['answer'] for r in results])
