"""
PolicyGuru agent for loan policy analysis using RAG with cosine similarity filtering.
"""

from typing import Dict, Any, TypedDict, List
import logging
import yaml
import numpy as np

from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from core.factory import get_embedding_llm
from core.routing_models import PolicyResponse
from pydantic import ValidationError


class GraphState(TypedDict):
    query: str
    filtered_docs: List[str]
    filtered_sources: List[Dict[str, Any]]
    answer: str | None
    no_docs: bool


from core.constants import (
    VECTOR_STORE_DIR,
    COLLECTION_NAME,
    COLLECTION_METADATA,
    EMBEDDING_MODEL,
    SEARCH_K,
    PROMPTS_DIR,
)
from .base_agent import BaseAgent, AgentResponse

# Configure logging
logger = logging.getLogger(__name__)


class PolicyGuru(BaseAgent):
    """PolicyGuru agent for loan policy analysis using RAG pattern."""

    def __init__(self):
        """Initialize PolicyGuru."""
        super().__init__("policy_guru", temperature=0)
        self.embedding = get_embedding_llm()
        self.similarity_threshold = 0.75

        # Create structured LLM for policy responses
        self.structured_llm = self.llm.with_structured_output(PolicyResponse)

        self._initialize_retriever()
        self._setup_processing_graph()

    def _initialize_retriever(self) -> None:
        """Initialize vector store retriever."""
        try:
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(VECTOR_STORE_DIR),
                embedding_function=self.embedding,
                collection_metadata=COLLECTION_METADATA,
            )

            self.retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": SEARCH_K}
            )

        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise

    def _setup_processing_graph(self) -> None:
        """Set up the LangGraph processing pipeline."""
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("retrieve", self._retrieve_docs)
        graph.add_node("generate", self._generate_answer)
        graph.add_node("handle_no_docs", self._handle_no_docs)

        # Set entry point
        graph.set_entry_point("retrieve")

        # Add conditional edges
        def route_after_retrieval(state: Dict[str, Any]) -> str:
            if state.get("no_docs"):
                return "handle_no_docs"
            else:
                return "generate"

        graph.add_conditional_edges("retrieve", route_after_retrieval)

        # Add terminal edges
        graph.add_edge("generate", END)
        graph.add_edge("handle_no_docs", END)

        self.graph = graph.compile()

    def _calculate_cosine_similarity(
        self, query_embedding: np.ndarray, doc_embedding: np.ndarray
    ) -> float:
        """Calculate cosine similarity between query and document embeddings."""
        # Normalize the vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norm = doc_embedding / np.linalg.norm(doc_embedding)

        # Calculate cosine similarity
        similarity = np.dot(query_norm, doc_norm)
        return float(similarity)

    def _retrieve_docs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and filter documents based on cosine similarity."""
        try:
            query = state["query"]

            # Get query embedding
            query_embedding = np.array(self.embedding.embed_query(query))

            # Retrieve documents
            docs = self.retriever.invoke(query)

            # Calculate similarity and filter
            filtered_docs = []
            filtered_sources = []

            for doc in docs:
                # Get document embedding
                doc_embedding = self.embedding.embed_query(doc.page_content)
                doc_embedding = np.array(doc_embedding)

                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(
                    query_embedding, doc_embedding
                )

                # Filter by threshold
                if similarity >= self.similarity_threshold:
                    filtered_docs.append(doc.page_content)
                    filtered_sources.append(doc.metadata)

            # Update state
            state["filtered_docs"] = filtered_docs
            state["filtered_sources"] = filtered_sources
            state["no_docs"] = len(filtered_docs) == 0

            return state

        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            raise

    def _generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using filtered documents."""
        try:
            filtered_docs = state["filtered_docs"]
            filtered_sources = state["filtered_sources"]
            query = state["query"]

            # Create context
            context = "\n\n---\n\n".join(filtered_docs)

            messages = [
                {"role": "system", "content": self.prompts["system_prompt"]},
                {
                    "role": "user",
                    "content": self.prompts["user_prompt"].format(
                        context=context, query=query
                    ),
                },
            ]

            try:
                # Get structured policy response
                policy_response: PolicyResponse = self.structured_llm.invoke(messages)
                state["answer"] = policy_response.answer
                state["policy_response"] = policy_response
            except (ValidationError, Exception) as e:
                # Fallback to regular response if structured output fails
                response = self.llm.invoke(messages)
                state["answer"] = response.content
                state["policy_response"] = None

            return state

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def _handle_no_docs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle case when no relevant documents are found."""
        msg = self.prompts["no_docs_message"]
        state["answer"] = msg
        return state

    def process(self, query_input) -> AgentResponse:
        """Process a user query about loan policies with flexible input handling."""
        try:
            # Handle new workflow engine input format
            if isinstance(query_input, dict) and "query" in query_input:
                query = query_input.get("query", "")
                customer_data = query_input.get(
                    "customer_data", []
                )  # Always a list now

                # Enhance query with customer context if available - let LLM handle multiple customers
                if customer_data:
                    enhanced_query = f"""
                    Policy Question: {query}
                    
                    Customer Data: {customer_data}
                    
                    Instructions:
                    - If single customer: Provide personalized policy information for their situation
                    - If multiple customers: Provide policy information relevant to all customers
                    - Consider loan amounts, interest rates, and status in policy guidance
                    - Handle missing customer data gracefully
                    
                    Please provide comprehensive policy information considering the customer context.
                    """
                    query = enhanced_query
            else:
                # Handle legacy string input
                query = str(query_input)
                customer_data = []

            state = {
                "query": query,
                "filtered_docs": [],
                "filtered_sources": [],
                "answer": None,
                "no_docs": False,
            }

            # Process through graph
            final_state = self.graph.invoke(state)

            # Calculate confidence based on filtered documents
            filtered_docs_count = len(final_state.get("filtered_docs", []))

            # Check if we need supervisor to enhance query
            if filtered_docs_count == 0:
                # Signal supervisor that query needs enhancement
                return AgentResponse(
                    answer="No relevant documents found. Query may need enhancement.",
                    metadata={
                        "needs_query_enhancement": True,
                        "reason": "No documents with sufficient similarity found",
                    },
                )

            # Use structured response if available
            policy_response = final_state.get("policy_response")
            if policy_response:
                return AgentResponse(
                    answer=policy_response.answer,
                    sources=final_state.get("filtered_sources", []),
                    metadata={
                        "policy_type": policy_response.policy_type,
                        "confidence": policy_response.confidence,
                        "relevant_sections": policy_response.relevant_sections,
                        "structured_output": True,
                    },
                )
            else:
                return AgentResponse(
                    answer=final_state.get(
                        "answer", self.prompts["default_error_response"]
                    ),
                    sources=final_state.get("filtered_sources", []),
                    metadata={"structured_output": False},
                )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self.handle_error(f"Query processing failed: {str(e)}")
