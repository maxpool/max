"""
Query Router module for Max Discord Bot.

This module classifies incoming queries to determine whether they need web research
or can be answered with standard model knowledge.
"""

from typing import Dict, Any, Tuple
import json
from logger import router_logger

class QueryRouter:
    """
    Routes queries to the appropriate LLM based on content analysis.
    Uses Perplexity for web research queries, Gemini for standard knowledge.
    """
    
    def __init__(self, 
                 perplexity_model="sonar-pro", 
                 gemini_model="gemini-2.0-flash-001",
                 classifier_model="claude-3-7-sonnet-20250219"):
        """
        Initialize the QueryRouter with model preferences.
        
        Args:
            perplexity_model: The Perplexity model to use for web research
            gemini_model: The Gemini model to use for standard responses
            classifier_model: The model to use for query classification
        """
        self.perplexity_model = perplexity_model
        self.gemini_model = gemini_model
        self.classifier_model = classifier_model
        router_logger.debug(f"QueryRouter initialized with models: perplexity={perplexity_model}, gemini={gemini_model}, classifier={classifier_model}")

    async def classify_with_llm(self, query: str, client) -> Dict:
        """
        Use an LLM to determine if a query requires web research and if it's AI-related.
        
        Args:
            query: The user's question or message
            client: The LLM client to use for classification
            
        Returns:
            Dictionary with classification results:
                - needs_web_research: Boolean indicating if web research is needed
                - is_ai_related: Boolean indicating if the query is related to AI/technology
        """
        router_logger.debug(f"Classifying query: '{query[:50]}...'")
        prompt = f"""
        You are Max, an AI assistant for the Maxpool Discord server focused on generative AI topics. You have been created by the Maxpool community.
        
        Analyze the following user query to determine:
        1. If it requires up-to-date information from the web
        2. If it is related to LLMs, AI, machine learning, coding, or technology topics
        
        Query: "{query}"
        
        For determining if web research is needed:
        - Does it request specific data, statistics, or factual information that changes frequently?
        - Does it ask about comparisons, prices, or reviews of AI tools/models that need current data?
        - Does it request technical information about recent AI software, products, or services?
        - Does it ask about recent research papers, model releases, or AI developments?
        - Does it involve recent product releases, updates, or version comparisons of AI tools?

        For determining if it's LLM/AI/technology related, it should match any of the following criteria:
        - Is it about LLMs, AI models, tools, techniques, or concepts?
        - Is it about specialized AI components like embeddings, retrievers, rerankers, RAG systems, vector databases?
        - Is it about model training, fine-tuning, inference, or optimization techniques?
        - Is it about programming, coding, or software development?
        
        Respond with a JSON object containing:
        1. "needs_web_research": boolean value (true/false)
        2. "is_ai_related": boolean value (true/false)
        """
        
        try:
            router_logger.debug("Invoking LLM for query classification")
            response = client.invoke(prompt)
            # Extract the content from the response object
            response_text = response.content
            
            try:
                result = json.loads(response_text)
                router_logger.debug(f"Classification result: web_research={result.get('needs_web_research', False)}, ai_related={result.get('is_ai_related', True)}")
                return {
                    "needs_web_research": result.get("needs_web_research", False),
                    "is_ai_related": result.get("is_ai_related", True)  # Default to True for backward compatibility
                }
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return valid JSON
                router_logger.warning("Failed to parse JSON response from LLM, using fallback classification")
                needs_web_research = "true" in response_text.lower() and "needs_web_research" in response_text.lower()
                is_ai_related = not ("false" in response_text.lower() and "is_ai_related" in response_text.lower())
                router_logger.debug(f"Fallback classification: web_research={needs_web_research}, ai_related={is_ai_related}")
                return {
                    "needs_web_research": needs_web_research,
                    "is_ai_related": is_ai_related
                }
        except Exception as e:
            router_logger.error(f"Error classifying query: {e}", exc_info=True)
            return {
                "needs_web_research": False,
                "is_ai_related": True  # Default to True in case of errors
            }

    async def route_query(self, query: str, llm_client) -> Tuple[str, str, Dict[str, Any], bool]:
        """
        Determine which LLM should handle the given query.
        
        Args:
            query: The user's question or message
            llm_client: The client to use for query classification
            
        Returns:
            Tuple containing:
                - provider name ("perplexity" or "google")
                - model name
                - additional parameters for the LLM
                - boolean indicating if the query is AI/technology related
        """
        router_logger.debug(f"Routing query: '{query[:50]}...'")
        classification = await self.classify_with_llm(query, llm_client)
        
        # Check if query is AI/technology related
        is_ai_related = classification.get("is_ai_related", True)
        
        # If non-AI query, return Google with default parameters and is_ai_related = False
        if not is_ai_related:
            router_logger.info("Query classified as non-AI related, routing to Google")
            return "google", self.gemini_model, {"temperature": 0.2}, False
            
        # For AI-related queries, route based on web research needs
        if classification.get("needs_web_research", False):
            router_logger.info("Query needs web research, routing to Perplexity")
            return "perplexity", self.perplexity_model, {"temperature": 0.0}, True
        else:
            router_logger.info("Query does not need web research, routing to Google")
            return "google", self.gemini_model, {"temperature": 0.2}, True
    
    async def should_ask_clarification(self, query: str, llm_client) -> bool:
        """
        Uses LLM to determine if the query is too vague and requires clarification.
        
        Args:
            query: The user's question or message
            llm_client: The client to use for classification
            
        Returns:
            Boolean indicating if clarification is needed
        """
        router_logger.debug(f"Checking if query needs clarification: '{query[:50]}...'")
        # Check for common greetings first - never ask for clarification for these
        query_lower = query.lower().strip()
        common_greetings = ["hey", "hello", "hi", "sup", "yo", "greetings", "hiya", "howdy", 
                            "hey max", "hello max", "hi max", "yo max", "howdy max",
                            "hey!", "hello!", "hi!", "sup!", "yo!", "howdy!", 
                            "good morning", "good afternoon", "good evening",
                            "morning", "afternoon", "evening", "what's up", "whats up",
                            "what up", "hey there", "hello there", "hi there",
                            "heya", "heyy", "hiii", "hiiii", "heyyy", "hellooo",
                            "wassup", "what is up", "what's happening", "whats happening"]
        
        # Check if the query exactly matches a greeting or starts with a greeting
        if query_lower in common_greetings:
            router_logger.debug("Query is a common greeting, no clarification needed")
            return False
            
        # Check if the query starts with a greeting
        for greeting in common_greetings:
            if query_lower.startswith(greeting + " "):
                router_logger.debug("Query starts with greeting, no clarification needed")
                return False
        
        # Check for short queries that might be conversational starters
        if len(query_lower.split()) <= 3:
            # Additional check for variations of greetings with emojis or punctuation
            for greeting in ["hey", "hello", "hi", "sup", "yo", "heya"]:
                if greeting in query_lower:
                    router_logger.debug("Short query contains greeting, no clarification needed")
                    return False
        
        prompt = f"""
        Determine if the following user query is too vague and requires clarification before providing a helpful response.
        
        Query: "{query}"
        
        Consider:
        - Is it extremely short (less than 3 words)?
        - Is it ambiguous with multiple possible interpretations?
        - Does it lack necessary context or specificity?
        - Is it overly broad or general?
        - IMPORTANT: Is it just a casual greeting like "hey", "hello", "hi", etc.? If so, do NOT ask for clarification.
        - IMPORTANT: If it's a simple greeting or conversation starter, do NOT ask for clarification.
        
        IMPORTANT: Respond with ONLY a JSON object and NOTHING ELSE. Your JSON response MUST be in exactly this format:
        {{
            "needs_clarification": true/false
        }}
        """
        
        try:
            router_logger.debug("Invoking LLM to determine if clarification is needed")
            response = llm_client.invoke(prompt)
            # Extract the content from the response object
            response_text = response.content
            
            try:
                result = json.loads(response_text)
                needs_clarification = result.get("needs_clarification", False)
                router_logger.debug(f"Clarification needed: {needs_clarification}")
                return needs_clarification
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return valid JSON
                router_logger.warning("Failed to parse JSON response from LLM, using fallback for clarification check")
                needs_clarification = "true" in response_text.lower() and "needs_clarification" in response_text.lower()
                router_logger.debug(f"Fallback clarification check: {needs_clarification}")
                return needs_clarification
        except Exception as e:
            router_logger.error(f"Error checking for clarification need: {e}", exc_info=True)
            return False

    async def detect_coreference(self, message: str, llm_client) -> bool:
        """
        Use an LLM to determine if a message contains coreferences (e.g., "this paper", "this question")
        that would benefit from additional context.
        
        Args:
            message: The user's message
            llm_client: The LLM client to use for detection
            
        Returns:
            Boolean indicating if the message contains coreferences that need context
        """
        router_logger.debug(f"Checking for coreferences in message: '{message[:50]}...'")
        # If the message is very short (just mentioning the bot), it's likely a reference
        if len(message.strip()) < 5:
            router_logger.debug("Message too short, treating as coreference")
            return True
            
        prompt = f"""
        Determine if the following message contains coreferences that would benefit from additional context.
        
        Message: "{message}"
        
        Consider the following:
        - Does it include demonstrative pronouns like "this", "that", "these", "those" without clear antecedents?
        - Does it refer to "this paper", "this article", "this topic", "this code", etc. without specifying which one?
        - Does it contain phrases like "thoughts?", "your take?", "what do you think?" without sufficient context?
        - Is it asking for an opinion, analysis, or evaluation of something that isn't fully specified?
        - Is the message very short but seems to expect knowledge of a previous context?
        - Does it use pronouns (it, they, them) without clear references?
        
        IMPORTANT: Respond with ONLY a JSON object and NOTHING ELSE. Your JSON response MUST be in exactly this format:
        {{
            "is_coreference": true/false
        }}
        """
        
        try:
            router_logger.debug("Invoking LLM for coreference detection")
            response = llm_client.invoke(prompt)
            response_text = response.content
            
            try:
                result = json.loads(response_text)
                is_coreference = result.get("is_coreference", False)
                router_logger.debug(f"Coreference detection result: {is_coreference}")
                return is_coreference
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return valid JSON
                router_logger.warning("Failed to parse JSON response from LLM, using fallback for coreference detection")
                is_coreference = "true" in response_text.lower() and "is_coreference" in response_text.lower()
                router_logger.debug(f"Fallback coreference detection: {is_coreference}")
                return is_coreference
        except Exception as e:
            router_logger.error(f"Error detecting coreference: {e}", exc_info=True)
            return False 