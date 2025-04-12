import os

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_perplexity import ChatPerplexity
from langchain_anthropic import ChatAnthropic       

class LLMHandler:
    """
    A class to handle different LLM providers (Mistral, Anthropic, and Google) using LangChain integrations.
    """

    def __init__(self):
        self.available_models = {

            "google": [
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-lite-001",
                "gemini-2.5-pro-exp-03-25",
                "gemini-2.5-pro-preview-03-25",
            ],
            "openai": [
                "gpt-4o-2024-11-20",
                "gpt-4o-mini",
                "o3-mini-2025-01-31",
                "gpt-4.5-preview-2025-02-27",
            ],
            "perplexity": [
                "sonar-pro",
                "sonar-small-online",
                "sonar-medium-online",
            ],
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-7-sonnet-20250219",
            ],
        }

        # Model name patterns for auto-detection of client
        self.model_patterns = {
            "o3": "openai",
            "gpt-4o": "openai",
            "gpt-4o-mini": "openai",
            "gemini": "google",
            "sonar": "perplexity",  
            "claude": "anthropic",
        }

    def _detect_provider(self, model_name: str) -> str:
        """
        Automatically detects the provider based on the model name.
        """
        for pattern, provider in self.model_patterns.items():
            if pattern in model_name.lower():
                return provider
        raise ValueError(f"Could not detect provider for model name: {model_name}")

    def _validate_model(self, provider: str, model_name: str) -> bool:
        """
        Validates if the requested model is available for the given provider.
        """
        if provider not in self.available_models:
            raise ValueError(
                f"Provider {provider} not supported. Available providers: {list(self.available_models.keys())}"
            )

        if model_name not in self.available_models[provider]:
            raise ValueError(
                f"Model {model_name} not available for {provider}. Available models: {self.available_models[provider]}"
            )

        return True

    def get_all_models(self):
        """
        Returns all available models from each provider as a list
        """
        all_models = []
        for provider, models in self.available_models.items():
            all_models.extend(models)
        return all_models

    def get_llm(
        self,
        model_name: str,
        provider: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 4000,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates and returns an LLM instance based on the provider and model name.

        Args:
            provider: The LLM provider ("anthropic", "mistral", or "google")
            model_name: The specific model to use
            temperature: Temperature parameter for generation (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            api_key: Optional API key (if not set in environment variables)
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            A LangChain chat model instance
        """
        # Auto-detect provider if not specified
        if provider is None:
            provider = self._detect_provider(model_name)

        self._validate_model(provider, model_name)

        # Set API key if provided
        if api_key:
            os.environ[f"{provider.upper()}_API_KEY"] = api_key

        model_params = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}

        # remove temperature if model name contains o1
        if ("o1" in model_name) or ("o3" in model_name):
            model_params.pop("temperature", None)

        # Remove None values
        model_params = {k: v for k, v in model_params.items() if v is not None}

        if provider == "google":
            print("Using Google")
            return ChatGoogleGenerativeAI(model=model_name, **model_params)

        elif provider == "openai":
            print("Using OpenAI")
            return ChatOpenAI(model=model_name, **model_params)

        elif provider == "perplexity":
            print("Using Perplexity")
            # , extra_body={"web_search_options": { "search_context_size": "medium" }}
            return ChatPerplexity(model=model_name, **model_params)

        elif provider == "anthropic":
            print("Using Anthropic")
            return ChatAnthropic(model=model_name, **model_params)
