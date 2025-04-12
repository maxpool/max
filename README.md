# Max Discord Bot

A Discord bot for answering AI-related questions using various LLMs.

## Features

- Routes queries to appropriate LLMs based on content (Perplexity for web research, Gemini for general knowledge)
- Rewrites ambiguous queries using conversation context
- Maintains chat history for better contextual understanding
- Creates organized conversation threads
- Handles message references and replies

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Set up your environment variables (see `.env.example`)

## Running the Bot

Basic usage:
```
python bot.py
```

With increased logging verbosity:
```
python bot.py --verbose
```

Save logs to a file:
```
python bot.py --verbose --log-file=bot.log
```

## Command Line Options

- `-v, --verbose`: Enable detailed logging (DEBUG level)
- `--log-file PATH`: Save logs to the specified file

## Bot Commands

- `@Max <your question>`: Ask a question directly
- `!clear history`: Clear your chat history with the bot

## Development

The codebase is organized into several components:

- `bot.py`: Main Discord client
- `bot_handler.py`: Core logic for processing messages
- `query_router.py`: Routes queries to appropriate LLMs
- `query_rewriter.py`: Rewrites ambiguous queries
- `chat_history.py`: Manages conversation history
- `llm_handler.py`: Manages LLM clients
- `logger.py`: Logging utilities

## System Architecture

The bot is built with a modular design consisting of:

- **LLMHandler**: Manages different LLM providers (Perplexity, Gemini, etc.)
- **QueryRouter**: Routes queries to the appropriate LLM based on content analysis
- **ChatHistory**: Tracks conversation history for context-aware responses
- **BotHandler**: Orchestrates the entire flow of message processing

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   DISCORD_MAX_TOKEN=your_discord_token
   PERPLEXITY_API_KEY=your_perplexity_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```
4. Run the bot:
   ```
   python bot.py
   ```

## Usage

- Mention the bot (`@Max`) followed by your question
- Reply to the bot's message to continue a conversation
- Use `!clear history` to reset conversation context
- When asked for clarification, simply reply with the details

## Example Interactions

```
User: @Max What are the latest updates to the Gemini model?
Max: [Uses Perplexity for web research and provides current information]

User: @Max Can you explain how attention mechanisms work in transformers?
Max: [Uses Gemini to provide a concise, technical explanation]

User: @Max help
Max: I'd like to help with that. Could you provide a bit more detail about what specific task or concept you need help with?
```

## Customization

- Edit `prompts.py` to customize the bot's personality and response style
- Modify `query_router.py` to adjust how queries are classified
- Update `llm_handler.py` to add more LLM providers

## Requirements

- Python 3.8+
- Discord.py
- LangChain
- Perplexity API access
- Google Gemini API access 