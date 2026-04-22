import os
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


def create_client() -> OpenAI:
    """Creates an OpenAI client using OPENAI_API_KEY environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before running this script.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


class ChatSession:
    """Keeps conversation context and sends it on every request."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", system_prompt: str = ""):
        self.client = client
        self.model = model
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def ask(self, user_text: str) -> str:
        """Sends user message with full history and stores assistant response."""
        self.messages.append({"role": "user", "content": user_text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
        )

        assistant_text = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def reset(self, keep_system_prompt: bool = True) -> None:
        """Clears history. Optional: keep system prompt if present."""
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the current in-memory chat history."""
        return self.messages


def interactive_chat() -> None:
    """Tiny CLI test: continuous dialog mode with context memory."""
    client = create_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    chat = ChatSession(
        client=client,
        model=model,
        system_prompt="You are a concise and helpful assistant.",
    )

    print("Chat started. Type 'exit' to stop, 'reset' to clear history.")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Bye!")
            break
        if user_input.lower() == "reset":
            chat.reset()
            print("History reset.")
            continue

        answer = chat.ask(user_input)
        print(f"Assistant: {answer}")


if __name__ == "__main__":
    interactive_chat()
