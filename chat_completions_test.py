import os
from typing import List, Dict, Tuple

import anthropic
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


def _wait_for_any_key() -> None:
    """Pauses execution until user presses a key."""
    print("\nНажмите любую клавишу для выхода...")
    if os.name == "nt":
        import msvcrt

        msvcrt.getch()
    else:
        input()


def _describe_runtime_error(error: Exception) -> str:
    """Maps known errors to user-friendly Russian messages."""
    status_code = getattr(error, "status_code", None)
    error_text = str(error).lower()
    error_name = error.__class__.__name__.lower()

    if isinstance(error, RuntimeError) and "not initialized" in error_text:
        return "Ошибка инициализации чата: сессия не была создана."

    if status_code in {401, 403} or "auth" in error_name or "authentication" in error_name:
        return "Ошибка авторизации: проверьте AI_API_KEY и права доступа."

    if status_code in {404, 422} or "notfound" in error_name or "unprocessable" in error_name:
        return "Ошибка модели или запроса: проверьте имя модели и параметры запроса."

    if (
        status_code in {408, 429, 500, 502, 503, 504}
        or "timeout" in error_name
        or "connection" in error_name
        or "timed out" in error_text
        or "network" in error_text
    ):
        return "Сетевая ошибка или недоступность ProxyAPI. Проверьте интернет и попробуйте снова."

    return "Непредвиденная ошибка во время работы чата."


def _exit_with_error(error: Exception) -> None:
    """Prints friendly error, waits keypress, and exits."""
    print("\nОшибка:", _describe_runtime_error(error))
    print(f"Детали: {error}")
    _wait_for_any_key()
    raise SystemExit(1)


def _read_api_key() -> str:
    """Reads API key for proxy providers."""
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        raise ValueError("Set AI_API_KEY before running this script.")
    return api_key


def create_openai_client() -> OpenAI:
    """Creates OpenAI-compatible client for Chat Completions mode."""
    api_key = _read_api_key()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def create_anthropic_client() -> anthropic.Anthropic:
    """Creates Anthropic client for thinking mode."""
    api_key = _read_api_key()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.proxyapi.ru/anthropic")
    return anthropic.Anthropic(api_key=api_key, base_url=base_url)


class OpenAIChatSession:
    """Keeps OpenAI conversation context and sends it on every request."""

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


class AnthropicThinkingSession:
    """Dialog session with Claude thinking blocks and persistent context."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
        system_prompt: str = "",
        max_tokens: int = 1400,
        thinking_budget_tokens: int = 1024,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.thinking_budget_tokens = thinking_budget_tokens
        self.messages: List[Dict[str, str]] = []

    def ask(self, user_text: str) -> Tuple[str, List[str]]:
        """Sends full chat history and returns assistant text + thinking steps."""
        self.messages.append({"role": "user", "content": user_text})

        request_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget_tokens},
            "messages": self.messages,
        }
        if self.system_prompt:
            request_kwargs["system"] = self.system_prompt

        response = self.client.messages.create(**request_kwargs)

        thinking_steps: List[str] = []
        assistant_chunks: List[str] = []
        for block in response.content:
            if block.type == "thinking":
                step_text = (block.thinking or "").strip()
                if step_text:
                    thinking_steps.append(step_text)
            elif block.type == "text":
                text = (block.text or "").strip()
                if text:
                    assistant_chunks.append(text)

        assistant_text = "\n".join(assistant_chunks).strip()
        if not assistant_text:
            assistant_text = "[No final text returned.]"

        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text, thinking_steps

    def reset(self) -> None:
        """Clears conversation history."""
        self.messages = []


def select_chat_mode() -> str:
    """Lets user choose model mode at startup; default is thinking."""
    default_mode = os.getenv("CHAT_MODE", "thinking").strip().lower()
    if default_mode not in {"thinking", "normal"}:
        default_mode = "thinking"

    choice = input(
        f"Select mode [thinking/normal] (default: {default_mode}): "
    ).strip().lower()
    if not choice:
        return default_mode
    if choice in {"thinking", "t", "claude", "anthropic"}:
        return "thinking"
    if choice in {"normal", "n", "openai", "gpt"}:
        return "normal"
    print(f"Unknown mode '{choice}', using default: {default_mode}.")
    return default_mode


def interactive_chat() -> None:
    """CLI chat with selectable mode and context memory."""
    try:
        mode = select_chat_mode()

        openai_chat = None
        anthropic_chat = None

        if mode == "thinking":
            client = create_anthropic_client()
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            anthropic_chat = AnthropicThinkingSession(
                client=client,
                model=model,
                system_prompt=(
                    "Ты полезный и лаконичный ассистент. "
                    "Всегда отвечай пользователю на русском языке. "
                    "Все размышления и шаги thinking формулируй только на русском языке."
                ),
                max_tokens=1400,
                thinking_budget_tokens=1024,
            )
            print(f"Chat started in THINKING mode with model: {model}")
            print("Thinking steps will be printed before final answer.")
        else:
            client = create_openai_client()
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            openai_chat = OpenAIChatSession(
                client=client,
                model=model,
                system_prompt="You are a concise and helpful assistant.",
            )
            print(f"Chat started in NORMAL mode with model: {model}")
    except Exception as error:
        _exit_with_error(error)

    print("Type 'exit' to stop, 'reset' to clear history.")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Bye!")
            break
        if user_input.lower() == "reset":
            if mode == "thinking" and anthropic_chat is not None:
                anthropic_chat.reset()
            elif openai_chat is not None:
                openai_chat.reset()
            print("History reset.")
            continue

        try:
            if mode == "thinking" and anthropic_chat is not None:
                answer, thinking_steps = anthropic_chat.ask(user_input)
                if thinking_steps:
                    print("Thinking:")
                    for idx, step in enumerate(thinking_steps, start=1):
                        print(f"  Step {idx}: {step}")
            elif openai_chat is not None:
                answer = openai_chat.ask(user_input)
            else:
                raise RuntimeError("Chat session was not initialized.")
        except Exception as error:
            _exit_with_error(error)

        print(f"Assistant: {answer}")


if __name__ == "__main__":
    interactive_chat()
