import time
from typing import List, Dict
import requests
from requests.exceptions import RequestException

from groq import Groq


class LM:
    """
    Unified Language Model interface.

    Automatically routes:
        - llama* models  -> Ollama (local)
        - others         -> Groq API
    """

    # ---------------- INIT ---------------- #

    def __init__(
        self,
        model: str,
        # ---- Groq config ----
        api_keys: List[str] = [
                            "apikey1", "apikey2"
                            ],
        calls_per_key: int = 3,

        # ---- Ollama config ----
        base_url: str = "http://localhost:11434",

        # ---- shared config ----
        retries: int = 1,
        temperature: float = 0.1,
        timeout: int = 120,
    ):

        self.model = model
        self.retries = retries
        self.temperature = temperature
        self.timeout = timeout

        # -------- Auto backend detection --------
        self.backend = self._detect_backend(model)

        # -------- Ollama setup --------
        self.base_url = base_url.rstrip("/")

        # -------- Groq setup --------
        self.api_keys = api_keys or []
        self.calls_per_key = calls_per_key
        self._key_index = 0
        self._call_count = 0

        if self.backend == "groq":
            if not self.api_keys:
                raise ValueError("Groq backend requires api_keys")

            self.client = Groq(api_key=self.api_keys[self._key_index])

    # ---------------- BACKEND ROUTER ---------------- #

    def _detect_backend(self, model: str) -> str:
        """
        Decide which backend to use.
        """
        name = model.lower()

        if name.startswith("llama"):
            return "ollama"

        return "groq"

    # ---------------- PUBLIC ---------------- #

    def get_model(self) -> str:
        return self.model

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        if self.backend == "ollama":
            return self._call_ollama(messages)
        else:
            return self._call_groq(messages)

    # =====================================================
    #                     GROQ
    # =====================================================

    def _rotate_key_if_needed(self):
        if self._call_count >= self.calls_per_key:
            self._call_count = 0
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            self.client = Groq(api_key=self.api_keys[self._key_index])

    def _call_groq(self, messages):
        for attempt in range(1, self.retries + 1):
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )

                response = (
                    chat_completion.choices[0].message.content or ""
                )

                self._call_count += 1
                self._rotate_key_if_needed()

                return response.strip()

            except RequestException as e:
                print(f"(Groq {attempt}/{self.retries}) Request error:", e)

            except Exception as e:
                print(f"(Groq {attempt}/{self.retries}) Unexpected:", e)

            if attempt < self.retries:
                time.sleep(1)

        raise Exception("Groq request failed after retries")

    # =====================================================
    #                     OLLAMA
    # =====================================================

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant:"
        return prompt

    def _call_ollama(self, messages):
        prompt = self._format_messages(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": self.temperature},
            "stream": False,
        }

        url = f"{self.base_url}/api/generate"

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                
                print(prompt)
                print(response.text)
                return data.get("response", "").strip()

            except requests.exceptions.RequestException as e:
                print(f"(Ollama {attempt}/{self.retries}) Request error:", e)

            except Exception as e:
                print(f"(Ollama {attempt}/{self.retries}) Unexpected:", e)

            if attempt < self.retries:
                time.sleep(1)

        raise Exception("Ollama request failed after retries")


# =====================================================
#                    EXAMPLE
# =====================================================

if __name__ == "__main__":

    # AUTO → OLLAMA
    lm_local = LM(model="llama3.2")
    print(lm_local([{"role": "user", "content": "Hello"}]))

    # AUTO → GROQ
    lm_cloud = LM(
        model="openai/gpt-oss-120b",
    )

    print(lm_cloud([{"role": "user", "content": "Hello"}]))