from __future__ import annotations

import dspy

"""
Sakana-style chat program for GSM8K parity.

This small DSPy module constructs messages that match the official
Sakana evaluation format, bypassing DSPy's ChatAdapter wrappers while
still using the tokenizer's chat template via the LM provider.

Behavior:
- system: empty content
- user: optional fixed ICL block + blank line +
        "Please answer the following question: {question}"

It returns an object with a single attribute `response` so it can be
used interchangeably with dspy.Predict-based programs in evaluation
loops that expect `.response`.
"""


class SakanaChatProgram(dspy.Module):
    def __init__(
        self,
        *,
        icl_block: str = "",
        include_icl: bool = True,
        question_prefix: str = "Please answer the following question:",
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        super().__init__()
        self.icl_block = icl_block.strip() if icl_block else ""
        self.include_icl = include_icl and bool(self.icl_block)
        self.question_prefix = question_prefix.strip()
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    def _strip_leading_question_prefix(self, text: str) -> str:
        t = (text or "").strip()
        # Many GSM8K exports start with our prefix already.
        if t.lower().startswith(self.question_prefix.lower()):
            return t[len(self.question_prefix) :].strip()
        return t

    def build_messages(self, prompt: str) -> list[dict]:
        """Construct the exact messages to send (without generating)."""
        question = self._strip_leading_question_prefix(prompt)

        parts = []
        if self.prefix:
            parts.append(self.prefix.rstrip())
        if self.include_icl and self.icl_block:
            parts.append(self.icl_block.rstrip())
        parts.append(f"{self.question_prefix} {question}".rstrip())
        if self.suffix:
            parts.append(self.suffix.rstrip())

        user_content = "\n\n".join([p for p in parts if p])
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
        ]
        return messages

    def forward(self, prompt: str):  # returns object with .response
        messages = self.build_messages(prompt)

        # Use the configured LM directly to preserve the chat template
        lm = dspy.settings.lm
        if lm is None:
            raise RuntimeError("dspy.settings.lm is not configured")

        outputs = lm(messages=messages)
        text = outputs[0] if isinstance(outputs, list) and outputs else str(outputs)

        class _Resp:
            def __init__(self, response: str):
                self.response = response

        return _Resp(text)
