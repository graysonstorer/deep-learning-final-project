from __future__ import annotations

from typing import Dict, List, Optional

from transformers import AutoTokenizer


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant answering questions about the Marvel Cinematic Universe."


def build_messages(
    question: str,
    context: str,
    answer: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def _manual_tinyllama_chat(
    question: str,
    context: str,
    answer: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    prompt = (
        f"<|system|>\n"
        f"{system_prompt}\n"
        f"</s>\n"
        f"<|user|>\n"
        f"Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"</s>\n"
        f"<|assistant|>\n"
    )
    if answer is None:
        return prompt
    return f"{prompt}{answer}</s>"


def render_chat_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    answer: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = build_messages(
        question=question,
        context=context,
        answer=answer,
        system_prompt=system_prompt,
    )
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)

    if callable(apply_chat_template) and chat_template:
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(answer is None),
            )
        except Exception:
            pass

    return _manual_tinyllama_chat(
        question=question,
        context=context,
        answer=answer,
        system_prompt=system_prompt,
    )


def prepare_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
