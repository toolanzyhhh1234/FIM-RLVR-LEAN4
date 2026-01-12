"""
FIM Prompt Formatter for Lean4 proof infilling.

This module provides consistent prompt construction for Fill-in-the-Middle (FIM)
tasks, matching the format used in the Unsloth training pipeline.

Requirements covered:
- 2.1: Construct prompts in format {prefix}[MISSING_BLOCK]\n{suffix}
- 2.2: Prepend system instruction for Lean 4 expert with examples
- 2.3: Handle empty suffix case (100% masking / full solution)
- 2.4: Configurable prompt templates for experimentation
- 2.5: Preserve exact whitespace and newlines
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# Tag names matching the Unsloth pipeline
FIM_CODE_TAG = "FIM_CODE"
FULL_CODE_TAG = "FULL_CODE"


@dataclass
class PromptTemplate:
    """
    Configurable prompt template for FIM tasks.
    
    Matches the format used in train_gspo_fim_qwen3-vl-8b.py for consistency
    between Unsloth and Tinker training pipelines.
    
    Attributes:
        fim_code_tag: Tag for FIM completions (default: FIM_CODE).
        full_code_tag: Tag for full solution completions (default: FULL_CODE).
        hole_marker: Marker indicating where the model should fill in code.
        include_think_tags: Whether to instruct model to use [THINK] tags.
    """
    fim_code_tag: str = FIM_CODE_TAG
    full_code_tag: str = FULL_CODE_TAG
    hole_marker: str = "[MISSING_BLOCK]"
    include_think_tags: bool = True


def build_system_prompt(template: PromptTemplate) -> str:
    """
    Build the system prompt with examples, matching the Unsloth pipeline.
    
    This is the exact format from train_gspo_fim_qwen3-vl-8b.py to ensure
    consistent model behavior between training pipelines.
    """
    return (
        "You are a Lean 4 expert. Solve the task strictly following this format:\n"
        "1) First write your reasoning inside [THINK]...[/THINK].\n"
        f"2) Then output ONLY the code inside <{template.fim_code_tag}>...</{template.fim_code_tag}> "
        f"for fill-in-the-middle tasks, or <{template.full_code_tag}>...</{template.full_code_tag}> "
        "for full solutions.\n"
        "3) Do NOT include markdown fences or extra text outside the tags.\n"
        "4) The tagged code must be valid Lean 4.\n"
        "5) End FIM snippets with a separator (newline or `;`) so the next command parses correctly.\n"
        "If the user includes [FULL-SOLUTION-REQUIRED], output a full solution in <FULL_CODE>.\n\n"
        "[USER]\n"
        "theorem simple_add (n : ℕ) : 0 + n = n := by\n"
        "  [MISSING_BLOCK]\n\n"
        "[ASSISTANT]\n"
        "[THINK]\n"
        "The definition of addition recurses on the second argument, so 0+n requires induction or a lemma. \n"
        "`simp` uses Nat.zero_add to solve this.\n"
        "[/THINK]\n"
        f"<{template.fim_code_tag}>\n"
        "  simp\n"
        f"</{template.fim_code_tag}>\n\n"
        "Example (full):\n\n"
        "[USER]\n"
        "theorem add_zero_triv (n : ℕ) : n + 0 = n :=\n\n"
        "[ASSISTANT]\n"
        "[THINK]\n"
        "Addition is defined by recursion on the second argument. \n"
        "Therefore, `n + 0 = n` is true by definition (reflexivity).\n"
        "[/THINK]\n"
        f"<{template.full_code_tag}>\n"
        "by\n"
        "  rfl\n"
        f"</{template.full_code_tag}>"
    )


class FIMPromptFormatter:
    """
    Formats FIM prompts for Lean4 proof infilling.
    
    This class handles the construction of prompts for Fill-in-the-Middle tasks,
    matching the format used in the Unsloth training pipeline for consistency.
    
    Key features:
    - Uses detailed system prompt with examples (matching Unsloth)
    - Preserves exact whitespace and newlines from input segments
    - Handles empty suffix case (100% masking) with [FULL-SOLUTION-REQUIRED]
    - Supports configurable templates for experimentation
    
    Example:
        >>> formatter = FIMPromptFormatter()
        >>> prompt = formatter.format(
        ...     prefix="theorem foo : 1 + 1 = 2 := by\\n  ",
        ...     suffix="\\n  rfl"
        ... )
        >>> print(prompt)  # Contains system instruction + prefix + [MISSING_BLOCK] + suffix
    """
    
    def __init__(
        self, 
        template: Optional[PromptTemplate] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize the formatter with an optional custom template.
        
        Args:
            template: Custom PromptTemplate for experimentation.
                     If None, uses default template.
            tokenizer: Optional tokenizer with apply_chat_template method.
                      If provided, uses the tokenizer's chat template.
                      If None, uses a simple text format.
        """
        self.template = template or PromptTemplate()
        self.tokenizer = tokenizer
        self._system_prompt = build_system_prompt(self.template)
    
    def format(self, prefix: str, suffix: str) -> str:
        """
        Construct FIM prompt from prefix and suffix.
        
        Preserves exact whitespace and handles empty suffix case.
        The returned prompt follows the format from train_gspo_fim_qwen3-vl-8b.py.
        
        Args:
            prefix: The code before the hole. Whitespace is preserved exactly.
            suffix: The code after the hole. Whitespace is preserved exactly.
                   If empty or whitespace-only, uses full solution format.
        
        Returns:
            Formatted prompt string ready for tokenization.
        """
        if not suffix.strip():
            # 100% masking case - full solution required
            return self._format_full_solution(prefix)
        
        return self._format_fim(prefix, suffix)

    def normalize_boundaries(self, prefix: str, suffix: str) -> tuple[str, str]:
        """
        Ensure separators around the hole so inserted code never concatenates
        directly with prefix or suffix (FM-1 mitigation).
        """
        # IMPORTANT: do not force a newline after `prefix`.
        # Many Lean proofs end the prefix with indentation spaces (e.g. `:= by\n  `).
        # Adding an extra newline would drop indentation and can break parsing.
        safe_prefix = prefix
        # We do ensure that the suffix is separated from the hole marker.
        safe_suffix = suffix if suffix.startswith("\n") else "\n" + suffix
        return safe_prefix, safe_suffix

    def _format_fim(self, prefix: str, suffix: str) -> str:
        """
        Standard FIM format with hole marker.
        
        Format: {prefix}[MISSING_BLOCK]\\n{suffix}
        
        This matches the format in train_gspo_fim_qwen3-vl-8b.py.
        """
        # Ensure explicit separators around the hole (FM-1 Option A1).
        safe_prefix, safe_suffix = self.normalize_boundaries(prefix, suffix)
        # User content: prefix + [MISSING_BLOCK] + newline + suffix
        user_content = f"{safe_prefix}{self.template.hole_marker}{safe_suffix}"
        return self._apply_chat_template(user_content)
    
    def _format_full_solution(self, prefix: str) -> str:
        """
        Format for 100% masking (full solution required).
        
        When the entire proof body needs to be generated, we mark it
        with [FULL-SOLUTION-REQUIRED] so the model knows to output
        a complete solution in <FULL_CODE> tags.
        """
        # For full solution, just provide the theorem statement
        user_content = f"{prefix}\n[FULL-SOLUTION-REQUIRED]"
        return self._apply_chat_template(user_content)
    
    def _apply_chat_template(self, user_content: str) -> str:
        """
        Apply chat template to format the prompt.
        
        If a tokenizer with apply_chat_template is available, uses it.
        Otherwise, falls back to a simple text format.
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        if self.tokenizer is not None and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fall back to simple format if chat template fails
                pass
        
        # Simple text format fallback
        return (
            f"<|system|>\n{self._system_prompt}\n"
            f"<|user|>\n{user_content}\n"
            f"<|assistant|>\n"
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt text."""
        return self._system_prompt
    
    def extract_code_from_response(self, response: str, task_type: str = "fim") -> Optional[str]:
        """
        Extract code from model response based on task type.
        
        Looks for code inside the appropriate tags:
        - FIM tasks: <FIM_CODE>...</FIM_CODE>
        - Full solution tasks: <FULL_CODE>...</FULL_CODE>
        
        Uses the LAST occurrence of the tags since models often mention
        the tags in their reasoning before outputting the actual code.
        
        Args:
            response: The model's response text.
            task_type: Either "fim" or "full".
        
        Returns:
            Extracted code string, or None if tags not found.
        
        Note (GPT-OSS / Harmony format):
            For models using OpenAI's Harmony response format (e.g., gpt-oss-120b),
            responses contain multiple "channels":
            - analysis: Chain-of-thought reasoning (may mention tags but not final answer)
            - final: The actual response intended for the user (contains real code tags)
            
            The channel markers look like: <|channel|>analysis<|message|>...<|end|>
            
            TODO: For better extraction quality metrics, we could check whether the
            extracted <FIM_CODE> tags appear in the 'analysis' channel vs 'final' channel.
            Tags in 'analysis' are likely just the model discussing the format, while
            tags in 'final' contain the actual answer. This could help diagnose:
            - Model truncation (never reached 'final' channel)
            - Extraction from wrong channel (grabbed analysis instead of final)
            
            Implementation hint: Look for <|channel|>final<|message|> marker and only
            search for code tags after that position. If no 'final' channel exists,
            the model likely ran out of tokens during reasoning.
        """
        if task_type == "fim":
            tag = self.template.fim_code_tag
        else:
            tag = self.template.full_code_tag
        
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        def _extract_from(text: str) -> Optional[str]:
            start_idx = text.rfind(start_tag)
            if start_idx == -1:
                return None
            start_idx += len(start_tag)
            end_idx = text.find(end_tag, start_idx)
            if end_idx == -1:
                # IMPORTANT: do not `.strip()` here; it can remove leading indentation
                # that is syntactically meaningful inside Lean tactic blocks.
                extracted = text[start_idx:].strip("\n")
            else:
                extracted = text[start_idx:end_idx]
            return extracted.strip("\n")

        # Prefer extracting from the Harmony "final" channel to avoid grabbing tags mentioned in analysis.
        final_marker = "<|channel|>final<|message|>"
        final_idx = response.find(final_marker)
        if final_idx != -1:
            extracted = _extract_from(response[final_idx + len(final_marker):])
            if extracted is not None:
                return extracted

        # Fallback: search entire response (can recover cases where the model never reached final).
        return _extract_from(response)
    
    def strip_markdown_fences(self, text: str) -> str:
        """
        Remove markdown code fences from text.
        
        Models sometimes wrap code in ```lean ... ``` even when instructed not to.
        """
        if not text:
            return text
        lines = text.splitlines()
        cleaned = [line for line in lines if not line.strip().startswith("```")]
        # Preserve leading indentation: many holes are inside indented tactic blocks.
        # Only trim extra leading/trailing newlines introduced by tag extraction.
        return "\n".join(cleaned).strip("\n")
