"""
FIM Prompt Formatter for Lean4 proof infilling.

This module provides consistent prompt construction for Fill-in-the-Middle (FIM)
tasks, handling both standard cases (with suffix) and 100% masking cases (no suffix).

Requirements covered:
- 2.1: Construct prompts in format {prefix}[MISSING_BLOCK]\n{suffix}
- 2.2: Prepend system instruction for Lean 4 expert
- 2.3: Handle empty suffix case (100% masking)
- 2.4: Configurable prompt templates for experimentation
- 2.5: Preserve exact whitespace and newlines
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """
    Configurable prompt template for FIM tasks.
    
    Allows experimentation with different prompt formats while maintaining
    consistent structure across the codebase.
    
    Attributes:
        system_instruction: System prompt for standard FIM with suffix.
        hole_marker: Marker indicating where the model should fill in code.
        empty_suffix_instruction: System prompt for 100% masking (no suffix).
    """
    system_instruction: str = (
        "You are a Lean 4 expert. Complete the code at [MISSING_BLOCK]. "
        "Output ONLY the missing code."
    )
    hole_marker: str = "[MISSING_BLOCK]"
    empty_suffix_instruction: str = (
        "You are a Lean 4 expert. Complete the proof after the theorem statement. "
        "Output ONLY the proof tactics."
    )


class FIMPromptFormatter:
    """
    Formats FIM prompts for Lean4 proof infilling.
    
    This class handles the construction of prompts for Fill-in-the-Middle tasks,
    ensuring consistent formatting across all training and inference scenarios.
    
    Key features:
    - Preserves exact whitespace and newlines from input segments
    - Handles empty suffix case (100% masking) with adjusted instructions
    - Supports configurable templates for experimentation
    
    Example:
        >>> formatter = FIMPromptFormatter()
        >>> prompt = formatter.format(
        ...     prefix="theorem foo : 1 + 1 = 2 := by\\n  ",
        ...     suffix="\\n  rfl"
        ... )
        >>> print(prompt)  # Contains system instruction + prefix + [MISSING_BLOCK] + suffix
    """
    
    def __init__(self, template: Optional[PromptTemplate] = None):
        """
        Initialize the formatter with an optional custom template.
        
        Args:
            template: Custom PromptTemplate for experimentation.
                     If None, uses default template.
        """
        self.template = template or PromptTemplate()
    
    def format(self, prefix: str, suffix: str) -> str:
        """
        Construct FIM prompt from prefix and suffix.
        
        Preserves exact whitespace and handles empty suffix case.
        The returned prompt follows the format:
        - With suffix: {system_instruction} + {prefix}[MISSING_BLOCK]\\n{suffix}
        - Without suffix: {empty_suffix_instruction} + {prefix}
        
        Args:
            prefix: The code before the hole. Whitespace is preserved exactly.
            suffix: The code after the hole. Whitespace is preserved exactly.
                   If empty or whitespace-only, uses 100% masking format.
        
        Returns:
            Formatted prompt string ready for tokenization.
        """
        if not suffix.strip():
            # 100% masking case - no suffix provided
            return self._format_no_suffix(prefix)
        
        return self._format_with_suffix(prefix, suffix)
    
    def _format_with_suffix(self, prefix: str, suffix: str) -> str:
        """
        Standard FIM format with hole marker.
        
        Format: {prefix}[MISSING_BLOCK]\\n{suffix}
        
        Note: Whitespace in prefix and suffix is preserved exactly.
        The newline after the hole marker is intentional to separate
        the marker from the suffix content.
        
        Args:
            prefix: Code before the hole (whitespace preserved).
            suffix: Code after the hole (whitespace preserved).
        
        Returns:
            Formatted user content wrapped with system instruction.
        """
        # Requirement 2.1: format is {prefix}[MISSING_BLOCK]\n{suffix}
        # Requirement 2.5: preserve exact whitespace - no stripping
        user_content = f"{prefix}{self.template.hole_marker}\n{suffix}"
        return self._wrap_with_system(
            self.template.system_instruction,
            user_content
        )
    
    def _format_no_suffix(self, prefix: str) -> str:
        """
        Format for 100% masking (no suffix).
        
        When the entire proof body needs to be generated, we use a
        different system instruction that doesn't reference the hole marker.
        
        Args:
            prefix: Code before the hole (typically theorem statement).
        
        Returns:
            Formatted prompt with empty-suffix-specific instruction.
        """
        # Requirement 2.3: adjust instruction for empty suffix
        # Requirement 2.5: preserve exact whitespace in prefix
        return self._wrap_with_system(
            self.template.empty_suffix_instruction,
            prefix
        )
    
    def _wrap_with_system(self, system: str, user: str) -> str:
        """
        Wrap content with system instruction using chat template format.
        
        Returns raw text with special tokens; the tokenizer's chat template
        will handle final formatting for the specific model.
        
        Args:
            system: System instruction text.
            user: User content (prefix + hole marker + suffix).
        
        Returns:
            Formatted prompt with system and user sections.
        """
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
