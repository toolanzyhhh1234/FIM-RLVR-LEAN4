from tinker_integration.prompt_formatter import FIMPromptFormatter, PromptTemplate


def test_extract_preserves_indentation_from_harmony_final() -> None:
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    response = (
        "<|channel|>analysis<|message|>ignore <FIM_CODE>bad</FIM_CODE><|end|>"
        "<|channel|>final<|message|><FIM_CODE>\n  exact h\n</FIM_CODE>"
    )
    extracted = formatter.extract_code_from_response(response, task_type="fim")
    assert extracted is not None
    assert extracted.splitlines()[0].startswith("  ")
    assert extracted.strip("\n") == "  exact h"


def test_extract_prefers_harmony_final_over_analysis() -> None:
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    response = (
        "<|channel|>analysis<|message|><FIM_CODE>\n  simp\n</FIM_CODE><|end|>"
        "<|channel|>final<|message|><FIM_CODE>\n  exact h\n</FIM_CODE>"
    )
    extracted = formatter.extract_code_from_response(response, task_type="fim")
    assert extracted == "  exact h"


def test_extract_missing_end_tag_does_not_left_strip() -> None:
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    response = "<|channel|>final<|message|><FIM_CODE>\n  exact h\n"  # no </FIM_CODE>
    extracted = formatter.extract_code_from_response(response, task_type="fim")
    assert extracted is not None
    assert extracted.startswith("  ")


def test_normalize_boundaries_option_a() -> None:
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    prefix = "theorem foo : True := by\n  "
    suffix = "  · trivial"
    safe_prefix, safe_suffix = formatter.normalize_boundaries(prefix, suffix)
    assert safe_prefix == prefix
    assert safe_suffix.startswith("\n")
    assert safe_suffix == "\n  · trivial"


def test_stitching_never_glues_suffix_onto_last_token() -> None:
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    prefix = "theorem foo : True := by\n  "
    extracted = "trivial"
    suffix = "  · trivial"
    _, safe_suffix = formatter.normalize_boundaries(prefix, suffix)
    full_code = prefix + extracted + safe_suffix
    assert "trivial\n  ·" in full_code
    assert "trivial  ·" not in full_code


def test_strip_markdown_fences_preserves_indentation() -> None:
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    fenced = "```lean\n  simp\n```\n"
    cleaned = formatter.strip_markdown_fences(fenced)
    assert cleaned == "  simp"


def test_strip_harmony_tokens_removes_return_marker() -> None:
    """Test that <|return|> and subsequent content is stripped."""
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    code_with_return = "  simp\n  exact h\n<|return|>\nextra junk"
    cleaned = formatter.strip_harmony_tokens(code_with_return)
    assert "<|return|>" not in cleaned
    assert "extra junk" not in cleaned
    assert "  simp" in cleaned
    assert "  exact h" in cleaned


def test_strip_harmony_tokens_removes_all_markers() -> None:
    """Test that all Harmony markers like <|end|>, <|channel|>, etc. are removed."""
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    # Realistic case: code with inline Harmony markers (the regex removes the markers only)
    code_with_markers = "  rfl<|end|>"
    cleaned = formatter.strip_harmony_tokens(code_with_markers)
    assert "<|end|>" not in cleaned
    assert cleaned == "  rfl"


def test_strip_harmony_tokens_preserves_indentation() -> None:
    """Test that indentation is preserved after stripping Harmony tokens."""
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    indented_code = "    simp [pow_two]<|return|>"
    cleaned = formatter.strip_harmony_tokens(indented_code)
    assert cleaned.startswith("    ")
    assert cleaned == "    simp [pow_two]"


def test_strip_harmony_tokens_handles_empty_input() -> None:
    """Test edge case of empty string."""
    formatter = FIMPromptFormatter(template=PromptTemplate(include_think_tags=False))
    assert formatter.strip_harmony_tokens("") == ""
    assert formatter.strip_harmony_tokens(None) is None
