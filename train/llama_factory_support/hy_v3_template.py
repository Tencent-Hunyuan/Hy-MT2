"""
HYV3 chat template registration for LLaMA Factory.

Usage:
    1. Copy this file's register_template block into LLaMA Factory's
       src/llamafactory/data/template.py  (for upstream MR).
    2. Or import this module before training to register at runtime:
       import hy_v3_template
"""

from llamafactory.data.template import ReasoningTemplate, register_template
from llamafactory.data.formatter import EmptyFormatter, StringFormatter


# ---------------------------------------------------------------------------
# HYV3 (MoE, pure text) chat template
#
# Token format (from chat_template.jinja & tokenizer_config.json):
#   BOS:        <｜hy_begin▁of▁sentence｜>
#   System:    {system_content}                (directly after BOS, no role tag)
#   User:       <｜hy_User｜>{user_content}
#   Assistant:  <｜hy_Assistant｜>{assistant_content}<｜hy_eos｜>
#   EOS:       <｜hy_eos｜>
#
# Loss mask: only compute loss on assistant content (including <｜hy_eos｜>).
#
# Note: The system message has NO explicit role token -- it is placed right
# after BOS.  The eos_token is <｜hy_eos｜>.
#
# Reasoning: Supports think tags via ReasoningTemplate.
#   - thought_words: ("<think>", "</think>") matching jinja template
#   - enable_thinking: set globally via data_args.enable_thinking (default True)
#   - Training data always includes think tags (empty or with content)
# ---------------------------------------------------------------------------

register_template(
    name="hy_v3",
    template_class=ReasoningTemplate,
    format_user=StringFormatter(slots=["<｜hy_User｜>{{content}}"]),
    format_assistant=StringFormatter(slots=["<｜hy_Assistant｜>{{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    thought_words=("<think>", "</think>"),
    stop_words=["<｜hy_eos｜>"],
    efficient_eos=True,
)
