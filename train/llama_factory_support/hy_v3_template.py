"""
HYV3 chat template registration for LLaMA Factory.

Usage:
    1. Copy this file's register_template block into LLaMA Factory's
       src/llamafactory/data/template.py  (for upstream MR).
    2. Or import this module before training to register at runtime:
       import hy_v3_template
"""

from llamafactory.data.template import register_template
from llamafactory.data.formatter import EmptyFormatter, StringFormatter


# ---------------------------------------------------------------------------
# HYV3 (MoE, pure text) chat template - no_think mode
#
# Token format (from chat_template.jinja, is_training=true, no_think mode):
#   BOS:            <｜hy_begin▁of▁sentence｜>
#   ReasoningMode:  <｜reasoning_mode｜>reasoning_effort:no_think
#   System:         {system_content}  (between BOS+reasoning_mode and User)
#   User:           <｜hy_User｜>{user_content}
#   Assistant:      <｜hy_Assistant｜><think></think>{assistant_content}<eos:6124c78e>
#
# Full format (no system message):
#   <｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>{user}<｜hy_Assistant｜><think></think>{assistant}<eos:6124c78e>
#
# Full format (with system message):
#   <｜hy_begin▁of▁sentence｜>{system}<｜reasoning_mode｜>reasoning_effort:no_think<｜hy_User｜>{user}<｜hy_Assistant｜><think></think>{assistant}<eos:6124c78e>
#
# Loss mask: only compute loss on assistant content (including eos).
#
# Note: We do NOT use ReasoningTemplate because it places <think></think>
# in the wrong position (before <｜hy_Assistant｜> instead of after).
# Instead, we hardcode <think></think> in format_assistant to match the
# exact format defined in chat_template.jinja for is_training=true mode.
# ---------------------------------------------------------------------------

register_template(
    name="hy_v3",
    format_user=StringFormatter(slots=["<｜hy_User｜>{{content}}"]),
    format_assistant=StringFormatter(slots=["<｜hy_Assistant｜><think></think>{{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}, "<｜reasoning_mode｜>reasoning_effort:no_think"]),
    stop_words=["<｜hy_eos｜>"],
    efficient_eos=True,
)
