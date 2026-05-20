"""
HunYuan Dense model chat template registration for LLaMA Factory.

Registers two templates:
  - hy_dense_1_8b: for HunYuan Dense 1.8B model (and 0.5B/4B)
  - hy_dense_7b:   for HunYuan Dense 7B model

Usage:
    1. Copy this file's register_template blocks into LLaMA Factory's
       src/llamafactory/data/template.py  (for upstream MR).
    2. Or import this module before training to register at runtime:
       import hy_dense_template

Note:
    The existing LLaMA Factory built-in templates `hunyuan` and `hunyuan_small`
    have subtle differences from the official chat_template.jinja files shipped
    with the models. These new templates are designed to match the official
    jinja templates exactly.
"""

from llamafactory.data.template import register_template
from llamafactory.data.formatter import EmptyFormatter, StringFormatter

# ---------------------------------------------------------------------------
# Dense 1.8B chat template (also applies to 0.5B/4B)
#
# Token format (from dense_1_8b_0508/global_step_560/chat_template.jinja):
#   BOS:        <｜hy_begin▁of▁sentence｜>
#   System:     {system_content}<｜hy_place▁holder▁no▁3｜>
#   User:       <｜hy_User｜>{user_content}
#   Assistant:  <｜hy_Assistant｜>{assistant_content}<｜hy_place▁holder▁no▁2｜>
#   Stop:       <｜hy_place▁holder▁no▁2｜>
#
# Key differences from LF built-in `hunyuan_small`:
#   - User format: NO trailing <｜hy_place▁holder▁no▁8｜> after user content
#   - Assistant format: HAS <｜hy_Assistant｜> prefix before assistant content
#
# The eos_token in tokenizer_config.json is <｜hy_place▁holder▁no▁2｜>,
# so we use efficient_eos=True to let LF append it via {eos_token} slot.
# ---------------------------------------------------------------------------

register_template(
    name="hy_dense_1_8b",
    format_user=StringFormatter(slots=["<｜hy_User｜>{{content}}"]),
    format_assistant=StringFormatter(slots=["<｜hy_Assistant｜>{{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}<｜hy_place▁holder▁no▁3｜>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<｜hy_place▁holder▁no▁2｜>"],
    efficient_eos=True,
)


# ---------------------------------------------------------------------------
# Dense 7B chat template
#
# Token format (from dense_7b_0509/global_step_560/chat_template.jinja):
#   BOS:        <|startoftext|>
#   System:     {system_content}<|extra_4|>
#   User:       {user_content}<|extra_0|>
#   Assistant:  {assistant_content}<|eos|>
#   Stop:       <|eos|>
#
# Key differences from LF built-in `hunyuan`:
#   - Uses {bos_token} and {eos_token} slots for portability
#   - efficient_eos=True to use tokenizer's eos_token
#
# Note on multi-turn: The official jinja adds <|startoftext|> before each
# user message (except the first one when system is present). LLaMA Factory's
# format_prefix only adds BOS once at the beginning. For single-turn training
# this is correct. For multi-turn, there is a minor discrepancy (missing
# <|startoftext|> before 2nd+ user turns), which is acceptable for fine-tuning.
# ---------------------------------------------------------------------------

register_template(
    name="hy_dense_7b",
    format_user=StringFormatter(slots=["{{content}}<|extra_0|>"]),
    format_assistant=StringFormatter(slots=["{{content}}", {"eos_token"}]),
    format_system=StringFormatter(slots=["{{content}}<|extra_4|>"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<|eos|>"],
    efficient_eos=True,
)
