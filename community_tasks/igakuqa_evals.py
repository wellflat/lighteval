# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


# fmt: off
LETTER_INDICES = ["A", "B", "C", "D", "E"]
# INTEGER_INDICES = list(map(str, list(range(1, 6))))
# fmt: on

def post_process_answers(pred: str) -> str:
    """
    LLMの生成した回答から不要な空白を除去し、アルファベット順に並び替えて正規化する。
    これにより、回答の順序や不要な空白の有無に影響されずに評価できる。
    例: "  C, A  " -> "A,C"
    """
    # カンマで分割し、各要素の空白を削除
    parts = [part.strip() for part in pred.split(",")]
    # 空の文字列（例: "A,,C"のような入力）を除外
    non_empty_parts = [part for part in parts if part]
    # アルファベット順にソートし、カンマ区切りで結合
    return ",".join(sorted(non_empty_parts))



# DEFINE YOUR PROMPT FUNCTIONS
# Define as many as you need for your different tasks
def prompt_fn(line, task_name: str = None):
    """
    IgakuQAのデータセットの行をDocオブジェクトに変換します。
    MMLUと同様のプロンプト形式を採用し、モデルにタスクの内容と回答形式を明確に伝えます。
    """
    # 複数選択問題であることをモデルに明確に指示
    instruction = """
    以下は、日本の医師国家試験に関する多肢選択問題です。正しい選択肢のアルファベットを必ず全てカンマ区切りのアルファベット順で答えてください。（例: A, C）
    追加の解説などは不要です。

    """

    gold_indices = as_list(line["answer"])

    # MMLU形式のプロンプトを構築
    query = line["problem_text"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    query += "答え:"

    #print(query)

    # 正解の選択肢を "A,C" のような文字列形式に変換
    gold_str = ", ".join(sorted([LETTER_INDICES[i] for i in gold_indices]))

    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        # 評価のため、choicesに正解の文字列のみを設定
        choices=[gold_str],
        # gold_indexは常に0になる
        gold_index=0,
        instruction=instruction,
    )


# EVAL WITH NO SUBSET ##
# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
task = LightevalTaskConfig(
    name="igakuqa",
    prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="wellflat/IgakuQA-filtered",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.exact_match],  # 生成された文字列が正解と完全に一致するかを判定
    stop_sequence=None,
    version=0
)

TASKS_TABLE = [task]
