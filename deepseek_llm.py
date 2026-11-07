"""DeepSeek LLM adapter for pandasai 3.0.0."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from pandasai.llm.base import LLM

# 从当前文件所在目录的 .env 读取配置
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

_SYSTEM_PROMPT = (
    "你是一个严谨的数据分析助手。\n"
    "只能使用 DataFrame 表结构里给出的表名和列名回答问题；"
    "禁止查询 information_schema、columns 等任何数据库系统表或元数据表。\n"
    "只输出可执行的 Python 代码，不要输出多余解释。"
)


class DeepSeekLLM(LLM):
    """自定义 DeepSeek LLM，供 pandasai Agent 调用。"""

    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置，请在项目根目录的 .env 文件中配置。"
            )
        super().__init__(api_key=api_key)
        self.model = model
        self.api_url = "https://api.deepseek.com/chat/completions"

    @property
    def type(self) -> str:
        return "deepseek"

    def call(self, instruction, context=None) -> str:
        """调用 DeepSeek，原样返回 choices[0].message.content。"""
        prompt = instruction.to_string() if hasattr(instruction, "to_string") else str(instruction)

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
