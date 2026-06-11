"""核心生成器：调用 Anthropic API 生成文案，并用 Pydantic 自检 + 重试修正。"""

from __future__ import annotations

import json
import logging
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from src.context_manager import (
    build_system_prompt,
    build_user_prompt,
    load_context,
)
from src.schema import ContentOutput, MarketingContext, ProductInput

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
MAX_RETRIES = 2  # 解析失败后额外重试次数

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """惰性初始化 Anthropic Client（复用单例）。"""
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "未检测到 ANTHROPIC_API_KEY，请复制 .env.example 为 .env 并填入密钥。"
            )
        _client = Anthropic()
    return _client


def _extract_json(text: str) -> str:
    """从模型回复中提取 JSON 主体，容忍偶发的 ```json 包裹。"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def generate_content(
    product: ProductInput, context: MarketingContext
) -> ContentOutput:
    """为单个商品生成结构化的多平台文案。

    Refiner 机制：解析失败时，将错误回传给模型令其自我修正，最多重试 MAX_RETRIES 次。
    """
    client = _get_client()
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(product, context)

    # 系统 Prompt（含 Schema 与 Few-Shot）稳定不变，开启 prompt caching 省成本。
    system_blocks = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages = [{"role": "user", "content": user_prompt}]

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            messages=messages,
        )
        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        try:
            payload = json.loads(_extract_json(raw_text))
            return ContentOutput(**payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "[%s] 第 %d 次解析失败：%s",
                product.sku_id,
                attempt + 1,
                exc,
            )
            if attempt < MAX_RETRIES:
                # 把上一轮回复与错误反馈给模型，要求自我修正。
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你上一次的输出无法通过校验，错误如下：\n"
                            f"{exc}\n\n"
                            "请严格按照系统提示中的 JSON Schema 重新输出，"
                            "只返回合法的 JSON 对象，不要任何多余文字。"
                        ),
                    }
                )

    raise ValueError(
        f"SKU {product.sku_id} 在 {MAX_RETRIES + 1} 次尝试后仍无法生成合法输出：{last_error}"
    )


if __name__ == "__main__":
    # 手动冒烟测试：python -m src.generator
    logging.basicConfig(level=logging.INFO)
    demo_product = ProductInput(
        sku_id="DEMO-001",
        product_name="云感轻氧防晒衣",
        category="服饰",
        features=["UPF50+", "凉感面料", "轻薄透气", "显瘦版型"],
        target_audience="20-35 岁通勤女性",
    )
    result = generate_content(demo_product, load_context())
    print(result.model_dump_json(indent=2))
