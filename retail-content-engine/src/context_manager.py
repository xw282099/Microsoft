"""上下文与模板管理器：加载营销上下文、Few-Shot 样本，并构建系统 Prompt。"""

from __future__ import annotations

import json
from pathlib import Path

from src.schema import ContentOutput, MarketingContext, ProductInput

# 项目根目录（context_manager.py 位于 <root>/src/ 下）
ROOT_DIR = Path(__file__).resolve().parent.parent
CONTEXT_PATH = ROOT_DIR / "config" / "context.json"
EXAMPLES_PATH = ROOT_DIR / "data" / "examples.json"


def load_context(path: Path = CONTEXT_PATH) -> MarketingContext:
    """读取 config/context.json 并校验为 MarketingContext。"""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return MarketingContext(**data)


def load_few_shot_examples(path: Path = EXAMPLES_PATH) -> str:
    """读取满分文案样本并序列化为可注入 Prompt 的字符串。

    文件缺失或为空时返回空串，使 Few-Shot 注入成为可选项。
    """
    if not path.exists():
        return ""
    with path.open(encoding="utf-8") as f:
        examples = json.load(f)
    if not examples:
        return ""

    blocks = []
    for i, ex in enumerate(examples, start=1):
        blocks.append(
            f"【满分样本 {i}】\n"
            + json.dumps(ex, ensure_ascii=False, indent=2)
        )
    return "\n\n".join(blocks)


def build_system_prompt() -> str:
    """构建系统级 Prompt：角色设定 + 强制 JSON Schema 输出。

    内容稳定，便于在 generator 中开启 prompt caching。
    """
    schema_json = json.dumps(
        ContentOutput.model_json_schema(), ensure_ascii=False, indent=2
    )
    few_shot = load_few_shot_examples()

    prompt = f"""你是一位资深零售品牌营销专家，精通小红书、抖音/TikTok 与电商详情页的文案创作。
你的任务是：基于给定的商品信息与营销上下文，产出符合品牌调性、能直接投放的多平台营销文案。

要求：
1. 必须贴合给定的品牌调性、季节与流行趋势。
2. 文案要有传播力与转化力，杜绝空洞套话。
3. 你的输出必须是「单个」严格合法的 JSON 对象，且完全符合下方 JSON Schema。
4. 不要输出任何解释、Markdown 代码块标记或多余文字，只输出 JSON 本身。

输出必须严格符合以下 JSON Schema：
{schema_json}"""

    if few_shot:
        prompt += f"""

以下是「满分文案样本」，请学习其风格与结构（注意：仅供参考风格，请勿照抄内容）：
{few_shot}"""

    return prompt


def build_user_prompt(product: ProductInput, context: MarketingContext) -> str:
    """构建单次生成请求的用户消息。"""
    return f"""请为以下商品生成多平台营销文案。

## 营销上下文
- 季节：{context.season}
- 流行趋势：{context.current_trend}
- 品牌调性：{context.brand_voice}

## 商品信息
- SKU：{product.sku_id}
- 名称：{product.product_name}
- 品类：{product.category}
- 卖点：{"、".join(product.features)}
- 目标受众：{product.target_audience}

请直接输出符合 Schema 的 JSON。"""
