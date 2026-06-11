"""批处理入口：读取 SKU CSV，批量生成文案，扁平化写出 CSV。

用法：
    python main.py
可选参数：
    python main.py --input data/input_skus.csv --output data/output_contents.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.context_manager import load_context
from src.generator import generate_content
from src.schema import ContentOutput, ProductInput

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT_DIR / "data" / "input_skus.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "output_contents.csv"

FEATURES_SEP = ";"  # input CSV 中 features 字段的分隔符

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retail-content-engine")


def row_to_product(row: pd.Series) -> ProductInput:
    """将一行 CSV 数据转换为 ProductInput。"""
    features = [
        f.strip()
        for f in str(row["features"]).split(FEATURES_SEP)
        if f.strip()
    ]
    return ProductInput(
        sku_id=str(row["sku_id"]).strip(),
        product_name=str(row["product_name"]).strip(),
        category=str(row["category"]).strip(),
        features=features,
        target_audience=str(row["target_audience"]).strip(),
    )


def flatten_output(product: ProductInput, content: ContentOutput) -> dict:
    """将嵌套的 ContentOutput 扁平化为单行字典，便于写入 CSV。"""
    xhs = content.xiaohongshu
    tt = content.tiktok_script
    ec = content.ecommerce_details
    return {
        "sku_id": product.sku_id,
        "product_name": product.product_name,
        "category": product.category,
        "xhs_title": xhs.title,
        "xhs_content": xhs.content,
        "xhs_tags": " ".join(xhs.tags),
        "tiktok_hook": tt.hook,
        "tiktok_body": tt.body,
        "tiktok_cta": tt.cta,
        "ecommerce_bullets": "\n".join(f"- {b}" for b in ec.bullet_points),
        "status": "success",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail-Content-Engine 批处理")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    context = load_context()
    df = pd.read_csv(args.input)
    total = len(df)
    logger.info("读取到 %d 个 SKU，开始生成…", total)

    rows: list[dict] = []
    success_count = 0
    for idx, row in df.iterrows():
        sku_id = str(row.get("sku_id", f"row-{idx}"))
        logger.info("[%d/%d] 处理 SKU %s …", idx + 1, total, sku_id)
        try:
            product = row_to_product(row)
            content = generate_content(product, context)
            rows.append(flatten_output(product, content))
            success_count += 1
            logger.info("[%d/%d] ✅ %s 生成成功", idx + 1, total, sku_id)
        except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整批
            logger.error("[%d/%d] ❌ %s 生成失败：%s", idx + 1, total, sku_id, exc)
            rows.append({"sku_id": sku_id, "status": f"failed: {exc}"})

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    logger.info(
        "全部完成：成功 %d / 共 %d，结果已写入 %s",
        success_count,
        total,
        args.output,
    )


if __name__ == "__main__":
    main()
