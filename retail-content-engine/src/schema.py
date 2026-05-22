"""Pydantic 数据模型：锁死输入/上下文/输出的结构契约。"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ProductInput(BaseModel):
    """单个商品（SKU）的结构化输入。"""

    sku_id: str = Field(..., description="商品唯一编号")
    product_name: str = Field(..., description="商品名称")
    category: str = Field(..., description="商品品类")
    features: List[str] = Field(..., description="卖点 / 功能特性列表")
    target_audience: str = Field(..., description="目标受众画像")


class MarketingContext(BaseModel):
    """当前营销上下文：季节、趋势、品牌调性。"""

    season: str = Field(..., description="当前营销季节，如 '2026 夏季'")
    current_trend: str = Field(..., description="当前流行趋势，如 'Y2K 复古'")
    brand_voice: str = Field(..., description="品牌调性，如 '极简、高级、专业'")


class Xiaohongshu(BaseModel):
    """小红书平台文案。"""

    title: str = Field(..., description="带 emoji 的吸睛标题")
    content: str = Field(..., description="正文，可分段（用换行）")
    tags: List[str] = Field(..., description="话题标签列表，每项以 # 开头")


class TiktokScript(BaseModel):
    """抖音 / TikTok 短视频脚本。"""

    hook: str = Field(..., description="前 3 秒钩子")
    body: str = Field(..., description="产品展示讲解")
    cta: str = Field(..., description="行动呼吁")


class EcommerceDetails(BaseModel):
    """电商详情页内容。"""

    bullet_points: List[str] = Field(..., description="产品卖点要点")


class ContentOutput(BaseModel):
    """多平台营销文案的最终结构化输出。"""

    xiaohongshu: Xiaohongshu
    tiktok_script: TiktokScript
    ecommerce_details: EcommerceDetails
