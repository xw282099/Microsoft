# Retail-Content-Engine 零售内容自动化引擎

读取结构化商品（SKU）数据，结合营销上下文与 Few-Shot 满分范例，调用 Anthropic API
自动生成符合品牌调性的多平台营销文案（小红书 / 抖音 / 电商详情页），并保证输出 100% 结构化。

## 目录结构

```
retail-content-engine/
├── requirements.txt          # pydantic / anthropic / python-dotenv / pandas
├── .env.example              # 预留 ANTHROPIC_API_KEY
├── main.py                   # 批处理入口（CSV → CSV + 进度日志）
├── config/context.json       # 营销上下文（season / current_trend / brand_voice）
├── data/
│   ├── examples.json         # Few-Shot 满分文案样本
│   └── input_skus.csv        # 测试 SKU 数据
└── src/
    ├── schema.py             # Pydantic 数据模型（输入 / 上下文 / 三平台输出）
    ├── context_manager.py    # 上下文/Few-Shot 加载 + 系统 Prompt 模板
    └── generator.py          # Anthropic 调用 + prompt caching + Refiner 自检重试
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入真实 ANTHROPIC_API_KEY
python main.py                # 读 data/input_skus.csv → 写 data/output_contents.csv
```

可选参数：

```bash
python main.py --input data/input_skus.csv --output data/output_contents.csv
```

## 核心设计

- **结构锁死**：`ContentOutput` 把三平台输出嵌套为子模型，并将其 JSON Schema 注入系统
  Prompt，强制模型按 Schema 输出。
- **Refiner 自检**：API 返回后用 Pydantic 解析，失败时把错误回传给模型令其自我修正，
  最多重试 2 次。
- **省成本**：稳定的系统 Prompt（含 Schema 与 Few-Shot）开启 prompt caching。
- **健壮批处理**：单条 SKU 失败不中断整批，失败状态写入输出 CSV 的 `status` 列。

## 输入 CSV 格式

| 列名 | 说明 |
| --- | --- |
| `sku_id` | 商品唯一编号 |
| `product_name` | 商品名称 |
| `category` | 品类 |
| `features` | 卖点，多个用 `;` 分隔 |
| `target_audience` | 目标受众 |
