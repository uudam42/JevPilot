[English](README.md) | 中文

# JevPilot

JevPilot 是一个领域无关的 Agentic AI 编排框架：由可选的 LangChain 大语言模型解释层、
TypeSafe Jev 决策路由、专业执行能力和确定性科学模型组合而成。

**状态：** v0.1 研究原型，已冻结；不保证 API 稳定。

## JevPilot 做什么

它把一段自然语言的工程需求，变成一份证据可追溯的工程报告。每个部分只负责自己的事：

| 组成 | 职责 | 绝不做 |
|---|---|---|
| Claude（经 LangChain 或原生接口） | 大语言模型解释层：把需求转成经过校验的结构化需求 | 编造材料属性、数值限制或单位 |
| TypeSafe Jev | Jev 决策路由：决定下一步做什么（或结束、或请人工补充） | 执行任何操作 |
| JevPilot | 编排与状态控制：共享状态、校验、执行控制 | 猜测用户的意思 |
| Capability | 专业执行能力：完成具体的工作流步骤 | 自己决定路由 |
| 科学模型 | 确定性科学模型：负责真实的工程计算 | 超出文献依据下结论 |

材料数值只来自数据、物理模型和用户明确给出的输入，JevPilot 不让大语言模型编造材料属性。
这是在共享状态下协调多个专业能力的编排，而不是一群各自为政的自主 LLM 智能体；
LangChain 不参与路由。

## 架构

```text
用户
  ↓
Claude（经 LangChain / 原生接口）     自然语言 → 结构化需求
  ↓
TypeSafe Jev                         决策路由
  ↓
JevPilot 控制器 + 共享状态            校验 · 执行 · 更新 · 评估 · 停止
  ↓
6 个专业执行能力
  ↓
UAV 材料领域                          真实数据 · 需求语义 · 证据
  ↓
科学模型                              细观力学 · 经典层合板理论
  ↓
工程报告
```

核心包 `jevpilot/` 不认识任何材料、LangChain 或模型厂商。详见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## UAV 材料示例

```text
需求 → 结构化需求 → 在 46 条航空材料记录中检索
     → 基于证据的决策（用现有材料 / 进入设计 / 请求补充信息）
     → 需要时进行有界复合材料逆向设计（共 28 个设计）
     → 细观力学 + 经典层合板理论 → 报告
```

内置的机翼蒙皮需求（密度 ≤ 1800 kg/m³，x、y 方向刚度 ≥ 40 GPa，面内剪切刚度 ≥
15 GPa）没有现有材料能满足。在检索的 28 个设计中有 8 个可行，最优的是一个虚拟的
[0/45/-45/90]s 碳/环氧层合板（V_f 0.65，预测密度 1579 kg/m³，Ex = Ey = 53.4 GPa，
Gxy = 20.4 GPa）。报告会明确说明它的强度、腐蚀和温度性能没有被预测。方法详见
[docs/UAV_MATERIALS.md](docs/UAV_MATERIALS.md)。

## 快速开始

```bash
git clone https://github.com/uudam42/JevPilot.git && cd JevPilot
uv venv && uv pip install -e '.[dev]'        # 或：python -m venv .venv && pip install -e '.[dev]'
jevpilot uav-materials --demo                 # 离线、确定性，不需要密钥和网络
jevpilot uav-materials --demo --request "The density must not exceed 3000 kg/m^3 ..."
jevpilot uav-materials --demo --output report.md --json report.json
```

离线运行一律标记为 **OFFLINE_DEMO**，不代表任何模型的能力。

## 真实服务模式

```bash
uv pip install -e '.[jev,langchain]'
export TYPESAFE_API_KEY="..."                 # 仅为占位符，切勿提交真实密钥
export ANTHROPIC_API_KEY="..."
jevpilot uav-materials --live --llm-backend langchain --request "..."   # Claude + Jev
jevpilot uav-materials --live-routing          # 真实 Jev 路由，需求在本地离线解释
pytest -m live_jev                             # 可选的真实服务测试
```

密钥只从环境变量读取；缺少凭据时命令以退出码 2 停止，绝不会悄悄退回 Demo。
LangChain 是可选依赖（只用 `langchain-core` 和 `langchain-anthropic`，不引入 LangGraph），
能降低对单一厂商的耦合，但不同模型在 schema 遵循程度、延迟和成本上仍有差异。配置详见
[docs/REAL_ROUTING.md](docs/REAL_ROUTING.md)。

## 验证结果

测量日期 2026-09-29。Claude 为 `claude-opus-5`；Jev 请求 `jev-preview`，实际服务模型为 `jev-1.13.0`。

- **全链路真实运行**（真实 Claude → LangChain → 真实 Jev → JevPilot → UAV 工作流）：
  5/5 次运行全部完成，其中 3 次生成了完整报告；另外 2 次（信息不足、属性不受支持）
  Jev 在生成报告前请求人工补充，因此报告标记为未完成。所有决策都与冻结的离线参考结果一致。
- **真实 Jev 冻结评测集：** 540 次 API 请求，0 次 API 错误；重复 3 次时，32 个冻结用例中
  31 个决策完全一致；冻结工作流评测完成率 89.7%（26/29）。
- **延迟（全链路真实运行）：** Claude 解释 4.0–10.4 秒；Jev 路由 p50 113 ms、p95 308 ms
  （24 次决策）；端到端 4.7–12.2 秒。
- **测试：** 离线 843 通过、3 跳过、3 未选中（真实服务测试）；`pytest -m live_jev`
  3 项通过；11 个示例全部可运行。
- **科学结果：** 接入 LangChain 和 Jev 后科学计算结果完全不变。Claude 验证的是解释层角色，
  不是路由角色。

完整结果、方法和发现的问题见 [docs/REAL_ROUTING.md](docs/REAL_ROUTING.md)（第 7–8 节），
脱敏后的原始结果见
[`experiments/routing/results/published/`](experiments/routing/results/published/)。

## 科学范围

- 46 条航空材料记录（MIL-HDBK-5J、NRL 与 NASA 报告），证据可追溯
- NASA 来源的单层板细观力学和经典层合板理论，均用原始文献核对
- 有界复合材料逆向设计：4 种纤维体积分数 × 7 种对称铺层
- 缺数据记为"证据缺口"，预测值永远不当作实测值

## 局限性

- 这是研究原型；验收策略只是演示用，不是设计规范。
- 数据集较小且有缺口；不预测复合材料的强度、失效、腐蚀、吸湿和温度性能；
  模型没有经过实验验证。
- 真实服务验证规模有限：一个冻结评测集加 5 次全链路真实运行，只测试了一家大语言模型厂商；
  只是证据，不是普遍保证。

## 目录与文档

```text
jevpilot/       核心：状态、控制循环、路由器、校验
integrations/   TypeSafe Jev、Anthropic、LangChain 适配器；脱敏工具
domains/        uav_materials（科学计算 + 工作流）及两个测试用的小领域
apps/           UAV 材料应用与命令行
benchmarks/, experiments/   冻结的路由评测集、评测框架、已发布结果
tests/, examples/, docs/
```

[ARCHITECTURE](docs/ARCHITECTURE.md) · [UAV_MATERIALS](docs/UAV_MATERIALS.md) ·
[REAL_ROUTING](docs/REAL_ROUTING.md) · [BENCHMARK_DESIGN](docs/BENCHMARK_DESIGN.md)

软件许可证尚未确定。材料数据全部来自美国政府公开文件（复用依据和校验和见
`data/uav_materials/sources/`）。TypeSafe Jev 和 Anthropic 是第三方服务，适用其各自条款。
