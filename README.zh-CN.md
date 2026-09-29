[English](README.md) | 中文

# JevPilot

JevPilot 是一个领域无关的 Agentic AI 编排框架：

- **LangChain / LLM** 负责自然语言理解和结构化输出；
- **Jev**（TypeSafe）负责决策和路由；
- **JevPilot** 负责共享状态、执行控制和能力编排；
- **专业 Capability** 负责执行具体任务；
- **科学模型** 负责真实计算。

项目自带一个 UAV（无人机）材料示例：用户用一句自然语言描述需求，系统最后给出一份
有证据、有出处的工程报告。

**状态：** 实验性研究原型（v0.1.0），不保证 API 稳定。

## 为什么这样设计

工程结论里的每个数字都必须可追溯，所以 JevPilot 不让 LLM 编造材料属性。材料数值只来自：

- 数据（真实航空材料数据集，保留原始单位、测试条件和出处）；
- 物理模型（细观力学、经典层合板理论）；
- 明确的用户输入（每个数值都必须能在用户原话里找到）。

这里说的"Agentic"，是指在共享工作流状态下协调多个专业能力，而不是让一群自主的
LLM 智能体各行其是。每一步只有一个路由器在"前置条件已满足"的能力中做选择，
并且决策在执行前都会被校验。

## 架构

```text
用户请求
  ↓
LangChain / 原生 LLM 解释器     → 经过校验的结构化需求
  ↓
TypeSafe Jev                   → 下一步调用哪个能力（或结束 / 请人工补充）
  ↓
JevPilot 控制器 + 共享状态      → 执行 · 更新状态 · 评估 · 停止
  ↓
专业 Capability
  ↓
科学领域（UAV 材料）            → 数据检索 · 细观力学 · CLT
  ↓
评估 → 报告
```

核心包 `jevpilot/` 不认识任何材料、LangChain 或模型厂商；领域包只依赖核心的公开
API；应用层 `apps/` 负责选择解释器和路由器。

## UAV Materials 示例

```text
自然语言需求
 → 结构化需求（数值必须引用用户原话）
 → 在 46 条真实航空材料记录中检索，并保留出处
 → 基于证据的决策：用现有材料、进入设计，或请求补充信息
 → 需要时进行有界复合材料逆向设计（纤维体积分数 × 对称铺层）
 → 细观力学 + 经典层合板理论预测，用同一个评估器评估
 → 仅根据计算状态生成结构化报告和 Markdown
```

内置的机翼蒙皮需求（密度 ≤ 1800 kg/m³，x、y 方向刚度 ≥ 40 GPa，面内剪切刚度 ≥
15 GPa）没有现有材料能满足，于是进入设计分支，给出一个虚拟的 [0/45/-45/90]s
碳/环氧层合板（V_f 0.65，预测密度 1579 kg/m³，Ex = Ey = 53.4 GPa，Gxy = 20.4 GPa），
并明确说明其强度、腐蚀和温度性能没有被预测。

## 离线 Demo

```bash
git clone https://github.com/uudam42/JevPilot.git && cd JevPilot
uv venv && uv pip install -e '.[dev]'
jevpilot uav-materials --demo                  # 离线、确定性，不需要密钥和网络
jevpilot uav-materials --demo --request "The density must not exceed 3000 kg/m^3 ..."
jevpilot uav-materials --demo --output report.md --json report.json --routing-log routing.jsonl
```

离线运行一律标记为 **OFFLINE_DEMO**：需求由确定性的短语解析器理解，路由由脚本化
策略给出。它只演示编排流程，不代表任何模型的能力。

## LangChain

LangChain 是可选依赖（`pip install -e '.[langchain]'`，只装 `langchain-core` 和
`langchain-anthropic`，不引入 LangGraph），提供模型厂商抽象和结构化输出，降低对单一
厂商 SDK 的耦合：

```bash
jevpilot uav-materials --demo --llm-backend langchain                  # 离线脚本化聊天模型
jevpilot uav-materials --live --llm-backend langchain --request "..."  # 真实 LLM + 真实 Jev
```

`LangChainRequirementInterpreter` 与原生解释器实现同一个接口，使用同一份提示词、同一个
schema（通过 tool calling 获得结构化输出），并经过同一套校验。**LangChain 不做 Jev 的
路由**，也不替代控制器。换了模型厂商后，schema 遵循程度、延迟、成本和理解质量仍可能不同。

## 真实 Jev 配置

```bash
uv pip install -e '.[jev]'
export TYPESAFE_API_KEY="..."                 # 仅为占位符，切勿提交或粘贴真实密钥
python -m experiments.routing.real_jev preflight         # 鉴权 + 模型发现
jevpilot uav-materials --live-routing                    # 真实 Jev 路由；需求在本地离线解释
pytest -m live_jev                                       # 可选的真实服务测试
```

密钥只从环境变量读取。模型通过 `GET /v1/models` 自动发现（也可用
`TYPESAFE_DEFAULT_MODEL` 指定），路由请求经官方 SDK 发往 `POST /v1/systemone`。
`--live` 还需要 `ANTHROPIC_API_KEY`。缺少凭据时命令以退出码 2 停止，绝不会悄悄退回
Demo。详见 [docs/REAL_ROUTING.md](docs/REAL_ROUTING.md)。

## 测试结果

```bash
pytest                                                    # 离线运行，无需密钥和网络
ruff check . && ruff format --check .
mypy --strict jevpilot domains experiments integrations examples apps
```

- 离线测试：**842 通过，3 跳过**（旧的可选真实服务测试），3 未选中（`live_jev`）。
- 真实 Jev：评测配置已冻结（`benchmarks/routing/samples/jev_live_v1.json` 中的 32 个
  eval 决策用例，加上顺序扰动、干扰能力、3 次重复，以及 4 条 UAV 分支）。
  **目前还没有真实 Jev 的测量结果**：真实服务测试尚未在当前环境中运行。

## 科学范围

- 46 条真实航空材料记录（MIL-HDBK-5J、NRL 与 NASA 报告），全部保留出处
- NASA 来源的单层板细观力学（TM-83320、TP-3290），用原文算例核对
- 经典层合板理论（NASA RP-1351），用原文给出的矩阵核对
- 在小而明确的网格上做有界复合材料逆向设计
- 证据处理：缺数据记为"证据缺口"，预测值永远不当作实测值

## 局限性

- 这是研究原型；验收策略只是演示用，不是设计规范。
- 数据集较小（46 条）且有缺口；组分材料数据为手册值。
- 不预测复合材料的强度、失效、腐蚀、吸湿和温度性能。
- 模型只与原始文献核对过，没有经过实验验证。
- 真实 LLM 解释和真实 Jev 路由依赖外部服务，实际测过什么以"测试结果"一节为准。

## 目录结构

```text
jevpilot/        核心：状态、控制循环、路由器、校验（不含领域知识，不依赖 LangChain）
integrations/    Jev（TypeSafe SDK）、Anthropic、LangChain 聊天模型、脱敏工具
domains/         uav_materials（科学计算 + 工作流）及两个测试用的小领域
apps/            UAV 材料应用：命令行、解释器、离线/真实服务接线
benchmarks/, experiments/   路由评测数据、评测框架和已发布的真实评测结果
tests/, examples/, docs/
```

## 许可与数据来源

软件许可证尚未确定，由项目所有者决定。材料数据全部来自美国政府公开文件（带
Distribution Statement A 的 MIL-HDBK-5J，以及标注为公开发布的 NASA、NRL 报告）。
`data/uav_materials/sources/` 中的清单记录了复用依据、获取日期和校验和；原始 PDF 不入库。
TypeSafe Jev 和 Anthropic 是第三方服务，适用其各自条款。

更多文档：[ARCHITECTURE](docs/ARCHITECTURE.md) · [UAV_MATERIALS](docs/UAV_MATERIALS.md) ·
[REAL_ROUTING](docs/REAL_ROUTING.md) · [BENCHMARK_DESIGN](docs/BENCHMARK_DESIGN.md)
