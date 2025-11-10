
# **AgenticInfer：Agentic-native 推理引擎架构设计文档 v1.2**  

---

## 一、设计目标与核心理念

### 1.1 核心理念（严格遵循 AgenticDSL v3.7）

> **“推理行为应成为可验证、可组合、可归档的 DAG 节点，而非黑盒。”**

- ✅ **原生兼容 `llm_call`**：利用规范允许的“额外字段”机制（5.7），无需新增字段  
- ✅ **能力契约化**：通过 `/lib/reasoning/**` 标准子图暴露推理能力（6.2）  
- ✅ **资源声明驱动**：通过 `/__meta__/resources` 声明 `reasoning` 能力（6.4）  
- ✅ **完全三层架构**：执行原语层 (`llm_call`) → 标准原语层 (`/lib/reasoning/**`) → 知识应用层 (`/app/inference/**`)  
- ✅ **引擎即应用**：AgenticInfer 本身是一个 AgenticDSL 应用（`/app/inference/native_engine_v1`）

---

## 二、整体架构

### 2.1 执行流程（无语义扩展）

```mermaid
flowchart LR
    A[/main/task] --> B[/lib/reasoning/structured_generate@v1]
    B --> C{执行器：是否声明 native_inference_core?}
    C -->|是| D[启动嵌套引擎: /app/inference/native_engine_v1]
    C -->|否| E[调用传统后端（如 llama.cpp）]
    D --> F[/app/inference/tokenize@v1]
    F --> G[/app/inference/alloc_kv@v1]
    G --> H[/app/inference/run_attention@v1]
    H --> I[/app/inference/apply_grammar@v1]
    I --> J[/app/inference/stream_output@v1]
    J --> K[返回主上下文]
```

### 2.2 触发机制（合规方式）

- 用户在 `/__meta__/resources` 中声明：
  ```agentic
  - type: tool
    name: native_inference_core
    scope: internal
  ```
- 执行器在 DAG 启动时检测该资源
- **所有 `llm_call` 节点自动路由至 AgenticInfer**
- **未声明则降级至传统后端**

> ✅ **合规依据**：5.7 允许 `llm` 对象包含额外字段；6.4 允许工具资源声明

---

## 三、标准原语层：`/lib/reasoning/**`（必须实现）

根据 AgenticDSL v3.7 **附录 C** 与 **10.2 推理原语**，新增以下 5 个标准子图（均带 `signature`）：

### 1. **`/lib/reasoning/generate_text@v1`（stable）**

```yaml
signature:
  inputs:
    - name: prompt; type: string; required: true
    - name: model; type: string; required: true
    - name: seed; type: integer; required: true
    - name: temperature; type: number; default: 0.0
    - name: max_tokens; type: integer; default: 256
  outputs:
    - name: text; type: string
    - name: kv_handle; type: string
version: "1.0"
stability: stable
permissions: [reasoning: lmm_generate]
type: llm_call
llm:
  model: "{{ $.model }}"
  seed: "{{ $.seed }}"
  temperature: "{{ $.temperature }}"
  max_tokens: "{{ $.max_tokens }}"
```

---

### 2. **`/lib/reasoning/structured_generate@v1`（stable）**

```yaml
signature:
  inputs:
    - name: prompt; type: string; required: true
    - name: output_schema; type: object; required: true
    - name: seed; type: integer; required: true
    - name: model; type: string; required: true
  outputs:
    - name: parsed_output; type: object
version: "1.0"
stability: stable
permissions: [reasoning: structured_generate]
type: llm_call
llm:
  model: "{{ $.model }}"
  seed: "{{ $.seed }}"
  temperature: 0.0
  # output_schema 作为额外字段，由 AgenticInfer 识别
```

---

### 3. **`/lib/reasoning/continue_from_kv@v1`（stable）**

```yaml
signature:
  inputs:
    - name: kv_handle; type: string; required: true
    - name: new_prompt; type: string; required: true
    - name: model; type: string; required: true
  outputs:
    - name: continuation; type: string
    - name: updated_kv_handle; type: string
version: "1.0"
stability: stable
permissions: [reasoning: lmm_generate]
type: llm_call
llm:
  model: "{{ $.model }}"
  kv_handle: "{{ $.kv_handle }}"
  prompt: "{{ $.new_prompt }}"
```

---

### 4. **`/lib/reasoning/stream_until@v1`（stable）**

```yaml
signature:
  inputs:
    - name: prompt; type: string; required: true
    - name: stop_condition; type: string; required: true
    - name: max_tokens; type: integer; default: 2048
    - name: model; type: string; required: true
  outputs:
    - name: streamed_output; type: string
version: "1.0"
stability: stable
permissions: [reasoning: stream_output]
type: llm_call
llm:
  model: "{{ $.model }}"
  prompt: "{{ $.prompt }}"
  stop_condition: "{{ $.stop_condition }}"
  max_tokens: "{{ $.max_tokens }}"
```

---

### 5. **`/lib/reasoning/speculative_decode@v1`（experimental）**

```yaml
signature:
  inputs:
    - name: prompt; type: string; required: true
    - name: draft_model; type: string; default: "phi-3-mini"
    - name: target_model; type: string; required: true
  outputs:
    - name: verified_output; type: string
    - name: acceptance_rate; type: number
version: "1.0"
stability: experimental
permissions: [reasoning: speculative_decode]
type: llm_call
llm:
  model: "{{ $.target_model }}"
  draft_model: "{{ $.draft_model }}"
  prompt: "{{ $.prompt }}"
```

> ✅ **合规依据**：10.2 允许新增推理原语；6.2 要求带 `signature`

---

## 四、推理引擎专属工作流：`/app/inference/**`

### 4.1 引擎入口：`/app/inference/native_engine_v1`

```agentic
AgenticDSL '/app/inference/native_engine_v1'
type: assign
assign:
  expr: "{{ $.llm.prompt }}"
  path: "engine_input.prompt"
next: "/app/inference/tokenize@v1"

AgenticDSL '/app/inference/tokenize@v1'
type: tool_call
tool: native_tokenize
arguments:
  text: "{{ $.engine_input.prompt }}"
output_mapping:
  tokens: "engine_state.input_ids"
next: "/app/inference/alloc_kv@v1"

AgenticDSL '/app/inference/alloc_kv@v1'
type: tool_call
tool: kv_alloc
arguments:
  num_blocks: "{{ (len($.engine_state.input_ids) + 255) // 256 }}"
output_mapping:
  block_ids: "engine_state.kv_blocks"
next: "/app/inference/run_attention@v1"

AgenticDSL '/app/inference/run_attention@v1'
type: tool_call
tool: model_step
arguments:
  tokens: "{{ $.engine_state.input_ids }}"
  kv_ref: "{{ $.engine_state.kv_blocks }}"
output_mapping:
  logits: "engine_state.logits"
  updated_kv: "engine_state.kv_blocks"
next: "{{ $.llm.output_schema ? '/app/inference/apply_grammar@v1' : '/app/inference/stream_output@v1' }}"
```

> ✅ **合规依据**：2.1 允许 `/app/**` 作为知识应用层；5.2 `tool_call` 为合法叶子节点

---

## 五、C++ 执行原语层模块

| C++ 模块 | 工具名 | 输入 | 输出 | 权限 |
|--------|--------|------|------|------|
| Tokenizer | `native_tokenize` | `{text}` | `{tokens}` | `internal: inference_core` |
| KVBlockAllocator | `kv_alloc` | `{num_blocks}` | `{block_ids}` | `internal: inference_core` |
| ModelExecutor | `model_step` | `{tokens, kv_ref}` | `{logits, updated_kv}` | `internal: inference_core` |
| GrammarCompiler | `compile_grammar` | `{schema}` | `{logits_mask}` | `internal: inference_core` |
| StreamingController | `stream_until` | `{stop_condition, max_tokens}` | `{text}` | `internal: inference_core` |

> 🔒 所有工具权限为 `internal`，禁止外部直接调用（7.2）

---

## 六、工作流示例

### 6.1 基础示例：文本生成

```agentic
AgenticDSL '/main/greet'
type: assign
assign:
  expr: "Hello"
next: "/lib/reasoning/generate_text@v1"
```

### 6.2 高级示例：结构化生成 + KV 复用

```agentic
AgenticDSL '/main/solve_math'
type: assign
assign:
  expr: "解方程: x^2 + 2x + 1 = 0"
next: "/lib/reasoning/structured_generate@v1"

AgenticDSL '/main/explain'
type: assign
assign:
  expr: "请解释为什么根是 -1"
next: "/lib/reasoning/continue_from_kv@v1"
```

---

## 七、AgenticInfer 的本质超越点

| 能力 | llama.cpp / vLLM / SGLang | AgenticInfer |
|------|-------------------------|-------------|
| **控制粒度** | 请求级 / Token 级 | **DAG 节点级**（每步可 `assert` / Trace） |
| **组合性** | 固定 pipeline | **任意组合**（通过 DAG 编排） |
| **可验证性** | 黑盒输出 | **过程可验证**（`expected_output` + Trace） |
| **演进性** | 模型为中心 | **子图为中心**（`archive_to` 成功 DAG） |
| **缓存单位** | Token 前缀 | **子图语义 + Token 前缀** |
| **调度单位** | 请求 batch | **DAG 分支感知 batch** |

> 🚀 **核心突破**：将“推理策略”从 C++ 代码转化为 DAG 节点，实现 **推理即程序**。

---

## 八、合规性与安全性

| 规范要求 | AgenticInfer 实现 |
|--------|------------------|
| **三层架构** | ✅ 无跨层调用 |
| **标准库契约** | ✅ 所有 `/lib/reasoning/**` 带 `signature` |
| **权限最小化** | ✅ C++ 工具权限为 `internal` |
| **预算控制** | ✅ 嵌套引擎继承 `max_nodes * 0.8` |
| **可终止性** | ✅ `stream_until` 强制 `max_tokens` |
| **Trace** | ✅ 每步记录 `reasoning_evidence` + `backend_used` |

---

## 九、总结

**AgenticInfer v1.2**：

- ✅ **完全兼容 AgenticDSL v3.7**，无需任何语义扩展  
- ✅ **通过标准 `/lib/reasoning/**` 子图暴露能力**  
- ✅ **C++ 模块仅作为 `tool_call` 实现**  
- ✅ **推理流程由 `/app/inference/**` DAG 编排**  
- ✅ **本质超越传统引擎：推理即 DAG，策略即子图**

> **标语**：  
> **“AgenticInfer: Where Inference Becomes a Verifiable DAG.”**
