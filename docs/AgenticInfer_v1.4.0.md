# **AgenticInfer：Agentic-native 推理引擎架构设计文档 v1.4**  
> **完全兼容 AgenticDSL v3.7，无需扩展语义，通过标准 `/lib/reasoning/**` 接口驱动 DAG-native 推理**

---

## 一、设计目标与核心理念

### 1.1 核心理念（严格遵循 AgenticDSL v3.7）

> **“推理行为应成为可验证、可组合、可归档的 DAG 节点，而非黑盒。”**

AgenticInfer 在 **不修改 AgenticDSL 任何语法或节点语义** 的前提下，实现：

- ✅ **原生兼容 `llm_call`**：利用规范允许的“额外字段”机制（5.7）  
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

### 3.1 5 个新增标准子图（带 `requires` 与错误处理）

#### 示例：`/lib/reasoning/structured_generate@v1`（stable）

```agentic
### AgenticDSL '/lib/reasoning/structured_generate@v1'
signature:
  inputs:
    - name: prompt; type: string; required: true
    - name: model; type: string; required: true
    - name: seed; type: integer; required: true
    - name: output_schema; type: object; required: true
  outputs:
    - name: parsed_output; type: object; required: true
version: "1.0"
stability: stable
requires:
  - lib: "/lib/reasoning/generate_text@^1.0"
  - tool: "native_inference_core"
permissions:
  - reasoning: structured_generate
on_error: "/lib/reasoning/fallback_to_text@v1"
type: llm_call
llm:
  model: "{{ $.model }}"
  seed: "{{ $.seed }}"
  temperature: 0.0
  prompt: "{{ $.prompt }}"
  output_schema: "{{ $.output_schema }}"
```

> ✅ **合规依据**：7.4（依赖声明）、7.2（`on_error`）、6.2（`signature`）

---

## 四、C++ 执行原语层模块架构

### 4.1 项目结构

```text
agentic-native-inference/
├── CMakeLists.txt
├── include/
│   ├── model/               # Transformer 模型加载与执行
│   ├── scheduler/           # 推理任务调度
│   ├── kv/                  # 分页 KV 管理
│   ├── prefix/              # 前缀共享索引
│   ├── decode/              # 解码策略
│   ├── grammar/             # 结构化输出约束
│   └── tools/               # 所有 `tool_call` 实现
├── src/
│   ├── model/
│   ├── scheduler/
│   ├── kv/
│   ├── prefix/
│   ├── decode/
│   ├── grammar/
│   └── tools/
├── kernels/                 # CUDA kernels
└── tests/
```

### 4.2 C++ 工具注册代码模板（符合 2.2 适配器模式）

```cpp
// include/tools/tool_registry.h
struct ToolSchema {
  std::vector<std::pair<std::string, std::string>> inputs;   // {name, type}
  std::vector<std::pair<std::string, std::string>> outputs;  // {name, type}
  std::vector<std::string> required_permissions;
};

class ToolRegistry {
public:
  using ToolFunc = std::function<JsonValue(const JsonValue& args)>;

  void registerTool(
    const std::string& name,
    ToolFunc impl,
    const ToolSchema& schema
  );
};
```

```cpp
// src/tools/native_inference_tools.cpp
#include "tools/tool_registry.h"
#include "grammar/grammar_compiler.h"
#include "model/model_executor.h"
#include "kv/kv_block_allocator.h"
#include "decode/streaming_controller.h"

void registerNativeInferenceTools(ToolRegistry& reg) {
  // 1. Tokenizer
  reg.registerTool("native_tokenize", 
    [](const JsonValue& args) -> JsonValue {
      auto text = args["text"].asString();
      auto tokens = Tokenizer::encode(text);
      return JsonValue::object({{"tokens", JsonValue::array(tokens)}});
    },
    ToolSchema{
      .inputs = {{"text", "string"}},
      .outputs = {{"tokens", "array"}},
      .required_permissions = {"internal: inference_core"}
    }
  );

  // 2. KV Allocator
  reg.registerTool("kv_alloc", 
    [](const JsonValue& args) -> JsonValue {
      int num_blocks = args["num_blocks"].asInt();
      auto blocks = KVBlockAllocator::allocate(num_blocks);
      return JsonValue::object({{"block_ids", JsonValue::array(blocks)}});
    },
    ToolSchema{
      .inputs = {{"num_blocks", "integer"}},
      .outputs = {{"block_ids", "array"}},
      .required_permissions = {"internal: inference_core"}
    }
  );

  // 3. Model Step
  reg.registerTool("model_step", 
    [](const JsonValue& args) -> JsonValue {
      auto tokens = args["tokens"].asArray<int>();
      auto kv_ref = args["kv_ref"].asArray<int>();
      auto result = ModelExecutor::step(tokens, kv_ref);
      return JsonValue::object({
        {"logits", result.logits},
        {"updated_kv", result.updated_kv}
      });
    },
    ToolSchema{
      .inputs = {{"tokens", "array"}, {"kv_ref", "array"}},
      .outputs = {{"logits", "array"}, {"updated_kv", "array"}},
      .required_permissions = {"internal: inference_core"}
    }
  );

  // 4. Grammar Compiler
  reg.registerTool("compile_grammar", 
    [](const JsonValue& args) -> JsonValue {
      auto schema = args["schema"].asObject();
      auto mask = GrammarCompiler::compile(schema);
      return JsonValue::object({{"mask", mask}});
    },
    ToolSchema{
      .inputs = {{"schema", "object"}},
      .outputs = {{"mask", "object"}},
      .required_permissions = {"internal: inference_core"}
    }
  );

  // 5. Streaming Controller
  reg.registerTool("stream_until", 
    [](const JsonValue& args) -> JsonValue {
      auto logits_provider = /* ... */;
      auto stop_condition = args["stop_condition"].asString();
      int max_tokens = args["max_tokens"].asInt();
      auto text = StreamingController::streamUntil(
        logits_provider, stop_condition, max_tokens
      );
      return JsonValue::object({{"text", text}});
    },
    ToolSchema{
      .inputs = {{"stop_condition", "string"}, {"max_tokens", "integer"}},
      .outputs = {{"text", "string"}},
      .required_permissions = {"internal: inference_core"}
    }
  );
}
```

> ✅ **合规依据**：2.2（适配器模式）、7.2（权限声明）

---

## 五、推理引擎专属工作流：`/app/inference/**`

### 5.1 引擎入口：`/app/inference/native_engine_v1`

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
  tokens: "memory.state.inference.tokens_{{ $.task.id }}"  # ← TTL 路径
meta:
  ttl_seconds: 300
next: "/app/inference/alloc_kv@v1"
```

> ✅ **合规依据**：5.1（TTL 仅限 `memory.state.*`）

---

## 六、开发/生产模式行为差异（8.5）

| 行为 | `dev` 模式 | `prod` 模式 |
|------|-----------|------------|
| `last_write_wins` | 允许 | 禁止 |
| 中间 Trace | 输出 logits / KV 状态 | 仅 `backend_used` |
| `expected_output` 验证 | 启用 | 禁用 |
| 权限检查 | 宽松 | 严格 |
| 上下文快照 | 启用 | 禁用 |

---

## 七、工作流示例

### 7.1 基础示例：文本生成

```agentic
AgenticDSL '/main/greet'
type: assign
assign:
  expr: "Hello"
next: "/lib/reasoning/generate_text@v1"
```

### 7.2 高级示例：结构化生成 + KV 复用 + 错误处理

```agentic
AgenticDSL '/main/solve_math'
type: assign
assign:
  expr: "解方程: x^2 + 2x + 1 = 0"
next: "/lib/reasoning/structured_generate@v1"

AgenticDSL '/lib/reasoning/fallback_to_text@v1'
type: assign
assign:
  expr: {"roots": [-1]}  # 模拟兜底
  path: "result.parsed_output"
next: "/end"
```

---

## 八、AgenticInfer 的本质超越点

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

## 九、合规性与安全性

| 规范要求 | AgenticInfer 实现 |
|--------|------------------|
| **三层架构** | ✅ 无跨层调用 |
| **标准库契约** | ✅ 所有 `/lib/reasoning/**` 带 `signature` |
| **权限最小化** | ✅ C++ 工具权限为 `internal` |
| **预算控制** | ✅ 嵌套引擎继承 `max_nodes * 0.8` |
| **可终止性** | ✅ `stream_until` 强制 `max_tokens` |
| **Trace** | ✅ 每步记录 `reasoning_evidence` + `backend_used` |
| **dev/prod 模式** | ✅ 行为差异化（8.5） |
| **TTL 管理** | ✅ 仅 `memory.state.*` 路径（5.1） |
| **错误处理** | ✅ `on_error` 跳转（7.2） |

---

## 十、总结

**AgenticInfer v1.4**：

- ✅ **完全兼容 AgenticDSL v3.7**，无需任何语义扩展  
- ✅ **通过标准 `/lib/reasoning/**` 子图暴露能力**  
- ✅ **C++ 模块仅作为 `tool_call` 实现**，注册符合 2.2 适配器模式  
- ✅ **推理流程由 `/app/inference/**` DAG 编排**  
- ✅ **本质超越传统引擎：推理即 DAG，策略即子图**

> **标语**：  
> **“AgenticInfer: Where Inference Becomes a Verifiable DAG.”**


