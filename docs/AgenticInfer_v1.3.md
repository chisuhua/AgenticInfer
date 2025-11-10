# **AgenticInfer：Agentic-native 推理引擎架构设计文档 v1.3**  
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

根据 AgenticDSL v3.7 **附录 C** 与 **10.2 推理原语**，新增以下 5 个标准子图（均带 `signature`）：

> 完整 YAML 实现已在前文提供，此处省略，仅列清单：

| 子图 | 稳定性 | 权限 |
|------|--------|------|
| `/lib/reasoning/generate_text@v1` | stable | `reasoning: lmm_generate` |
| `/lib/reasoning/structured_generate@v1` | stable | `reasoning: structured_generate` |
| `/lib/reasoning/continue_from_kv@v1` | stable | `reasoning: lmm_generate` |
| `/lib/reasoning/stream_until@v1` | stable | `reasoning: stream_output` |
| `/lib/reasoning/speculative_decode@v1` | experimental | `reasoning: speculative_decode` |

---

## 四、C++ 执行原语层模块架构

AgenticInfer 的 **C++ 推理核心** 完全自研，模块化解耦，**不依赖 llama.cpp / vLLM / SGLang**。

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
├── kernels/                 # CUDA kernels（paged_attention.cu, fused_mlp.cu）
└── tests/
```

---

### 4.2 核心 C++ 模块设计

#### 1. **`ModelLoader`（模型加载器）**

```cpp
class ModelLoader {
public:
  static std::unique_ptr<TransformerModel> loadFromGGUF(const std::string& path);
  static std::unique_ptr<TransformerModel> loadFromSafetensors(const std::string& path);
};
```

- 支持 GGUF / Safetensors 直接解析
- 不依赖 PyTorch / Transformers

#### 2. **`ModelExecutor`（模型执行器）**

```cpp
class ModelExecutor {
  std::unique_ptr<TransformerModel> model_;
  std::shared_ptr<KVBlockAllocator> kv_allocator_;

public:
  InferenceStepResult step(
    const std::vector<int>& input_tokens,
    const PageTableRef& kv_cache,
    const LogitsMask* mask = nullptr
  );
};
```

- 手写 CUDA kernel：`paged_attention`, `fused_mlp`
- 支持 Q4_K / Q5_K / F16 量化

#### 3. **`KVBlockAllocator`（分页 KV 管理器）**

```cpp
class KVBlockAllocator {
  std::vector<GPUPage> physical_pages_;
  std::queue<int> free_pages_;

public:
  PageTableRef allocate(size_t num_blocks);
  void free(const PageTableRef& ref);
  void sharePrefix(const PageTableRef& src, PageTableRef& dst, int shared_len);
};
```

- 兼容 vLLM PagedAttention 格式
- 支持 Copy-on-Write（COW）

#### 4. **`RadixPrefixIndex`（前缀共享索引）**

```cpp
class RadixPrefixIndex {
  struct RadixNode {
    std::map<int, std::unique_ptr<RadixNode>> children;
    std::optional<PageTableRef> kv_ref;
    int ref_count = 0;
  };

public:
  int registerPrefix(const std::vector<int>& tokens, const PageTableRef& kv);
  std::pair<int, PageTableRef> findLongestPrefix(const std::vector<int>& tokens);
};
```

- 实现 SGLang 的 RadixAttention 语义
- 与 DAG `fork` 分支自动绑定

#### 5. **`GrammarCompiler`（结构化输出约束）**

```cpp
class GrammarCompiler {
public:
  static LogitsMask compile(const nlohmann::json& schema);
};
```

- 将 JSON Schema → Context-Free Grammar → Logits Mask
- 支持嵌套对象、数组、枚举

#### 6. **`StreamingController`（流式输出控制）**

```cpp
class StreamingController {
public:
  std::string streamUntil(
    const std::function<std::pair<float, bool>()>& logits_provider,
    const std::string& stop_condition,
    int max_tokens
  );
};
```

- 支持字符串/正则终止条件
- 内置 `max_tokens` 保护

---

### 4.3 C++ `tool_call` 实现映射

| 工具名 | C++ 实现 | 文件 |
|--------|--------|------|
| `native_tokenize` | `Tokenizer::encode()` | `src/tools/tokenize.cpp` |
| `kv_alloc` | `KVBlockAllocator::allocate()` | `src/kv/kv_block_allocator.cpp` |
| `model_step` | `ModelExecutor::step()` | `src/model/model_executor.cpp` |
| `compile_grammar` | `GrammarCompiler::compile()` | `src/grammar/grammar_compiler.cpp` |
| `stream_until` | `StreamingController::streamUntil()` | `src/decode/streaming_controller.cpp` |

> 🔒 所有工具注册时绑定权限：`permissions: [internal: inference_core]`

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

AgenticDSL '/app/inference/run_attention@v1`
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

**AgenticInfer v1.3**：

- ✅ **完全兼容 AgenticDSL v3.7**，无需任何语义扩展  
- ✅ **通过标准 `/lib/reasoning/**` 子图暴露能力**  
- ✅ **C++ 模块仅作为 `tool_call` 实现**  
- ✅ **推理流程由 `/app/inference/**` DAG 编排**  
- ✅ **本质超越传统引擎：推理即 DAG，策略即子图**

> **标语**：  
> **“AgenticInfer: Where Inference Becomes a Verifiable DAG.”**
