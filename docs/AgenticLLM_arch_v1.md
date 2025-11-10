下面重新设计 **AgenticDSL Native 推理引擎 v4.0** 的 C++ 架构：

---

# AgenticDSL Native 推理引擎 v4.0  
**——原生语义驱动、模块化解耦、完全自研的推理核心**

---

## 一、设计原则

1. **不依赖任何外部推理后端**（llama.cpp / vLLM / SGLang 等）
2. **推理即 DAG 节点**：每个推理动作是一个可调度、可缓存、可 Trace 的 DAG 节点
3. **语义优先**：推理能力通过 `/lib/reasoning/**` 子图契约化，而非 API
4. **模块化内核**：将传统 LLM 推理解构为 KV 管理、前缀共享、解码策略、缓存复用等独立模块
5. **预算 & 权限内嵌**：推理执行天然受全局预算与权限约束

---

## 二、推理核心解构：关键模块清单

| 模块 | 职责 | 是否暴露为工具 | 对应子图 |
|------|------|----------------|----------|
| **TokenEncoder** | 文本 ↔ Token ID 转换 | 否 | 内部 |
| **ModelExecutor** | 执行单步 forward（含 attention + FFN） | 否 | 内部 |
| **KVBlockAllocator** | 分配/释放 Paged KV Blocks | 否 | 内部 |
| **RadixPrefixIndex** | 管理全局 Token 前缀共享树 | 否 | 内部 |
| **DecoderStrategy** | 解码策略（greedy / sampling / speculative） | 是 | `/lib/reasoning/decode_with@v1` |
| **InferenceScheduler** | DAG 节点级推理调度器 | 否 | 内部 |
| **SubgraphKVCache** | 子图语义级 KV 缓存（L0） | 是 | 内部优化，不暴露 |
| **GrammarEnforcer** | 结构化输出约束（JSON Schema → logits mask） | 是 | `/lib/reasoning/structured_generate@v1` |
| **StreamingController** | 流式输出 + 条件终止 | 是 | `/lib/reasoning/stream_until@v1` |
| **SpeculativeRunner** | 推测解码（小模型 draft + 大模型 verify）| 是 | `/lib/reasoning/speculative_decode@v1` |

---

## 三、核心 C++ 模块架构设计

### 3.1 `ModelExecutor`（模型执行器）
```cpp
// 纯 C++ 实现，不依赖 llama.cpp/vLLM
class ModelExecutor {
  std::unique_ptr<TransformerModel> model_;
  std::shared_ptr<KVBlockAllocator> kv_allocator_;

public:
  // 执行单步推理（1 token）
  InferenceStepResult step(
    const std::vector<int>& input_tokens,
    const PageTableRef& kv_cache,
    const LogitsMask* mask = nullptr
  );
};
```
- **模型加载**：支持 GGUF / Safetensors 直接解析
- **内核优化**：手写 CUDA kernel（paged attention, fused MLP）
- **无框架依赖**：不链接 PyTorch / Transformers

### 3.2 `KVBlockAllocator`（分页 KV 管理器）
```cpp
struct PageTable {
  std::vector<int> block_ids;        // 逻辑块 → 物理页
  std::map<int, int> ref_counts;     // 页引用计数
};

class KVBlockAllocator {
  std::vector<GPUPage> physical_pages_;
  std::queue<int> free_pages_;

public:
  PageTableRef allocate(size_t num_blocks);
  void free(const PageTableRef& ref);
  void sharePrefix(const PageTableRef& src, PageTableRef& dst, int shared_len);
};
```
- **兼容 vLLM PagedAttention** 格式，但**自研实现**
- **支持 COW（Copy-on-Write）**：共享页修改时自动复制

### 3.3 `RadixPrefixIndex`（前缀共享索引）
```cpp
struct RadixNode {
  std::map<int, std::unique_ptr<RadixNode>> children;
  std::optional<PageTableRef> kv_ref;  // 指向共享 KV 前缀
  int ref_count = 0;
};

class RadixPrefixIndex {
  std::unique_ptr<RadixNode> root_;

public:
  // 注册新前缀，返回共享长度
  int registerPrefix(const std::vector<int>& tokens, const PageTableRef& kv);
  
  // 查询最长匹配前缀
  std::pair<int, PageTableRef> findLongestPrefix(const std::vector<int>& tokens);
};
```
- **实现 SGLang 的 RadixAttention 语义，但自研**
- **与 DAG 节点绑定**：每个 `fork` 分支自动注册前缀

### 3.4 `InferenceScheduler`（推理调度器）
```cpp
class InferenceScheduler {
  std::queue<InferenceTask> ready_queue_;   // DAG 节点 → 推理任务
  std::shared_ptr<ModelExecutor> executor_;
  std::shared_ptr<RadixPrefixIndex> prefix_index_;
  std::shared_ptr<KVBlockAllocator> kv_allocator_;

public:
  void submitTask(const Node& node);  // DAG 节点提交
  void run();                         // 执行主循环
};
```
- **DAG 节点 → 推理任务**：每个 `generate_text` 节点生成一个任务
- **自动批处理**：合并相同模型、相邻 prompt 的任务
- **预算感知**：超预算任务直接拒绝

---

## 四、对外暴露的标准子图（`/lib/reasoning/**`）

所有推理能力**必须通过子图暴露**，不可直接调用 C++ 函数。

### 4.1 `/lib/reasoning/generate_text@v1`
```agentic
signature:
  inputs:
    - name: prompt
    - name: max_tokens
    - name: seed
    - name: temperature
    - name: decoder_strategy  # "greedy", "sampling", "speculative"
  outputs:
    - name: text
    - name: kv_handle        # 可用于 continue_from_kv
type: tool_call
tool: native_infer       # 调用 InferenceScheduler
```

### 4.2 `/lib/reasoning/structured_generate@v1`
```agentic
signature:
  inputs:
    - name: prompt
    - name: output_schema    # JSON Schema
  outputs:
    - name: parsed_output
type: tool_call
tool: grammar_enforced_infer  # GrammarEnforcer + native_infer
```

### 4.3 `/lib/reasoning/speculative_decode@v1`
```agentic
signature:
  inputs:
    - name: prompt
    - name: draft_model_id   # 小模型 ID（如 "phi-3-mini-gguf"）
    - name: max_speculative_tokens
  outputs:
    - name: verified_output
type: tool_call
tool: speculative_runner    # SpeculativeRunner
```

---

## 五、C++ 项目总体架构

```
agentic-native-inference/
├── CMakeLists.txt
├── include/
│   ├── model/               # ModelExecutor, TransformerModel
│   ├── scheduler/           # InferenceScheduler, InferenceTask
│   ├── kv/                  # KVBlockAllocator, PageTable, GPUPage
│   ├── prefix/              # RadixPrefixIndex, RadixNode
│   ├── decode/              # DecoderStrategy, SpeculativeRunner
│   ├── grammar/             # GrammarEnforcer, JSONSchemaParser
│   └── tools/               # 所有 tool_call 实现（native_infer, ...）
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
    ├── model/               # 模型加载/推理正确性
    └── scheduler/           # DAG 节点调度、批处理
```

---

## 六、与 AgenticDSL 运行环境集成

### 6.1 工具注册（唯一对外接口）
```cpp
void registerNativeInferenceTools(ToolRegistry& reg) {
  reg.registerTool("native_infer", 
    &NativeInferenceTool::infer, 
    ToolSchema{...});
  
  reg.registerTool("grammar_enforced_infer", 
    &GrammarEnforcerTool::infer, 
    ToolSchema{...});
  
  reg.registerTool("speculative_runner", 
    &SpeculativeRunnerTool::run, 
    ToolSchema{...});
}
```

### 6.2 启动流程
```cpp
int main() {
  auto doc = parseAgenticDSL(...);
  auto engine = std::make_unique<ExecutionEngine>();
  
  // 1. 加载模型（GGUF/Safetensors）
  auto model = ModelExecutor::loadModel("model.gguf");
  
  // 2. 初始化推理核心
  auto kv_alloc = std::make_shared<KVBlockAllocator>(model);
  auto prefix_index = std::make_shared<RadixPrefixIndex>();
  auto scheduler = std::make_shared<InferenceScheduler>(model, kv_alloc, prefix_index);
  
  // 3. 注册工具
  registerNativeInferenceTools(engine->getToolRegistry());
  
  // 4. 启动
  engine->start(doc);
}
```

---

## 七、关键优势

| 维度 | 传统方案（封装 vLLM 等） | 本设计（原生推理核心） |
|------|-------------------------|------------------------|
| **控制粒度** | 请求级 | **DAG 节点级** |
| **缓存单位** | Token 前缀 | **子图语义 + Token 前缀** |
| **调度单位** | 请求 batch | **DAG 分支感知 batch** |
| **内存效率** | vLLM PagedAttention | **自研 + DAG 上下文感知释放** |
| **扩展性** | 受限于后端 API | **任意解码策略、任意约束** |
| **合规性** | 难以保证 | **天然集成预算/权限/Trace** |

---

## 八、总结

本设计完全**摒弃对现有推理后端的依赖**，构建了一个：

- ✅ **原生 Agentic 感知**：推理即 DAG 节点
- ✅ **模块化解耦**：KV、前缀、解码、调度独立
- ✅ **完全自研**：从模型加载到 CUDA kernel 全自控
- ✅ **规范合规**：所有能力通过 `/lib/reasoning/**` 契约化

这不仅是一个推理引擎，更是 **AgenticDSL 语义在执行层的原生实现**，为构建真正可验证、可进化、高性能的智能体系统奠定工业级基础。
