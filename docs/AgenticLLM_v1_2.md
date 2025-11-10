
1. **需设计的 C++ 执行原语模块**（工具注册层）  
2. **对应的 `/lib/reasoning/**` 子图语义接口**（标准原语层）  
3. **从简单到复杂的推理工作流子图示例**（知识应用层）

所有设计严格遵循：  
✅ 三层架构隔离  
✅ `/lib/**` 必须带 `signature`  
✅ 推理语义抽象，不暴露后端细节  
✅ LLM 生成必须通过 `/lib/dslgraph/generate@v1`

---


## 三、标准原语层：`/lib/reasoning/**` 子图语义设计

这些子图**定义语义契约**，

### 3.1 基础推理原语（必需）

#### `/lib/reasoning/generate_text@v1`（stable）
```agentic
signature:
  inputs:
    - name: prompt
      type: string
      required: true
    - name: max_tokens
      type: integer
      default: 256
    - name: temperature
      type: number
      default: 0.0
    - name: seed
      type: integer
      required: true
  outputs:
    - name: text
      type: string
    - name: kv_handle  # 用于后续复用
      type: string
version: "1.0"
permissions:
  - reasoning: lmm_generate
type: tool_call
tool: llm_infer
arguments:
  prompt: "{{ $.prompt }}"
  max_tokens: "{{ $.max_tokens }}"
  temperature: "{{ $.temperature }}"
  seed: "{{ $.seed }}"
output_mapping:
  text: "result.text"
  kv_handle: "result.kv_handle"
```

#### `/lib/reasoning/structured_generate@v1`（stable）
> 利用 Grammar Compiler 实现结构化输出（SGLang 风格）
```agentic
signature:
  inputs:
    - name: prompt
      type: string
      required: true
    - name: output_schema
      type: object  # JSON Schema
      required: true
    - name: seed
      type: integer
      required: true
  outputs:
    - name: parsed_output
      type: object
version: "1.0"
permissions:
  - reasoning: structured_generate
type: tool_call
tool: grammar_guided_infer  # 内部调用 grammar_compiler + llm_infer
arguments:
  prompt: "{{ $.prompt }}"
  schema: "{{ $.output_schema }}"
  seed: "{{ $.seed }}"
output_mapping:
  parsed_output: "result.parsed"
```

### 3.2 高级推理原语（可选）

#### `/lib/reasoning/continue_from_kv@v1`（stable）
> 复用已有 KV Cache（vLLM / SGLang 能力抽象）
```agentic
signature:
  inputs:
    - name: kv_handle
      type: string
      required: true
    - name: new_tokens
      type: array
      required: true
  outputs:
    - name: continuation
      type: string
    - name: updated_kv_handle
      type: string
```

#### `/lib/reasoning/hypothesize_and_verify@v1`
> 多假设并行验证（复用 `fork` + `generate_text`）

---

## 四、推理工作流子图（知识应用层）

### 4.1 简单工作流：单步文本生成
```agentic
### AgenticDSL '/main/greet_user'
type: assign
assign:
  expr: "你好，今天感觉如何？"
  path: "prompt"
next: "/lib/reasoning/generate_text@v1?seed=42"

### AgenticDSL '/main/echo_response'
type: assign
assign:
  expr: "{{ $.text }}"
  path: "response"
next: "/end"
```

### 4.2 中等工作流：结构化问答（利用图记忆）
```agentic
### AgenticDSL '/main/answer_geography'
type: assign
assign:
  expr: {
    question: "北京的首都是哪里？",
    kg_query: { start_entities: ["Beijing"], query_path: "(?x)-[capital_of]->(?y)" }
  }
next: "/lib/memory/kg/query_subgraph@v1"

### AgenticDSL '/main/generate_answer'
type: assign
assign:
  expr: "根据知识图谱，{{ $.subgraph.edges[0].tail }} 是北京的首都。"
  path: "prompt"
next: "/lib/reasoning/generate_text@v1?seed=42"
```

### 4.3 复杂工作流：IPER 闭环（意图-计划-执行-反思）
```agentic
### AgenticDSL '/main/solve_math_task'
type: assign
assign:
  expr: "解方程: x^2 + 2x + 1 = 0"
  path: "user_intent"
next: "/lib/reasoning/iper_loop@v1"

# /lib/reasoning/iper_loop@v1 内部逻辑：
# 1. call /lib/dslgraph/generate@v1 → /dynamic/plan_v1
# 2. execute /dynamic/plan_v1
# 3. if fail → reflect → regenerate plan
# 4. max_reflections=3
```

### 4.4 高级工作流：自进化 + 语义缓存
```agentic
### AgenticDSL '/main/learn_from_success'
type: generate_subgraph
prompt_template: "基于成功 Trace {{ $.trace_id }}，生成通用求解子图"
next: "/lib/dslgraph/generate@v1"

### AgenticDSL '/main/archive_pattern'
type: assign
assign:
  expr: "/lib/solved/quadratic@v1"
  path: "archive_path"
next: "/lib/dslgraph/archive_to@v1"

# 后续相同任务将命中子图语义缓存（L0）
```

---

## 五、关键技术整合点

| 能力 | 如何抽象 | 对应模块 |
|------|----------|----------|
| **前缀共享** | 通过 `radix_prefix_register` + `llm_infer` 的 `shared_prefix_id` | SGLang 优势 |
| **内存高效** | 通过 `paged_kv_manager` + `kv_handle` 复用 | vLLM 优势 |
| **量化推理** | `llm_infer` 自动选择 llama.cpp CPU/GPU 混合路径 | llama.cpp 优势 |
| **结构化生成** | `grammar_compiler` 将 JSON Schema → CFG/Logits Mask | SGLang + 扩展 |
| **语义缓存** | `subgraph_semantic_cache` 缓存 `(path, ctx_hash, sig_hash)` | 全新能力 |

---

## 六、总结：构建超越三者的 Agentic 原生推理引擎

| 维度 | 现有后端 | AgenticDSL v4.0 |
|------|----------|------------------|
| **缓存粒度** | Token 级 | **子图语义级 + Token 级** |
| **生成控制** | 有限 logits / CFG | **DAG 语义 + Grammar Compiler** |
| **调度单位** | 请求 / Token | **DAG 节点 / 分支** |
| **可验证性** | 无 | **Trace + assert + expected_output** |
| **演进性** | 模型为中心 | **子图为中心（可 archive_to / 复用）** |

> **核心突破**：不再是对 llama.cpp / vLLM 的封装，而是构建 **以 AgenticDSL 语义为核心的推理原生引擎**，将三者能力吸收为**执行策略选项**，而非架构依赖。

此设计完全兼容 AgenticDSL v3.7，同时为未来 Grammar Native、图神经网络推理等演进预留空间。
