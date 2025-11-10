# AgenticDSL 驱动的推理引擎 v4.0 架构设计文档  
**——声明式 DAG 语义驱动、模块化 C++ 执行原语、完全合规且面向演进的新一代推理系统**

---

## 一、设计目标与核心理念

### 1.1 核心理念（对齐 AgenticDSL v3.7）
> “优秀的 DSL 不是让机器更容易执行，而是让人类更容易表达意图，同时让机器能够可靠地验证这种意图。”

- **DAG 是意图表达层**：开发者/LLM 仅需组合标准子图（`/lib/**`），无须关心底层实现
- **C++ 是语义执行层**：将声明语义（如“子图查询”）映射为高性能、安全、确定性执行
- **后端完全不可见**：DAG 不暴露、不依赖任何具体执行后端（如 Qdrant、vLLM、Graphiti）

### 1.2 核心目标
| 维度 | 目标 |
|------|------|
| **合规性** | 严格满足 AgenticDSL v3.7（尤其三层架构、权限交集、资源声明、`/lib/dslgraph/generate@v1` 唯一入口） |
| **性能** | 通过子图语义缓存 + DAG 感知调度，实现 **3–5× 吞吐提升**（vs. 纯 token 缓存系统） |
| **灵活性** | C++ 执行原语完全模块化，支持热插拔、多后端适配、能力声明式选型 |
| **可演进性** | 为未来 Grammar Compiler、自进化闭环预留接口，平滑过渡至 v5.0 混合架构 |

---

## 二、整体架构概览

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          AgenticDSL v4.0 推理引擎                             │
├───────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    DAG 声明层（意图表达）                               │  │
│  │  • 完整 DAG 定义（/main/**, /lib/**, /dynamic/**）                      │  │
│  │  • /__meta__/resources 声明（能力契约，非后端绑定）                     │  │
│  │  • /__meta__ 元信息（entry_point, budget, mode）                        │  │
│  │  ✅ DAG 仅使用标准库子图（/lib/**）                                     │  │
│  │  ✅ LLM 生成必须通过 /lib/dslgraph/generate@v1                         │  │
│  └───────────────────────┬─────────────────────────────────────────────────┘  │
│                          │                                                    │
│  ┌───────────────────────▼─────────────────────────────────────────────────┐  │
│  │                    标准原语层（/lib/**）                                │  │
│  │  • 路径规范：/lib/dslgraph/**, /lib/reasoning/**, /lib/memory/** 等     │  │
│  │  • 所有子图必须含 signature（规范 6.2）                                │  │
│  │  • 语义契约化：inputs/outputs/schema/version/stability                 │  │
│  │  • 权限最小化：permissions（如 kg: subgraph_query）                    │  │
│  │  ✅ 禁止暴露后端实现细节                                                │  │
│  └───────────────────────┬─────────────────────────────────────────────────┘  │
│                          │                                                    │
│  ┌───────────────────────▼─────────────────────────────────────────────────┐  │
│  │                    C++ 执行原语层（语义实现机）                         │  │
│  │  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐           │  │
│  │  │ 工具注册表   │   │ 节点调度器    │   │ Trace 与可观测性   │           │  │
│  │  └──────┬──────┘   └──────┬───────┘   └─────────┬─────────┘           │  │
│  │         │                 │                     │                     │  │
│  │  ┌──────▼─────────────────▼─────────────────────▼──────────────┐      │  │
│  │  │                  核心执行上下文管理                         │      │  │
│  │  │  • Context 合并策略（error_on_conflict/default）           │      │  │
│  │  │  • 字段级 TTL（memory.state.*）                           │      │  │
│  │  │  • 快照机制（静态键限制，规范 11.1）                      │      │  │
│  │  └──────┬─────────────────┬─────────────────────┬──────────────┘      │  │
│  │         │                 │                     │                     │  │
│  │  ┌──────▼──────┐   ┌──────▼──────┐     ┌────────▼────────┐           │  │
│  │  │ 预算控制器   │   │ 权限管理器   │     │ 子图语义缓存器    │           │  │
│  │  └─────────────┘   └─────────────┘     └─────────────────┘           │  │
│  │         │                 │                     │                     │  │
│  │  ┌──────▼─────────────────▼─────────────────────▼──────────────┐      │  │
│  │  │                  两级 KV 缓存系统                           │      │  │
│  │  │  • L0: CPU Radix 树（子图级语义缓存）                      │      │  │
│  │  │  • L1: GPU Paged KV Cache（token 级前缀缓存）              │      │  │
│  │  └────────────────────────────────────────────────────────────┘      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 三、关键模块设计（严格对齐规范）

### 3.1 DAG 声明层：纯语义，零后端耦合

- **唯一入口**：`/lib/dslgraph/generate@v1`（规范 10.1，附录 A2）
  - 禁止 DAG 直接使用 `llm_generate_dsl`
  - 禁止生成子图包含 `/lib/**` 写入
- **资源声明**：`/__meta__/resources`（规范 6.4）
  ```agentic
  resources:
    - type: knowledge_graph
      capabilities: [multi_hop_query, evidence_path_extraction]
    - type: generate_subgraph
      max_depth: 2
  ```
  - **能力导向**：声明 `evidence_path_extraction`，而非 `backend: graphiti`
  - **执行器自由选型**：可选择 GFM-RAG、Neo4j、自研图引擎，只要满足能力

### 3.2 标准原语层：契约化接口，稳定演进

#### 核心子图清单（必须实现，附录 C）
| 路径 | 职责 | 权限 |
|------|------|------|
| `/lib/dslgraph/generate@v1` | 安全动态子图生成 | `generate_subgraph` |
| `/lib/reasoning/try_catch@v1` | 异常回溯 | 无 |
| `/lib/memory/kg/query_subgraph@v1` | 图子图查询 | `kg: subgraph_query` |
| `/lib/memory/vector/recall@v1` | 语义记忆检索 | `vector: recall` |

- **`signature` 强制校验**（规范 6.2）：
  ```yaml
  signature:
    inputs:
      - name: query
        type: string
        required: true
    outputs:
      - name: memories
        type: array
        schema: { ... }
    version: "1.0"
    stability: stable
  ```

### 3.3 C++ 执行原语层：语义实现机

#### 3.3.1 工具注册表（ToolRegistry）
- **注册语义能力，非后端**：
  ```cpp
  // 执行器内部
  tool_registry.registerTool("vector_recall", 
    qdrant_recall_impl, 
    ToolSchema{
      .required_permissions = {"vector: recall"},
      .capabilities = {"dense", "metadata_filter"}
    });
  ```
- **权限交集原则**（规范 7.2）：
  - 调用时检查：`节点权限 ∩ 父上下文权限`
  - 未满足 → 跳转 `on_error`

#### 3.3.2 子图语义缓存器（SubgraphSemanticCache）
- **缓存键**（规范语义，非后端）：
  ```cpp
  struct SubgraphCacheKey {
    std::string path;           // e.g. "/lib/memory/kg/query_subgraph"
    uint64_t context_hash;      // 输入变量哈希（基于 signature.inputs）
    uint64_t signature_hash;    // 子图签名哈希
    std::string engine_ver;
  };
  ```
- **模糊匹配**：支持变量重绑定（符合上下文模型，规范 4.1）
- **驱逐策略**：LRU + 语义热度（`/lib/reasoning/assert` 高优先级）

#### 3.3.3 两级 KV 缓存系统
| 层级 | 用途 | 对 DAG 可见性 |
|------|------|----------------|
| **L0** | 子图语义缓存 | ❌ 不可见（执行器内部优化） |
| **L1** | Token 前缀缓存 | ❌ 不可见（兼容 vLLM，但抽象化） |

- **缓存复用**：命中 L0 时，直接返回 AST + 复用 L1 KV Cache
- **降级路径**：L0 未命中 → 正常执行 → 结果写入 L0

#### 3.3.4 DAG 感知调度器
- **分支识别**：为 `fork.branches` 打标 `dag_id`
- **前缀共享**：同一 `dag_id` 分支自动共享 prompt 前缀（L1 优化）
- **依赖解析**：入队前解析 `wait_for`（规范 8.3）

---

## 四、合规性保障机制

### 4.1 三层架构隔离（规范 2.1）
| 违规行为 | 防御机制 |
|----------|----------|
| 知识应用层直接调用执行原语 | 启动时解析校验，拒绝非法 DAG |
| LLM 直接生成 `/lib/**` | `llm_generate_dsl` 强制 `namespace_prefix: "/dynamic/"` |
| 未声明资源调用工具 | 启动时验证 `/__meta__/resources`，缺失 → `ERR_RESOURCE_UNAVAILABLE` |

### 4.2 动态子图生成安全（规范 8.2, 10.1）
- **唯一入口**：`/lib/dslgraph/generate@v1`
- **权限继承**：生成的 `/dynamic/...` 权限 ≤ 父上下文
- **禁止行为**：LLM 生成的子图不得包含 `/lib/**` 写入

### 4.3 Trace 与可观测性（规范 7.3）
- **后端标识仅用于 Trace**：
  ```json
  {
    "memory_op_type": "kg_query",
    "backend_used": "gfm-retriever-v1",  // 仅调试，非控制逻辑
    "latency_ms": 15
  }
  ```
- **兼容 OpenTelemetry**：预算快照、上下文变更、推理证据

---

## 五、模块化设计与演进路线

### 5.1 模块化 C++ 框架
- **工具热插拔**：
  ```cpp
  void loadToolPlugin(const std::string& plugin_path) {
    auto handle = dlopen(plugin_path.c_str(), RTLD_LAZY);
    auto register_fn = (RegisterFn*)dlsym(handle, "register_tools");
    register_fn(tool_registry);
  }
  ```
- **能力驱动选型**：执行器根据 `/__meta__/resources` 声明的能力，动态选择满足条件的工具实现

### 5.2 演进路线图

| 阶段 | 版本 | 关键能力 | 合规性 |
|------|------|----------|--------|
| **基础版** | v4.0 | 三层架构 + 子图语义缓存 + 两级 KV | ✅ 完全合规 |
| **性能增强** | v4.5 | Grammar Compiler（作为 `/lib/dslgraph/generate@v1` 内部实现） | ✅ 合规 |
| **混合架构** | v5.0 | Branch-Aware Batching + 自进化闭环 | ✅ 合规 |
| **Agentic-Native** | v6.0 | 原生图神经网络模型 | 需新规范 |

> **演进原则**：所有新能力必须通过 `/lib/**` 标准接口暴露，绝不破坏 DAG 声明层的纯语义性。

---

## 六、总结

**AgenticDSL v4.0 推理引擎**是一个 **声明式 DAG 语义驱动、模块化 C++ 执行原语、完全合规** 的新一代推理系统：

- **DAG 层**：纯语义表达，零后端耦合，严格通过 `/lib/**` 组合
- **执行层**：C++ 实现语义能力，模块化、热插拔、能力驱动选型
- **创新点**：子图语义缓存（L0） + DAG 感知调度，性能超越现有 token 缓存系统
- **演进性**：为 Grammar Compiler、自进化闭环预留接口，平滑过渡

该设计 **严格遵循 AgenticDSL v3.7 核心哲学**，同时为构建 **可验证、可进化、高性能** 的智能体生态奠定坚实工程基础。
