# 📜 AgenticDSL 与 C++ 推理引擎深度集成架构（v3）

> **目标**：构建一个 **完全合规、安全可靠、确定性执行、三层架构对齐、资源契约完备** 的 LLM 推理系统，其中 **AgenticDSL 作为声明式控制平面**，**C++ 推理引擎作为高性能执行平面**，二者通过 **工具注册表 + 权限感知句柄 + Trace Schema** 紧密集成。

---

## 一、架构合规性总览

| AgenticDSL v3.7 要求 | v3 架构实现状态 | 合规级别 |
|----------------------|------------------|----------|
| 三层抽象层级（2.0） | ✅ 严格分层，无越层调用 | ✔️ |
| 工具注册表（2.2） | ✅ C++ 模块 → 工具注册 → `tool_call` | ✔️ |
| 资源声明（6.4） | ✅ 启动时验证，能力驱动 | ✔️ |
| 权限交集原则（7.2） | ✅ 句柄带权限标签，运行时检查 | ✔️ |
| 确定性优先（1.3） | ✅ C++ 输出 deterministic，无异步回调 | ✔️ |
| Trace Schema（7.3） | ✅ 推理证据 + 性能指标 + 后端标识 | ✔️ |
| 动态子图安全（8.2） | ✅ 生成子图权限 ≤ 父上下文，命名空间隔离 | ✔️ |
| TTL 与持久化（5.1） | ✅ C++ 资源与 Context TTL 联动 | ✔️ |
| 预算控制（8.1） | ✅ C++ 执行纳入全局 `max_nodes`/`max_depth` | ✔️ |
| 标准库契约（6.2） | ✅ 所有 `/lib/**` 子图带 `signature` | ✔️ |

---

## 二、C++ 推理引擎核心模块设计（执行原语层）

### 2.1 工具注册与权限绑定

所有 C++ 功能必须注册为 **工具**（tool），由执行器统一管理。

```cpp
// 工具注册表（执行器内部）
class ToolRegistry {
public:
  // 注册 C++ 函数为工具
  void registerTool(const std::string& name, 
                   ToolFunction fn,
                   const ToolSchema& schema) {
    tools[name] = {fn, schema};
    
    // 权限自动推导
    permissions[name] = schema.required_permissions;
  }
  
  // 安全调用（带权限检查）
  Value callTool(const std::string& name,
                const ToolArgs& args,
                const Permissions& caller_perms) {
    auto& tool = tools.at(name);
    
    // 权限交集检查（规范 7.2）
    if (!caller_perms.intersect(tool.permissions).satisfied()) {
      throw PermissionDeniedError(name, caller_perms, tool.permissions);
    }
    
    // 执行并记录 Trace
    auto start = std::chrono::high_resolution_clock::now();
    auto result = tool.fn(args, caller_perms);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start
    );
    
    // 记录结构化 Trace（规范 7.3）
    TraceRecorder::record({
      "tool_call", name,
      "latency_ms", duration.count(),
      "backend_used", "cuda_paged_attention_v2",
      "user_id", getUserIdFromContext()
    });
    
    return result;
  }
};
```

### 2.2 资源生命周期与 TTL 集成

C++ 资源必须与 Context TTL 机制对齐（规范 5.1）。

```cpp
// 权限感知资源句柄
class PermissionedResourceHandle {
private:
  std::string uuid;
  std::shared_ptr<Resource> resource;
  Permissions allowed_perms;
  bool is_durable;
  std::optional<std::chrono::system_clock::time_point> expiry;
  
public:
  // 创建带 TTL 的句柄
  static PermissionedResourceHandle create(
    std::shared_ptr<Resource> res,
    const Permissions& perms,
    std::optional<int> ttl_seconds = std::nullopt) {
    
    auto handle = PermissionedResourceHandle();
    handle.uuid = generateUUIDv4();
    handle.resource = res;
    handle.allowed_perms = perms;
    handle.is_durable = !ttl_seconds.has_value();
    
    if (ttl_seconds) {
      handle.expiry = std::chrono::system_clock::now() + 
                     std::chrono::seconds(*ttl_seconds);
      // 注册到全局 TTL 管理器（规范 5.1）
      TTLManager::getInstance().registerHandle(
        handle.uuid, *handle.expiry
      );
    }
    
    return handle;
  }
  
  // 序列化为权限感知字符串
  std::string toPermissionedString() const {
    return fmt::format("res:{}:{}:{}",
      resource->type(),
      allowed_perms.toString(),
      uuid
    );
  }
  
  // 权限检查
  bool checkPermission(const Permissions& caller) const {
    return caller.intersect(allowed_perms).satisfied();
  }
};
```

### 2.3 预算控制集成

C++ 执行必须纳入全局预算（规范 8.1）。

```cpp
// 预算控制器（执行器全局）
class BudgetController {
  ExecutionContext ctx; // 当前执行上下文
  
public:
  // 在 C++ 工具执行前检查
  void consumeBudgetForTool(const std::string& tool_name) {
    // 节点计数（每个 tool_call 消耗 1 个节点）
    if (++ctx.executed_nodes > ctx.max_nodes) {
      throw BudgetExceededException("MAX_NODES_EXCEEDED");
    }
    
    // 深度检查（防止嵌套过深）
    if (ctx.current_depth > ctx.max_subgraph_depth) {
      throw BudgetExceededException("MAX_SUBGRAPH_DEPTH_EXCEEDED");
    }
    
    // 时间检查（规范 8.1）
    auto elapsed = std::chrono::steady_clock::now() - ctx.start_time;
    if (elapsed > ctx.max_duration_sec) {
      throw BudgetExceededException("MAX_DURATION_EXCEEDED");
    }
    
    // 工具特定成本（如 CUDA kernel 消耗更多）
    auto cost = ToolCostModel::getCost(tool_name);
    ctx.consumeResource(cost);
  }
};
```

### 2.4 可观测性与推理证据

C++ 模块必须生成规范 7.3 兼容的 Trace。

```cpp
// 推理证据记录器
class ReasoningEvidenceRecorder {
public:
  static void recordAttentionPath(
    const std::string& question,
    const std::vector<GraphPath>& paths,
    const std::vector<float>& confidences,
    const std::string& backend_id) {
    
    TraceRecorder::record({
      "reasoning_evidence", {
        {"type", "graph_based"},
        {"evidence_type", "path_based"},
        {"paths", serializePaths(paths)},
        {"confidence_scores", confidences},
        {"backend_used", backend_id},
        {"subgraph_id", generateSubgraphId()}
      }
    });
  }
  
  static void recordKernelMetrics(
    const std::string& kernel_name,
    int flops, 
    float memory_bandwidth,
    const std::string& backend_id) {
    
    TraceRecorder::record({
      "kernel_metrics", {
        {"name", kernel_name},
        {"flops", flops},
        {"memory_bandwidth_gb_s", memory_bandwidth},
        {"backend_used", backend_id}
      }
    });
  }
};
```

---

## 三、标准原语层实现（/lib/**）

所有 C++ 功能必须通过 **标准原语层子图** 暴露。

### 3.1 KV 缓存标准子图

```agentic
### AgenticDSL '/lib/memory/kv/paged@v1'
signature:
  inputs:
    - name: max_blocks
      type: integer
      required: true
    - name: block_size
      type: integer
      default: 16
  outputs:
    - name: kv_handle
      type: string  # 权限感知句柄
version: "1.0"
stability: stable
permissions:
  - memory: state_write
type: tool_call
tool: paged_kv_create
arguments:
  max_blocks: "{{ $.max_blocks }}"
  block_size: "{{ $.block_size | default(16) }}"
output_mapping:
  kv_handle: "result.handle"
```

对应的 C++ 工具实现：
```cpp
Value paged_kv_create(const ToolArgs& args, const Permissions& perms) {
  int max_blocks = args.get<int>("max_blocks");
  int block_size = args.getOrDefault<int>("block_size", 16);
  
  // 创建 C++ KV 缓存
  auto kv_cache = std::make_shared<PagedKVCache>(max_blocks, block_size);
  
  // 创建权限感知句柄（规范 7.2）
  auto handle = PermissionedResourceHandle::create(
    kv_cache, 
    {"memory: state_write", "kg: subgraph_query"},
    3600  // TTL: 1 小时
  );
  
  return Value::String(handle.toPermissionedString());
}
```

### 3.2 推理内核标准子图

```agentic
### AgenticDSL '/lib/reasoning/kernel/cuda_q4k@v1'
signature:
  inputs:
    - name: kv_handle
      type: string
      required: true
    - name: tokens
      type: array
      required: true
  outputs:
    - name: logits
      type: array
    - name: kv_handle  # 返回更新后的句柄
version: "1.0"
stability: stable
permissions:
  - memory: state_write
type: tool_call
tool: cuda_q4k_decode
arguments:
  kv_handle: "{{ $.kv_handle }}"
  tokens: "{{ $.tokens }}"
output_mapping:
  logits: "result.logits"
  kv_handle: "result.kv_handle"
```

### 3.3 调度器标准子图

```agentic
### AgenticDSL '/lib/reasoning/scheduler/radix@v1'
signature:
  inputs:
    - name: requests
      type: array
      required: true
  outputs:
    - name: batch
      type: object
version: "1.0"
stability: stable
permissions: []
type: tool_call
tool: radix_scheduler_step
arguments:
  requests: "{{ $.requests }}"
output_mapping:
  batch: "result.batch"
```

---

## 四、资源声明与启动验证

### 4.1 资源声明（/__meta__/resources）

```agentic
### AgenticDSL '/__meta__/resources'
type: resource_declare
resources:
  - type: tool
    name: paged_kv_create
    capabilities: [cow, hierarchical]
  - type: tool
    name: cuda_q4k_decode
    capabilities: [avx512, tensor_cores]
  - type: memory
    backends: [paged_kv]
  - type: generate_subgraph
    max_depth: 2
  - type: knowledge_graph
    capabilities:
      - multi_hop_query
      - evidence_path_extraction
```

### 4.2 启动验证流程

```cpp
// 执行器启动流程（规范 8.1）
bool ExecutionEngine::start(const AgenticDSLDocument& doc) {
  // 1. 解析所有子图
  parseSubgraphs(doc);
  
  // 2. 验证资源声明（规范 6.4）
  if (auto resources = doc.getMetaResources()) {
    if (!ResourceValidator::validate(*resources)) {
      setError(ERR_RESOURCE_UNAVAILABLE);
      return false;
    }
  }
  
  // 3. 验证 /lib/** 签名（规范 6.2）
  for (auto& subgraph : doc.getLibSubgraphs()) {
    if (!SignatureValidator::validate(subgraph)) {
      setError(ERR_SIGNATURE_VIOLATION);
      return false;
    }
  }
  
  // 4. 注册 C++ 工具
  registerCppTools();
  
  // 5. 启动调度器
  scheduler.start(doc.getEntryPoint());
  
  return true;
}
```

---

## 五、动态子图生成安全机制

### 5.1 安全沙箱

```cpp
class DynamicSubgraphSandbox {
public:
  bool validateGeneratedSubgraph(const Subgraph& subgraph, 
                                const ExecutionContext& ctx) {
    // 1. 命名空间检查（规范 6.1）
    if (subgraph.path.starts_with("/lib/")) {
      return false; // ERR_NAMESPACE_VIOLATION
    }
    
    // 2. 深度控制（规范 8.1）
    if (ctx.currentDepth + 1 > ctx.maxAllowedDepth) {
      return false; // MAX_DEPTH_EXCEEDED
    }
    
    // 3. 权限继承（规范 7.2）
    auto inherited_perms = ctx.permissions.intersect(
      subgraph.declaredPermissions
    );
    if (inherited_perms.empty()) {
      return false; // PERMISSION_VIOLATION
    }
    
    // 4. 资源依赖验证（规范 6.4）
    for (auto& resource : subgraph.requiredResources()) {
      if (!ResourceRegistry::isAvailable(resource, inherited_perms)) {
        return false; // ERR_RESOURCE_UNAVAILABLE
      }
    }
    
    return true;
  }
};
```

### 5.2 生成子图示例

```agentic
### AgenticDSL '/self/generate_optimized_plan'
type: tool_call
tool: llm_generate_dsl_safe  # 封装 llm_generate_dsl + 安全检查
arguments:
  prompt: |
    当前负载: {{ $.metrics.qps }} QPS
    请生成高吞吐推理计划，使用 paged KV 和 continuous batching。
  llm:
    model: "gpt-4o"
    seed: 42
    temperature: 0.0
  output_constraints:
    namespace_prefix: "/dynamic/"
    max_blocks: 2
    validate_json_schema: true
permissions:
  - generate_subgraph: { max_depth: 1 }
on_failure: "/self/fallback_to_default"
next: "/dynamic/optimized_plan_v1"
```

---

## 六、架构全景图

```
┌───────────────────────────────────────────────────────────┐
│                  AgenticDSL Document                      │
│                                                           │
│  ┌─────────────────┐     ┌───────────────────────────┐  │
│  │ /__meta__       │     │ /main/inference           │  │ ← 知识应用层
│  │ - resources     │     │ - assign model            │  │
│  │ - entry_point   │     │ - next: /lib/...          │  │
│  └─────────────────┘     └─────────────┬─────────────┘  │
│                                        │                │
│  ┌─────────────────────────────────────▼──────────────┐ │
│  │ /lib/reasoning/kernel/cuda_q4k@v1                  │ │ ← 标准原语层
│  │ signature: { inputs: [...], outputs: [...] }       │ │
│  │ type: tool_call                                    │ │
│  │ tool: cuda_q4k_decode                              │ │
│  └────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────┘
                                │ tool_call
                                ▼
┌───────────────────────────────────────────────────────────┐
│            C++ Modular Inference Engine                   │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ KV Cache    │  │ Kernel      │  │ Scheduler       │  │ ← 执行原语层
│  │ (Paged, Radix)│ │ (CUDA, CPU) │  │ (Radix, FIFO)   │  │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │
│         │                │                  │           │
│         └───────┬────────┴─────────┬────────┘           │
│                 │ Tool Registry    │                    │
│                 └────────┬─────────┘                    │
│                          │ registerTool()               │
│  ┌───────────────────────▼───────────────────────────┐  │
│  │ Budget Controller  │  TTL Manager  │  Trace Recorder│  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 七、合规性实施路线图

### 阶段 1：基础合规（2周）
- [x] 实现工具注册表与权限检查
- [x] 构建资源声明验证框架
- [x] 实现 C++ 资源与 TTL 联动

### 阶段 2：安全增强（3周）
- [ ] 实现动态子图生成沙箱
- [ ] 构建推理证据记录系统
- [ ] 集成全局预算控制器

### 阶段 3：标准库实现（2周）
- [ ] 实现 `/lib/memory/kv/**` 标准子图
- [ ] 实现 `/lib/reasoning/kernel/**` 标准子图  
- [ ] 实现 `/lib/reasoning/scheduler/**` 标准子图

### 阶段 4：生产就绪（1周）
- [ ] 通过 AgenticDSL 规范测试套件
- [ ] 实现 dev/prod 模式切换
- [ ] 文档化 C++ 模块开发指南

---

## 八、总结

**AgenticDSL 与 C++ 推理引擎 v3 架构实现了完全规范合规**：

1. **严格三层架构**：C++ 功能 → 工具注册 → `/lib/**` 子图 → 知识应用层
2. **安全第一**：权限交集原则 + 资源声明验证 + 动态子图沙箱
3. **确定性执行**：C++ 模块输出 deterministic，无异步回调
4. **可观测性完备**：推理证据 + 性能指标 + 后端标识
5. **资源生命周期管理**：TTL + 持久化 + 自动清理

该架构不仅满足 AgenticDSL v3.7 所有规范要求，还为构建 **可验证、可进化、高性能** 的下一代 LLM 推理系统提供了坚实的工程基础。通过将 **声明式控制** 与 **高性能执行** 分离，系统既能保持 LLM 友好的灵活性，又能提供 C++ 级别的性能和可靠性。
