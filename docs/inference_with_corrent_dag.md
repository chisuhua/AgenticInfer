# 保证Markdown-YAML计算图DSL的语义正确性

**核心答案**：对于Markdown中嵌入YAML定义计算图DAG的DSL，**语义保证需要多层验证机制**：语法约束 + DAG结构验证 + 类型系统 + 执行验证 + 文档一致性检查。单一模型无法保证语义正确性，必须结合外部验证工具链。

## 1. DSL结构与语义挑战分析

### 1.1 典型DSL结构示例
```markdown
# 数据预处理与特征工程管道

该管道从原始数据开始，执行清洗、特征提取，最终输出训练集。

```yaml
dag:
  name: data_preprocessing_pipeline
  version: 1.2
  description: "处理用户行为数据，提取时序特征"
  
  nodes:
    - id: raw_data_loader
      type: data_source
      params:
        path: "s3://data-bucket/raw/user_events/"
        format: parquet
        time_range: "last_30_days"
    
    - id: data_cleaner
      type: transform
      inputs: 
        - raw_data_loader.output
      params:
        operations:
          - remove_duplicates
          - fill_missing: median
          - filter: "event_timestamp > 2023-01-01"
    
    - id: feature_extractor
      type: feature_engineering
      inputs:
        - data_cleaner.output
      params:
        features:
          - time_window_aggregates:
              windows: [1h, 24h, 7d]
              metrics: [count, avg, max]
          - session_segmentation:
              timeout: 30m
    
    - id: train_test_split
      type: splitter
      inputs:
        - feature_extractor.output
      params:
        test_size: 0.2
        random_seed: 42
        stratify_by: user_segment
  
  outputs:
    - train_set: train_test_split.train
    - test_set: train_test_split.test
```

生成的特征集将用于训练推荐模型，见[模型训练文档](model_training.md)。
```

### 1.2 语义保证的关键维度
| 验证层级 | 检查内容 | 重要性 | 验证方法 |
|---------|---------|-------|---------|
| **Markdown结构** | 代码块标记、文档结构 | 低 | 语法解析器 |
| **YAML语法** | 缩进、键值对、列表格式 | 高 | YAML解析器 |
| **DAG结构** | 无环性、连接完整性 | **关键** | 图算法验证 |
| **节点语义** | 节点类型与参数匹配 | **关键** | 模式验证 |
| **数据流** | 输入/输出类型兼容性 | **关键** | 类型系统 |
| **文档一致性** | Markdown描述与YAML一致 | 中 | 语义相似度 |

## 2. 语义保证的分层架构

### 2.1 整体架构设计
```
用户请求 → LLM生成 → 多层验证 → 修复/重生成 → 有效DAG
                │
                ├─ 1. Markdown结构验证
                ├─ 2. YAML语法验证
                ├─ 3. DAG结构验证 (无环、连接)
                ├─ 4. 节点语义验证 (类型+参数)
                ├─ 5. 数据流类型验证
                └─ 6. 文档一致性检查
```

### 2.2 关键语义验证组件实现

#### 1. DAG结构验证 (核心)
```python
import networkx as nx
from pydantic import BaseModel, validator, ValidationError
from typing import List, Dict, Any, Optional

class DAGNode(BaseModel):
    id: str
    type: str
    inputs: List[str] = []
    params: Dict[str, Any] = {}

class DAGDefinition(BaseModel):
    name: str
    nodes: List[DAGNode]
    outputs: Dict[str, str] = {}
    
    @validator('nodes')
    def validate_dag_structure(cls, nodes):
        """验证DAG无环性和连接完整性"""
        G = nx.DiGraph()
        
        # 添加所有节点
        node_ids = {node.id for node in nodes}
        G.add_nodes_from(node_ids)
        
        # 添加边 (依赖关系)
        for node in nodes:
            for input_ref in node.inputs:
                # 提取源节点ID (input_ref格式: "node_id.output_name")
                source_node_id = input_ref.split('.')[0]
                if source_node_id not in node_ids:
                    raise ValueError(f"Node '{node.id}' references non-existent input node '{source_node_id}'")
                G.add_edge(source_node_id, node.id)
        
        # 检查环路
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G.to_undirected()))
            raise ValueError(f"DAG contains cycles: {cycles}")
        
        # 检查孤立节点 (除数据源外)
        for node in nodes:
            if node.type != "data_source" and G.in_degree(node.id) == 0:
                raise ValueError(f"Non-source node '{node.id}' has no inputs")
        
        return nodes
    
    @validator('outputs')
    def validate_outputs(cls, outputs, values):
        """验证输出引用有效"""
        if 'nodes' not in values:
            return outputs
            
        node_ids = {node.id for node in values['nodes']}
        for output_name, ref in outputs.items():
            source_node_id = ref.split('.')[0]
            if source_node_id not in node_ids:
                raise ValueError(f"Output '{output_name}' references non-existent node '{source_node_id}'")
        
        return outputs
```

#### 2. 节点类型系统与参数验证
```python
from enum import Enum
from typing import Union, Literal

class NodeType(str, Enum):
    DATA_SOURCE = "data_source"
    TRANSFORM = "transform"
    FEATURE_ENGINEERING = "feature_engineering"
    SPLITTER = "splitter"
    MODEL_TRAINER = "model_trainer"
    EVALUATOR = "evaluator"

class DataSourceParams(BaseModel):
    path: str
    format: Literal["csv", "parquet", "json", "sqlite"]
    time_range: Optional[str] = None
    sampling_rate: Optional[float] = None

class TransformParams(BaseModel):
    operations: List[Union[
        Literal["remove_duplicates", "remove_nulls"],
        Dict[Literal["fill_missing"], Literal["mean", "median", "mode", "zero"]],
        Dict[Literal["filter"], str]  # SQL-like condition
    ]]

# 为每种节点类型定义参数模型
NODE_TYPE_PARAMS = {
    NodeType.DATA_SOURCE: DataSourceParams,
    NodeType.TRANSFORM: TransformParams,
    # ... 其他节点类型
}

def validate_node_semantics(node: DAGNode):
    """验证节点类型与参数的语义一致性"""
    try:
        node_type = NodeType(node.type)
    except ValueError:
        raise ValueError(f"Invalid node type: {node.type}. Valid types: {list(NodeType)}")
    
    # 获取对应参数模型
    params_model = NODE_TYPE_PARAMS.get(node_type)
    if not params_model:
        raise ValueError(f"No parameter validation defined for node type: {node_type}")
    
    # 验证参数
    try:
        params_model(**node.params)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for node '{node.id}' of type '{node_type}': {e}")
    
    # 验证输入数量 (某些节点类型有固定输入数)
    if node_type == NodeType.SPLITTER and len(node.inputs) != 1:
        raise ValueError(f"Splitter node '{node.id}' must have exactly 1 input, got {len(node.inputs)}")
    
    return True
```

#### 3. 数据流类型系统 (高级语义)
```python
class DataType(str, Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    TIME_SERIES = "time_series"
    GRAPH = "graph"

# 节点类型到输出类型的映射
NODE_OUTPUT_TYPES = {
    NodeType.DATA_SOURCE: DataType.TABULAR,
    NodeType.TRANSFORM: DataType.TABULAR,  # 假设transform保持类型
    NodeType.FEATURE_ENGINEERING: DataType.TABULAR,
    NodeType.SPLITTER: {
        "train": DataType.TABULAR,
        "test": DataType.TABULAR
    }
}

def validate_data_flow(dag: DAGDefinition):
    """验证数据流类型兼容性"""
    # 构建节点输出类型映射
    node_output_types = {}
    
    for node in dag.nodes:
        node_type = NodeType(node.type)
        
        if node_type == NodeType.SPLITTER:
            # 特殊处理多输出节点
            base_type = NODE_OUTPUT_TYPES.get(node_type, {})
            if isinstance(base_type, dict):
                for output_name in ["train", "test"]:
                    node_output_types[f"{node.id}.{output_name}"] = base_type.get(output_name, DataType.TABULAR)
        else:
            # 标准单输出节点
            output_type = NODE_OUTPUT_TYPES.get(node_type, DataType.TABULAR)
            node_output_types[f"{node.id}.output"] = output_type
    
    # 验证输入类型兼容性
    for node in dag.nodes:
        for input_ref in node.inputs:
            if input_ref not in node_output_types:
                raise ValueError(f"Input reference '{input_ref}' not found in DAG outputs")
            
            # 简单类型兼容性检查 (实际可更复杂)
            input_type = node_output_types[input_ref]
            expected_type = get_expected_input_type(NodeType(node.type))
            
            if input_type != expected_type:
                raise ValueError(
                    f"Type mismatch for node '{node.id}': "
                    f"expected {expected_type}, got {input_type} from '{input_ref}'"
                )
    
    return True

def get_expected_input_type(node_type: NodeType) -> DataType:
    """获取节点类型期望的输入类型"""
    expectations = {
        NodeType.TRANSFORM: DataType.TABULAR,
        NodeType.FEATURE_ENGINEERING: DataType.TABULAR,
        NodeType.SPLITTER: DataType.TABULAR,
        NodeType.MODEL_TRAINER: DataType.TABULAR,
        NodeType.EVALUATOR: DataType.TABULAR
    }
    return expectations.get(node_type, DataType.TABULAR)
```

## 3. LLM集成与语义引导生成

### 3.1 语义感知的LoRA微调

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

def create_semantic_lora_config():
    """为DAG语义理解优化的LoRA配置"""
    return LoraConfig(
        r=64,  # 高rank以捕获复杂语义
        lora_alpha=128,
        target_modules=[
            "q_proj", "v_proj",  # 基础注意力
            "k_proj",            # DAG结构依赖
            "gate_proj",         # 门控机制对类型系统重要
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"]
    )

def prepare_semantic_training_data(examples):
    """准备包含语义约束的训练数据"""
    semantic_examples = []
    
    for ex in examples:
        # 添加语义约束提示
        instruction = ex["instruction"]
        output_yaml = ex["output"]
        
        # 验证YAML语义
        try:
            dag_def = parse_and_validate_yaml(output_yaml)
            semantic_feedback = "VALID_DAG"
        except Exception as e:
            semantic_feedback = f"INVALID_DAG: {str(e)}"
        
        # 构建语义增强的训练样本
        semantic_examples.append({
            "instruction": f"{instruction}\n# SEMANTIC_RULES: DAG must be acyclic, node types must match parameters, data types must be compatible",
            "input": "",
            "output": f"# SEMANTIC_STATUS: {semantic_feedback}\n{output_yaml}"
        })
    
    return semantic_examples
```

### 3.2 语义约束的推理流程

```python
def generate_semantic_dag(model, tokenizer, prompt, max_attempts=3):
    """生成语义正确的DAG，带多层验证"""
    for attempt in range(max_attempts):
        try:
            # 1. 生成初始响应
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.3,  # 降低温度提高确定性
                do_sample=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 2. 提取YAML部分
            yaml_content = extract_yaml_block(generated_text)
            if not yaml_content:
                raise ValueError("No YAML block found in generated response")
            
            # 3. 多层语义验证
            dag_def = parse_and_validate_yaml(yaml_content)
            validate_node_semantics_all(dag_def)
            validate_data_flow(dag_def)
            
            # 4. 文档一致性检查 (可选)
            if not check_document_consistency(generated_text, dag_def):
                raise ValueError("Documentation inconsistent with DAG structure")
            
            return {
                "status": "success",
                "markdown": generated_text,
                "yaml": yaml_content,
                "dag": dag_def.dict()
            }
            
        except Exception as e:
            # 5. 错误反馈与修复
            error_feedback = f"GENERATION_ATTEMPT_{attempt+1}_FAILED: {str(e)}"
            print(f"Attempt {attempt+1} failed: {e}")
            
            # 构建修复提示
            repair_prompt = f"""
            {prompt}
            
            # PREVIOUS_ATTEMPT_FAILED
            Error: {str(e)}
            Generated YAML:
            ```yaml
            {yaml_content if 'yaml_content' in locals() else 'N/A'}
            ```
            
            # FIX_INSTRUCTIONS
            1. Ensure DAG has no cycles
            2. Verify node types match their parameters
            3. Check input/output references are valid
            4. Maintain proper YAML indentation
            """
            
            prompt = repair_prompt  # 用于下一次尝试
    
    raise RuntimeError(f"Failed to generate valid DAG after {max_attempts} attempts")
```

## 4. 语义验证工具链集成

### 4.1 完整验证流水线
```python
class DAGSemanticValidator:
    """集成所有语义验证的统一接口"""
    
    def __init__(self):
        self.yaml_parser = yaml.YAML(typ='safe')
        self.graph_validator = DAGStructureValidator()
        self.type_system = DataTypeSystem()
        self.node_registry = NodeRegistry()
    
    def validate_full_semantics(self, markdown_content: str) -> dict:
        """执行完整的语义验证"""
        results = {
            "status": "valid",
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # 1. Markdown结构验证
            yaml_blocks = self.extract_yaml_blocks(markdown_content)
            if len(yaml_blocks) == 0:
                results["status"] = "invalid"
                results["errors"].append("No YAML block found in Markdown")
                return results
            
            # 2. YAML语法验证
            try:
                dag_yaml = yaml_blocks[0]  # 假设第一个YAML块是DAG定义
                dag_dict = self.yaml_parser.load(dag_yaml)
            except Exception as e:
                results["status"] = "invalid"
                results["errors"].append(f"YAML syntax error: {str(e)}")
                return results
            
            # 3. DAG结构验证
            try:
                dag_def = DAGDefinition(**dag_dict)
                self.graph_validator.validate(dag_def)
                results["metrics"]["node_count"] = len(dag_def.nodes)
                results["metrics"]["edge_count"] = sum(len(node.inputs) for node in dag_def.nodes)
            except Exception as e:
                results["status"] = "invalid"
                results["errors"].append(f"DAG structure error: {str(e)}")
                return results
            
            # 4. 节点语义验证
            for node in dag_def.nodes:
                try:
                    self.node_registry.validate_node(node)
                except Exception as e:
                    results["warnings"].append(f"Node '{node.id}' semantic warning: {str(e)}")
            
            # 5. 数据流类型验证
            try:
                self.type_system.validate_data_flow(dag_def)
            except Exception as e:
                results["status"] = "invalid"
                results["errors"].append(f"Data flow type error: {str(e)}")
                return results
            
            # 6. 文档一致性检查
            doc_consistency = self.check_document_consistency(markdown_content, dag_def)
            if not doc_consistency["consistent"]:
                results["warnings"].extend(doc_consistency["issues"])
            
            # 7. 复杂度分析
            complexity = self.analyze_dag_complexity(dag_def)
            results["metrics"].update(complexity)
            
            # 8. 最终状态确定
            if results["errors"]:
                results["status"] = "invalid"
            elif results["warnings"]:
                results["status"] = "warning"
            
            return results
            
        except Exception as e:
            results["status"] = "invalid"
            results["errors"].append(f"Unexpected validation error: {str(e)}")
            return results
    
    def extract_yaml_blocks(self, markdown: str) -> list:
        """从Markdown中提取YAML代码块"""
        import re
        pattern = r'```(?:yaml|yml)?\n(.*?)\n```'
        matches = re.findall(pattern, markdown, re.DOTALL)
        return matches
```

### 4.2 与LLM训练/推理集成
```python
def train_semantic_dag_model(base_model_name, training_examples):
    """训练具有语义感知能力的DAG生成模型"""
    
    # 1. 准备带语义标签的训练数据
    semantic_dataset = []
    validator = DAGSemanticValidator()
    
    for ex in training_examples:
        validation_result = validator.validate_full_semantics(ex["output"])
        
        # 根据验证结果添加元数据
        semantic_label = "SEMANTIC_VALID" if validation_result["status"] == "valid" else "SEMANTIC_INVALID"
        error_summary = "; ".join(validation_result["errors"][:3])  # 取前3个错误
        
        semantic_dataset.append({
            "instruction": ex["instruction"],
            "input": "",
            "output": f"# {semantic_label}\n# ERRORS: {error_summary}\n{ex['output']}"
        })
    
    # 2. 配置和训练模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 扩展tokenizer以包含DSL特定token
    special_tokens = ["<DAG_START>", "<DAG_END>", "<SEMANTIC_VALID>", "<SEMANTIC_INVALID>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # 应用LoRA
    lora_config = create_semantic_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 训练
    trainer = create_semantic_trainer(model, semantic_dataset, tokenizer)
    trainer.train()
    
    return model, tokenizer

def create_semantic_trainer(model, dataset, tokenizer):
    """创建针对语义验证优化的训练器"""
    
    def compute_semantic_metrics(eval_preds):
        """计算语义相关的评估指标"""
        logits, labels = eval_preds
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        
        valid_count = 0
        total = len(predictions)
        
        for pred_ids in predictions:
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            validation_result = DAGSemanticValidator().validate_full_semantics(pred_text)
            
            if validation_result["status"] == "valid":
                valid_count += 1
        
        return {
            "semantic_validity_rate": valid_count / total,
            "avg_errors_per_invalid": sum(len(r["errors"]) for r in validation_results if r["status"] != "valid") / max(1, total - valid_count)
        }
    
    # Hugging Face Trainer配置
    training_args = TrainingArguments(
        output_dir="./dag_semantic_adapter",
        per_device_train_batch_size=2,  # DAG序列通常较长
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        num_train_epochs=20,  # 语义学习需要更多轮次
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="semantic_validity_rate",
        fp16=True
    )
    
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_semantic_metrics
    )
```

## 5. 高级语义保证技术

### 5.1 形式化验证与类型系统
```python
from typing import TypeVar, Generic, Type, Dict, Any
from pydantic.generics import GenericModel

T = TypeVar('T')

class TypedNode(GenericModel, Generic[T]):
    """带强类型的节点定义"""
    id: str
    type: str
    inputs: Dict[str, 'TypedNode'] = {}
    params: T
    output_type: DataType
    
    class Config:
        arbitrary_types_allowed = True

# 特定节点类型的类型定义
class TransformParams(BaseModel):
    operations: List[Union[
        Literal["remove_duplicates"],
        Dict[Literal["fill_missing"], Literal["mean", "median"]]
    ]]

class TransformNode(TypedNode[TransformParams]):
    output_type: DataType = DataType.TABULAR

# 运行时类型检查
def check_node_type_consistency(node: Any, expected_type: Type) -> bool:
    """运行时检查节点类型一致性"""
    try:
        expected_type(**node.dict())
        return True
    except ValidationError as e:
        print(f"Type consistency error: {e}")
        return False
```

### 5.2 执行验证（最严格的语义保证）
```python
class DAGExecutor:
    """DAG执行验证器"""
    
    def __init__(self, dry_run=True):
        self.dry_run = dry_run  # 是否只验证不执行
        self.node_handlers = {
            "data_source": self._handle_data_source,
            "transform": self._handle_transform,
            # ... 其他节点类型处理函数
        }
    
    def validate_execution(self, dag_def: DAGDefinition) -> dict:
        """通过模拟执行验证DAG语义"""
        execution_plan = self._create_execution_plan(dag_def)
        results = {"status": "valid", "errors": [], "outputs": {}}
        
        try:
            # 按拓扑顺序执行节点
            for node_id in execution_plan:
                node = next(n for n in dag_def.nodes if n.id == node_id)
                
                # 获取节点处理器
                handler = self.node_handlers.get(node.type)
                if not handler:
                    raise ValueError(f"No handler for node type: {node.type}")
                
                # 收集输入
                inputs = {}
                for input_ref in node.inputs:
                    source_node_id, output_name = input_ref.split('.', 1)
                    if source_node_id not in results["outputs"]:
                        raise ValueError(f"Input not ready: {input_ref}")
                    inputs[output_name] = results["outputs"][source_node_id]
                
                # 执行节点 (dry run)
                output = handler(node.params, inputs, dry_run=self.dry_run)
                results["outputs"][node_id] = output
            
            # 验证输出
            for output_name, ref in dag_def.outputs.items():
                if ref not in results["outputs"]:
                    raise ValueError(f"Output reference not found: {ref}")
            
            return results
            
        except Exception as e:
            results["status"] = "invalid"
            results["errors"].append(f"Execution validation failed: {str(e)}")
            return results
    
    def _handle_data_source(self, params: dict, inputs: dict, dry_run: bool) -> Any:
        """模拟数据源节点执行"""
        # 验证参数
        if "format" not in params:
            raise ValueError("Missing required parameter: format")
        
        if params["format"] not in ["csv", "parquet", "json"]:
            raise ValueError(f"Unsupported format: {params['format']}")
        
        # Dry run: 返回模拟的数据模式
        if dry_run:
            return {
                "type": "dataset",
                "schema": {"columns": ["id", "timestamp", "value"], "types": ["int", "timestamp", "float"]},
                "size_estimate": "1GB"
            }
        
        # 实际执行 (略)
        return actual_data_loading(params)
```

## 6. 实践建议与权衡

### 6.1 语义保证的层次选择
| 保证级别 | 验证深度 | 资源需求 | 适用场景 |
|---------|---------|---------|---------|
| **基础** | YAML语法 + DAG无环 | 低 | 快速原型、内部工具 |
| **标准** | + 节点类型验证 + 数据流 | 中 | 生产环境、团队协作 |
| **严格** | + 执行验证 + 形式化类型 | 高 | 关键系统、金融/医疗 |

### 6.2 性能与正确性权衡
```python
def configure_semantic_validation(level: str = "standard") -> DAGSemanticValidator:
    """根据需求级别配置验证器"""
    validator = DAGSemanticValidator()
    
    if level == "basic":
        # 仅基础验证
        validator.enable_validations([
            "yaml_syntax",
            "dag_acyclic"
        ])
    elif level == "standard":
        # 标准验证 (推荐)
        validator.enable_validations([
            "yaml_syntax",
            "dag_acyclic",
            "node_semantics",
            "data_flow_types",
            "document_consistency"
        ])
    elif level == "strict":
        # 严格验证
        validator.enable_validations([
            "yaml_syntax",
            "dag_acyclic",
            "node_semantics",
            "data_flow_types",
            "document_consistency",
            "execution_simulation"
        ])
        validator.set_timeout(30)  # 严格模式允许更长超时
    else:
        raise ValueError(f"Unknown validation level: {level}")
    
    return validator
```

## 7. 完整工作流示例

```python
# 端到端DAG生成与验证流程
def generate_valid_markdown_dag(instruction: str) -> dict:
    """生成语义有效的Markdown-YAML DAG"""
    
    # 1. 加载预训练的语义感知模型
    model, tokenizer = load_semantic_dag_model()
    
    # 2. 生成初始响应
    prompt = f"""
    Create a data processing pipeline in Markdown with embedded YAML DAG.
    {instruction}
    
    # OUTPUT_FORMAT
    - Start with Markdown documentation
    - Include exactly one YAML code block with DAG definition
    - End with additional Markdown documentation
    - Ensure DAG is acyclic and node types match their parameters
    
    # SEMANTIC_RULES
    1. All node IDs must be unique and lowercase_with_underscores
    2. Input references must follow format: "source_node.output_name"
    3. Data source nodes must have path and format parameters
    4. Transform nodes must specify operations array
    """
    
    # 3. 生成与验证循环
    validator = DAGSemanticValidator()
    max_attempts = 5
    
    for attempt in range(max_attempts):
        # 生成
        generated = generate_with_model(model, tokenizer, prompt)
        
        # 验证
        validation_result = validator.validate_full_semantics(generated["text"])
        
        if validation_result["status"] == "valid":
            return {
                "success": True,
                "markdown": generated["text"],
                "validation": validation_result
            }
        
        # 修复
        errors = "\n".join(validation_result["errors"])
        warnings = "\n".join(validation_result["warnings"])
        
        prompt += f"""
        
        # PREVIOUS_ATTEMPT_FAILED (attempt {attempt+1})
        Errors:
        {errors}
        Warnings:
        {warnings}
        
        # CORRECTION_GUIDANCE
        1. Fix syntax errors in YAML indentation and structure
        2. Remove cycles in the DAG by reordering nodes
        3. Ensure all input references point to existing node outputs
        4. Verify parameters match expected format for each node type
        """
    
    return {
        "success": False,
        "error": f"Failed to generate valid DAG after {max_attempts} attempts",
        "last_attempt": generated["text"],
        "validation": validation_result
    }

# 使用示例
result = generate_valid_markdown_dag(
    "Create a pipeline that loads user activity data, cleans it, extracts time-based features, and splits into train/test sets"
)

if result["success"]:
    print("✅ Valid DAG generated!")
    print(result["markdown"])
else:
    print("❌ Failed to generate valid DAG")
    print("Last attempt validation errors:", result["validation"]["errors"])
```

## 关键结论

1. **语义保证不能仅靠LLM**：必须结合外部验证工具链，LLM负责生成，验证器负责检查

2. **分层验证架构**：
   - **语法层**：Markdown/YAML正确性
   - **结构层**：DAG无环性、连接完整性
   - **类型层**：节点输入/输出类型兼容
   - **语义层**：参数与节点类型匹配
   - **执行层**：模拟执行验证（可选）

3. **训练优化**：在训练数据中包含语义验证结果，教导模型预测可能的错误

4. **渐进式严格性**：
   - 开发阶段：基础验证（快速迭代）
   - 测试阶段：标准验证（平衡速度与正确性）
   - 生产阶段：严格验证（最大正确性保证）

5. **实用建议**：
   - 从"标准"验证级别开始
   - 为常见错误模式创建修复模板
   - 维护错误-修复映射知识库
   - 设置合理的重试次数（3-5次）
   - 关键系统增加人工审核步骤

**最终建议**：对于计算图DAG这种关键结构，**语义验证应被视为一等公民**，与模型生成同等重要。理想架构是"生成-验证-修复"的闭环系统，而非依赖单一模型保证正确性。
