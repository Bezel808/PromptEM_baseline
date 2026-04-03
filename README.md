# PromptEM Baseline

## 任务
实体匹配（Entity Matching）二分类：判断左表实体与右表实体是否匹配。

## Input
- 数据目录：`data/<dataset>/`
- 实体文件：`left.(csv|json|txt)`、`right.(csv|json|txt)`
- 划分文件：`train.csv`、`valid.csv`、`test.csv`
- 划分文件字段：`left_id,right_id,label`

## Output
- 评估指标：`Precision/Recall/F1/Accuracy/AUC`
- 日志输出：`logs/PromptEM_*`
- 汇总输出（脚本运行时）：`runs/promptem_em/.../summary.md`、`summary.json`

## Loss Function
- `CrossEntropyLoss`
- 自训练阶段仍使用同一监督损失（伪标签样本并入训练集）

## Dataset Path
- 仓库内常用：
  - `/home/zongze/mengshichen_projects/PromptEM_baseline/data/wikidbs_1218`
  - `/home/zongze/mengshichen_projects/PromptEM_baseline/data/santos_benchmark_1218`
  - `/home/zongze/mengshichen_projects/PromptEM_baseline/data/magellan_1218`
  - 以及对应 `*_040303` 数据集
- 运行时常见原始根目录：
  - `/home/zongze/mengshichen_projects/datasets_joint_discovery_integration_split_work`
  - `/home/mengshi/table_quality/datasets_joint_discovery_integration`
