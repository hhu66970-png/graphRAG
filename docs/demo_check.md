# PPT 演示前检查

`eval/demo/ppt_demo_cases.json` 保护《最终版（1）(1).pptx》的 8 个问题，记录源文件 SHA-256 和物理页码；不包含个人路径或截图。共 9 次请求：第 8 张四题用 `auto`，第 9 张同一配伍题用 `vector` 和 `keyword_graph`，第 10 张三题用 `auto`。

默认命令只校验清单，不启动后端、不联网、不调用模型：

```powershell
python scripts/check_ppt_demo.py
# 或显式指定，仅校验清单：
python scripts/check_ppt_demo.py --validate-only
```

现场前先按 README 配置业务 Neo4j、已导入数据和 ACTIVE 批次，以及普通回答的模型配置。启动后端后，显式运行真实检查（普通问题可能调用所配置的模型）：

```powershell
python scripts/check_ppt_demo.py --run --base-url http://127.0.0.1:5000 --out demo-check.json
```

脚本先 GET `/api/health`，成功后逐项 POST `/api/graphrag`；记录 HTTP 状态、耗时、响应契约、四个设计状态和三个普通路由。健康失败会停止，状态/契约失败退出码为 1。失败报告不保存服务器异常详情、响应头、配置或密钥；正常报告也只保留状态、路由、计数等有限字段，不保存原始回答。

四个高风险预期为 `grounded / partial_grounded / source_only / abstain`，来自 PPT 的设计说明，**不是医学 gold，也不是已经通过的真实教材测试**。数据库内容变更可能使设计预期不成立，应查验原文和适用条件，不能为让演示变绿而修改生产回答或标注。自动检查不核验完整语义、原文真实性、引用是否正确，也不要求旧截图的 84/137、4/8 裁剪数量保持不变。

配伍两次请求只做接口契约检查。图检索旧截图由“慢惊风常用药”共同名单推出“可以搭配”，但同页文字明确指出：**共同病证归类不等于安全配伍已获证明**。旧肯定回答不能作答案 gold。请人工查看本次两种回答及其完整证据、页码和适用条件，核对是否越过安全边界，记录有分歧的结论并按 `docs/evaluation.md` 的人工仲裁流程处理。也应人工核对另外七题的证据与实际显示效果。

没有人工复核声明时，即使 9 次自动契约检查全部通过，报告仍为 `needs_manual_review`，退出码为 2，不能称“演示全部通过”。完成本次配伍双请求的实际核对后，操作者可显式声明：

```powershell
python scripts/check_ppt_demo.py --run --base-url http://127.0.0.1:5000 --reviewed-case renshen_wugong_pair --out demo-check-reviewed.json
```

`--reviewed-case` 仅记录操作者声明，不是程序验证；应先检查本次报告对应的相同数据、配置和回答，若再次调用产生不同结果要重新复核。其结果最多为 `checks_complete_with_operator_review`，`medical_correctness_verified` 始终为 `false`。脚本不接受 URL 中的用户名、密码或查询参数，也不读取 `.env` 或自动寻找服务。

离线回归：

```powershell
python -m pytest tests/api/test_ppt_demo_cases.py tests/scripts/test_ppt_demo_check.py
```

API 回归用现有合成断言和存储/模型替身，运行真实高风险核验；普通三例调用真实 `route_query`。这些测试用于保护接线、对象/人群边界和路由，不能证明业务数据库、模型、Windows 现场或实际演示可用。没有业务 `.env`、运行中的后端和可用 Neo4j 时，只能完成这些离线检查，真实演示仍未验证。
