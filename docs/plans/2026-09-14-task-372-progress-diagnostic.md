# 任务 372 停在 99%：生产诊断与修复验证

状态：用户已批准实现，本地代码、回归、浏览器验证完成；尚未发布。

批准证据：2026-09-14 当前会话用户明确确认“修复方案已确认，要给用户持续99%的原因和具体练习建议，并验证后续合格窗口能否正常完成任务。”

## 生产证据

2026-09-14 15:49（Asia/Shanghai）用户报告：
`[BATCH_EVAL] media-independent result: task_id=372 score=9 ready=False`。

通过 Zeabur executeCommand 在 workflow-service 内执行 PostgreSQL 只读事务，
仅查询任务数值、评分枚举和时间；未读取或保存用户对话、评分理由或凭据。
查询时任务状态为 pending，score=9，interaction_count=27，scoring_generation=0。

| 窗口 | 质量 | 累计分数 | 累计轮数 | intended_ready |
| --- | --- | --- | --- | --- |
| 1 | satisfactory | 2 | 3 | false |
| 2 | needs_work | 3 | 6 | false |
| 3 | needs_work | 4 | 9 | false |
| 4 | satisfactory | 6 | 12 | false |
| 5 | needs_work | 7 | 15 | false |
| 6 | needs_work | 8 | 18 | false |
| 7 | needs_work | 9 | 21 | false |
| 8 | needs_work | 9 | 24 | false |
| 9 | needs_work | 9 | 27 | false |

最后记录 expires_at=2026-09-17T07:49:23.124188Z；代码以事务 NOW()+72 小时
写入，对应 09-14 15:49:23。第 7、8 个窗口分别对应 15:37:26 和 15:39:44。

按实际 `_ready_key` 的 NUL 分隔算法查询 Redis：键存在，generation=0，order=27，
token 为空，TTL=258672 秒。Redis PING 成功。早先使用冒号分隔的探测键不匹配，
其“键不存在”结果已排除，不作为诊断依据。

生产与本地 `services/workflow-service/src/workflows/batch_evaluation.py`
SHA256 均为 `821a0a73cd4ab21c43450e700ae7f48f62e516ccfa2b2bd50a9294553b00836d`。
ai-omni 部署记录为 RUNNING，commitSHA=`b40cf9c42c5a0eb2b3904de29481c020f6cbbc25`。

普通 runtimeLogs（CLI 和 GraphQL）返回空数组；searchRuntimeLogs 返回
PERMISSION_DENIED，说明高级日志搜索要求 Pro 或 Team。未获取到原始 15:49 日志，
不能据此宣称模型质量判断正确或错误。

接口依据：https://zeabur.com/docs/en-US/developer/public-api 。

## 结论与真实调用链

`_evaluate_scene_turn_progress` → accumulator → workflow batch-evaluate →
`_apply_evaluation` → `_reconcile_readiness` → `_emit_scoring_result` →
Conversation proficiency_update。

`needs_work` 增加 1 分，累计分数封顶 9；ready 同时要求最新质量属于
mastered/strong/satisfactory。本次最新质量不满足，数据库与 Redis 一致，
没有发现完成凭据发布故障。不能靠强制补 token、改 status 或把 99% 改 100% 修复。

Conversation.js 把 reason 放进含“熟练度”的系统消息，3 秒后删除；没有持久展示
未完成条件。这会让“累计分已足、最新表现未达标”看起来像进度卡死。
实际对话与模型判定未审查，模型是否过严仍未证实。

## 本地验证

- `.venv/bin/python -m pytest services/workflow-service/tests/test_batch_evaluation.py -q`
  — 24 passed。
- 使用现有 FakeDB/FakeRedis、mock 模型质量，依次重放上表 9 个窗口：
  score/count/ready 与生产一致。再加入 satisfactory 窗口：score=9，count=30，
  ready=true，token 非空。该验证覆盖评分函数，不代表真实模型或浏览器端到端通过。

## 已批准实现方案

1. 服务端返回结构化未完成原因，区分最新表现未达标、窗口不足与凭据发布失败；
   保留既有完成门槛、重置代次与幂等约束，避免将不同原因混为 99%。
2. ai-omni 转发原因；前端在任务进度附近持续显示“仍需完成一组合格练习”及
   最新评分建议；在合格、切换任务、重置和完成时正确更新或清除。支持现有 i18n。
3. 对此次连续 needs_work 达到 9 分的序列添加回归，并验证下一合格窗口进入
   完成确认、零分窗口不误报完成、Redis 失败可区分、跨任务与旧代次消息不污染提示。
4. 运行相关 workflow/ai-omni/client 测试和 client 构建，完成人工可审阅 PR。
   合并与生产发布需人工批准，发布后核验任务完成确认链路；不代用户完成任务。

产品选择：若希望“累计 9 分即完成”，必须明确变更 AGENTS/core-rules 中
“最新窗口至少 satisfactory”的契约，并重新设计和审批评分规则。本方案不作该变更。

## 流程与验收边界

根目录六工件仍属于 ai-native-sdlc-bootstrap，release=ready、maintenance=pending；
`python3 scripts/sdlc.py validate` 返回 clean；GitHub 当前无 open PR。
本诊断不覆盖该活跃工件组，不伪造新循环或计划批准。
本方案已获当前会话明确批准，实施证据继续记录于本诊断文件，
不覆盖仍属于另一变更的根工件，也不虚构其 release/maintenance 完成。

没有生产数据改动、部署或生产修复验收。用户原有未提交文件保持不变。

## 实施内容

- Workflow 返回 completion_blocker，区分分数、窗口、最新质量和凭据发布失败。
  完成门槛不变；新增可缺省且最多 400 字符的 practice_tip，提示模型生成具体动作和目标语言示例。
- ai-omni 转发原因和建议；凭据重试耗尽仍冻结原窗口，不重复计分，向 UI 提供服务异常说明。
  重连从已存 Redis 结果恢复与用户/目标/任务/代次/分数/轮次一致的建议，不恢复 token。
- Conversation 的普通/CC 进度旁常驻原因和建议，九种 UI 语言都有文案；过长内容可滚动。
  继续深入练习后可重新打开完成确认；合格、完成、切换、重置和重连会更新或清除旧状态。
  新提示不写入 localStorage；只复用现有服务端评分缓存。
- 新回归覆盖生产九窗口序列、后续合格、无效窗口、Redis 发布恢复、消息隔离、
  建议恢复及浏览器确认链路。

## 审查

oral-app-sdlc 要求的独立只读审查已进行，范围为行为 diff 与相关调用链，依据 REVIEW.md。
审查发现重连后可能残留旧 ready/pending 状态，已修复；对应浏览器回归已加入。
独立 fresh-context 最终只读审查已核对暂存区完整修复 diff，结论为零项可操作问题，ready。
范围包含 UI、九种语言、workflow/ai-omni 及测试，排除用户 docs/TODO.md。
命令为 git diff、git diff --cached、git status --short、rg、sed、cat；
风险范围 scoring、WS/audio、data、UI；未解决 high/critical=0。
该审查未重复运行测试或操作生产，测试结果由主线程提供。

## 实际验证结果（2026-09-14）

- `npm run verify` — 最终 exit 0，decision=pass，score=100。
  包含 workflow 125、ai-omni 220、client 564、user-service 153 项通过；
  user-service 原有 3 项跳过，未把它们计作通过。前端生产构建成功，保留已有编译警告。
- `npm run verify:ui -- -- task-progress-guidance.spec.js --project=chromium-320 --project=chromium-390 --project=chromium-desktop --project=webkit-mobile`
  — exit 0，4 passed。覆盖提示超时后保留、跨任务/代次隔离、服务异常提示、恢复建议、
  CC 模式、重连取消旧确认、继续练习后重开确认，以及确认消息携带 task/token 后 100%。
- `python3 test_scenario_batch_and_daily_qa.py --scenario all --mock` — exit 0，25 passed。
- `docker compose build workflow-service ai-omni-service` — exit 0；最终 ai 修改后再次
  `docker compose build ai-omni-service` — exit 0。
- `python3 scripts/sdlc.py validate`、`git diff --check` — exit 0。

浏览器使用合成账号、mock REST/WS 和模型质量序列；没有录音、发送真实对话、
更改任务 372 或调用生产评分。Workflow 验证真实评分逻辑，user-service 既有确认测试
验证凭据校验/消费，浏览器验证用户确认消息与完成通知；这些是分层回归，不是线上端到端验收。

本地忽略目录 `quality/artifacts/latest/` 保存最终全仓结果，
`quality/artifacts/playwright-results/` 保存各 viewport 的 persistent-guidance.png、
cc-guidance.png。已人工查看 320px、390px CC、1440px 图像，确认提示可读、无横向溢出。
专用技能的静态状态扫描未另外运行，状态由针对性单测和浏览器用例验证。

验证期间曾发现并解决：翻译嵌套结构不符已有扁平键约束、测试夹具缺少任务数据/误选开发服务器
WebSocket，以及 CC 进度区域拦截退出按钮点击。失败运行未当作通过；CC 超时后测试进程
未退出，已终止该进程，修复后最终完整命令 exit 0。

## 发布与回滚

无数据库迁移。需要部署 workflow-service、ai-omni-service 和 client。
先部署兼容的后端字段，再部署前端；旧评分响应缺少建议时前端使用当前任务的练习动作提示。
回滚本 PR 即可恢复原行为；新字段为可选，原评分记录和 token 格式保持兼容。
合并与生产发布仍须人工批准，浏览器 mock 与本地单测不能替代生产验收。
