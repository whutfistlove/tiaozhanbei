# 微信 Bot 部署计划：算子优化 Agent 的可视化前端

> **定位**：在已建好的"算子优化 Agent 系统"（见 AGENT_DESIGN_PLAN.md）之上，加一层微信交互前端，实现"对话即优化"。
> **接入方式**：企业微信自建应用（HTTP 回调模式）—— 官方合规、个人可注册、支持文件收发、无 48 小时推送限制
> **核心场景**：用户在微信上传 `.cu` 文件或仓库链接 → 后端 Agent 跑优化 → 推送结果报告/改进意见回微信
> **产出方**：ZCode Agent（基于 2 份微信 Bot 调研报告 + 环境实测）

---

## 第一章 · 方案选型与依据

### 1.1 为什么是企业微信自建应用

对"收文件 + 长任务 + 异步推结果"场景，企业微信自建应用是**唯一在所有关键维度都满足**的方案：

| 能力 | 企业微信自建应用 | 个人微信Hook(WeChatFerry) | 公众号服务号 | 群机器人/Webhook |
|------|:---:|:---:|:---:|:---:|
| 接收用户文件(.cu) | ✅ HTTP回调 | ✅ | ⚠️ 弱 | ❌ |
| **任意时间主动推送** | ✅ **无限制** | ✅ | ❌ 48h窗口 | ✅(单向) |
| 个人可注册 | ✅ 免认证 | — | ❌ 需企业 | ✅ |
| 合规性 | ✅ 官方原生 | ❌ 违反协议 | ✅ | ✅ |
| 封号风险 | 极低 | 中-高 | 无 | 无 |

**决定性优势**：算子优化要几十分钟，公众号有 48 小时客服窗口限制、个人微信 Hook 有封号风险，**只有企业微信自建应用既合规又能任意时间主动推结果**。

### 1.2 必须避开的坑（调研发现）

> ⚠️ **用"自建应用 + 接收消息（HTTP 回调）"模式，不要用"智能机器人长连接"模式**——长连接模式目前**收不了文件**（GitHub openclaw#56140 确认）。要收 `.cu` 文件必须走 HTTP 回调。

### 1.3 已知环境约束（实测）

| 约束 | 实测结果 | 应对 |
|------|---------|------|
| C500 实例在 Docker 容器内 | `/.dockerenv` 存在 | 容器内端口无法被公网直接访问 |
| 出口 IP | 140.207.205.81 | 是出口 NAT，非可入站 IP |
| 内网穿透工具 | 无（frpc/ngrok/cpolar 都没装） | **需安装 cpolar 或 frp** |
| Web 框架 | flask/fastapi 都没装 | **需安装** |
| 回调超时 | 被动回复 5 秒超时 | **回调立即返回空串 200，长任务异步** |

---

## 第二章 · 系统总体架构

### 2.1 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│  ① 微信前端层（企业微信 App / 微信互通）                       │
│     用户操作：发 .cu 文件 / 发仓库链接 / 查进度 / 收报告        │
└──────────────────────────┬──────────────────────────────────┘
                           │ 企业微信回调(POST加密XML) + 主动推送API
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ② Bot 网关层（FastAPI，跑在 C500 实例，经内网穿透暴露公网）    │
│     · 回调验签/加解密(WXBizMsgCrypt)                          │
│     · 文件中转(media_id ↔ 本地文件)                           │
│     · 任务队列(Redis/SQLite) + 异步worker                     │
│     · 进度推送(主动发消息API)                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用 / 投递任务
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ③ 算子优化 Agent 层（已有，见 AGENT_DESIGN_PLAN.md）          │
│     Orchestrator → Analyst/Coder/Profiler/Judge/Reflector     │
│     + Roofline 引擎 + 领域记忆库 + mctlass skill              │
│     产出：优化后 run_kernel.cu + 优化报告 + 改进意见            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 异步长任务的标准流程（解决 5 秒超时难点）

```
[用户] 在企业微信发 .cu 文件 / 仓库链接
   │
   ▼
[企业微信] POST 加密回调到 Bot 网关 (含 media_id 或文本)
   │
   ▼
[Bot 网关] 5秒内完成：验签→解密→生成 task_id→回复空串 HTTP 200
   │   （超时企业微信会重试3次，必须幂等，用 task_id 去重）
   │
   ├─→ [文件中转] 下载 media_id 文件到 /data/wechat_uploads/{task_id}/
   ├─→ [入队] 任务投入 Celery/RQ 队列
   └─→ [即时反馈] 主动推送"✅ 已收到，task_id=xxx，开始优化（预计X分钟）"
   │
   ▼ (异步，几十分钟)
[Agent 层] 跑完整优化闭环（Analyst→Coder→Profiler→Judge→Reflector）
   │   · 每个里程碑主动推送进度（"分析瓶颈中…"/"生成候选…"/"验证中…"）
   │
   ▼ (完成)
[Bot 网关]
   ├─→ 生成优化报告 markdown + 改进意见
   ├─→ 优化后的 .cu 上传临时素材 → media_id
   └─→ 主动推送：文本报告(markdown) + 文件(优化版.cu) + 结果链接
   │
   ▼
[用户] 收到完整结果
```

### 2.3 五个技术难点的解法

| 难点 | 解法 |
|------|------|
| ① 异步长任务(5s超时) | 回调立即返回空串200 + 任务入队异步 + 完成后主动推送API |
| ② 文件中转 | 用户文件→media_id→下载；结果文件→上传media_id→发文件消息(≤20MB) |
| ③ 富文本报告 | 用 markdown 消息(企微支持) + 关键图表生成图片推送 + 完整报告附文件 |
| ④ 任务状态查询 | 用户发"进度"→Bot查队列状态回当前阶段 |
| ⑤ GPU 资源调度 | 任务队列串行/限并发，多用户排队，返回排队位置 |

---

## 第三章 · 企业微信自建应用配置（详细步骤）

### 3.1 注册企业微信（个人免认证）

1. 访问 https://work.weixin.qq.com/ → 注册
2. 主体类型选**「个人组建团队」**（身份证即可，无需营业执照）
3. 完成实名认证

### 3.2 创建自建应用

1. 管理后台 → **应用管理 → 自建 → 创建应用**
2. 填应用名（如"算子优化助手"）、logo、可见范围（自己）
3. 记录 **AgentId**

### 3.3 获取凭证

在应用详情页获取：
- **CorpID**（企业ID，在"我的企业"页）
- **CorpSecret**（应用 Secret）
- **AgentId**

### 3.4 配置接收消息（HTTP 回调）—— 关键

1. 应用详情 → **接收消息 → 设置 API 接收**
2. **回调 URL**：填内网穿透后的公网地址，如 `https://xxx.cpolar.top/wecom/callback`
3. **Token** 和 **EncodingAESKey**：随机生成并记录
4. 配置**企业可信 IP**：填 C500 实例出口 IP（140.207.205.81）

> ⚠️ 注意：必须用"接收消息（API 接收）"，**不要用"智能机器人长连接"**（后者收不了文件）。

### 3.5 内网穿透（解决容器无法公网访问）

C500 在 Docker 容器内，回调服务无法被公网直达。三选一：

| 工具 | 特点 | 适用 |
|------|------|------|
| **cpolar**（推荐） | 国内友好，免费版有随机子域名，简单 | 比赛演示 |
| frp | 需自有公网服务器作中转 | 有云服务器时 |
| ngrok | 国外，免费版随机域名 | 备选 |

安装 cpolar（示例）：
```bash
# 下载安装
curl -L https://www.cpolar.com/static/downloads/install-release-cpolar.sh | sudo bash
# 需注册 cpolar 账号获取 authtoken
cpolar authtoken <你的token>
# 暴露 Bot 网关的 8000 端口
cpolar http 8000
# 得到公网地址如 https://xxx.r6.cpolar.top
```

---

## 第四章 · Bot 网关实现

### 4.1 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| Web 框架 | **FastAPI** | 异步原生，适合回调+长任务 |
| 任务队列 | **Celery + Redis** 或 **RQ** | 异步执行几十分钟任务 |
| 企微加解密 | **WXBizMsgCrypt**（官方 Python 版） | 回调验签/AES解密 |
| HTTP 客户端 | httpx | 调企微 API |
| 状态存储 | SQLite（轻量）或 Redis | 任务状态追踪 |

### 4.2 目录结构（新增到 flashattn_task_package）

```
flashattn_task_package/
├── wecom_bot/                      # ★ 新增：微信 Bot 网关
│   ├── server.py                   # FastAPI 主服务（回调+推送）
│   ├── crypto.py                   # 企微消息加解密(WXBizMsgCrypt)
│   ├── wecom_api.py                # 企微 API 封装(access_token/上传/下载/发消息)
│   ├── tasks.py                    # 异步任务定义(Celery)
│   ├── task_queue.py               # 任务队列+状态管理
│   ├── file_bridge.py              # 文件中转(media_id ↔ 本地)
│   ├── report_formatter.py         # 优化结果→markdown报告+改进意见
│   ├── config.py                   # CorpID/Secret/AgentId/Token/AESKey
│   └── templates/
│       ├── progress.md             # 进度推送模板
│       └── final_report.md         # 最终报告模板
├── agent_system/                   # 已有：算子优化 Agent（被 Bot 调用）
│   ├── orchestrator_loop.py        # 主入口，Bot 调用它启动优化
│   └── ...
└── data/
    ├── wechat_uploads/             # 用户上传的 .cu 文件
    └── wechat_results/             # 优化结果文件
```

### 4.3 核心模块设计

#### 4.3.1 回调处理（server.py 核心）

```python
@app.post("/wecom/callback")
async def callback(msg_signature, timestamp, nonce, body: bytes):
    # 1. 验签 + AES 解密
    msg = WXBizMsgCrypt.decrypt(body, msg_signature, timestamp, nonce)
    
    # 2. 幂等去重（防 5s 重试 3 次）
    msg_id = msg["MsgId"]
    if redis.exists(f"seen:{msg_id}"):
        return ""  # 已处理，直接空串
    redis.setex(f"seen:{msg_id}", 300, "1")
    
    # 3. 分发
    if msg["MsgType"] == "file":
        # 用户上传了 .cu 文件
        task_id = create_task(user=msg["FromUserName"], type="file", media_id=msg["MediaId"])
        download_file_async(task_id, msg["MediaId"])  # 下载到本地
        enqueue_optimization(task_id)
        push_text(user, f"✅ 已收到源码，任务 {task_id} 开始优化（预计X分钟）")
    elif msg["MsgType"] == "text":
        text = msg["Content"]
        if text.startswith("http"):  # 仓库链接
            task_id = create_task(user=msg["FromUserName"], type="repo", url=text)
            enqueue_optimization(task_id)
            push_text(user, f"✅ 已收到仓库链接，任务 {task_id} 开始")
        elif text in ("进度", "progress", "状态"):
            status = get_task_status(user)
            push_text(user, status)
    
    # 4. 立即返回空串 200（不等长任务）
    return ""
```

#### 4.3.2 文件中转（file_bridge.py）

```python
def download_user_file(media_id, save_path):
    """企微 → 本地：用户上传的 .cu"""
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/get?access_token={token}&media_id={media_id}"
    resp = httpx.get(url)
    save_path.write_bytes(resp.content)  # media_id 3天有效，及时下载

def upload_result_file(file_path) -> str:
    """本地 → 企微：优化后的 .cu"""
    url = f"https://qyapi.weixin.qq.com/cgi-bin/media/upload?access_token={token}&type=file"
    with open(file_path, "rb") as f:
        resp = httpx.post(url, files={"media": f})
    return resp.json()["media_id"]  # 用于发送文件消息
```

#### 4.3.3 异步优化 + 进度推送（tasks.py）

```python
@celery.task
def run_optimization(task_id):
    # 进度回调：每个里程碑推送一次
    def on_progress(stage, detail):
        push_text(user, f"⏳ [{stage}] {detail}")  # 主动推送（无时间限制）
    
    # 调用已有 Agent 系统
    result = agent_system.orchestrator_loop.optimize(
        source_file=upload_path,
        progress_cb=on_progress
    )
    
    # 生成报告 + 上传结果文件
    report = report_formatter.render(result)  # markdown
    media_id = upload_result_file(result.kernel_path)
    
    # 推送最终结果
    push_markdown(user, report)               # 文字报告
    push_file(user, media_id, "optimized_kernel.cu")  # 文件
```

#### 4.3.4 报告格式化（report_formatter.py）

输出给用户的最终报告示例：
```markdown
# 🎯 算子优化报告

## 性能对比
| 配置 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| batch=1,seq=4096 | 0.67ms | 0.31ms | 2.2× |
| batch=1,seq=16384 | 2.67ms | 1.15ms | 2.3× |
| 有效带宽 | 47 GB/s | 380 GB/s | 8× |

## 优化措施
1. ✅ Split-K 并行（batch=1 提升主因）
2. ✅ mctlass EpilogueVisitorSoftmax 融合
3. ✅ 软件流水（num_stages=3）

## 改进意见（下一步可尝试）
- ⚠️ FlashDecoding++ 统一 max value，进一步消除 reduce 同步
- ⚠️ 尝试更大的 tile_n（当前32→64）
- 📊 当前达理论带宽 21%，仍有空间

## 附件
optimized_kernel.cu（优化后源码）
```

---

## 第五章 · 关键 API 速查（企业微信）

| 功能 | 接口 | 文档 |
|------|------|------|
| 获取 access_token | GET `/cgi-bin/gettoken?corpid=&corpsecret=` | path/91039 |
| **接收消息**（回调） | POST 你的回调URL（加密XML） | path/96426 |
| 验证URL有效性 | GET 回调URL（带echostr） | path/90930 |
| **下载用户文件** | GET `/cgi-bin/media/get?media_id=` | path/90254 |
| **上传结果文件** | POST `/cgi-bin/media/upload?type=file` | path/90253 |
| **发送文本消息** | POST `/cgi-bin/message/send` msgtype=text | path/90236 |
| **发送markdown** | POST `/cgi-bin/message/send` msgtype=markdown | path/90236 |
| **发送文件** | POST `/cgi-bin/message/send` msgtype=file | path/90236 |

频率限制：每用户每应用 ≤ 20 条/分钟（足够推送进度+结果）。

---

## 第六章 · 分阶段实施计划

### 阶段 0：环境与依赖（半天）
- [ ] 注册企业微信（个人组建团队）+ 创建自建应用 + 记录凭证
- [ ] 安装 cpolar 内网穿透，获得公网回调 URL
- [ ] `pip install fastapi uvicorn httpx redis celery` + 企微加解密 SDK
- [ ] 配置回调 URL + Token + EncodingAESKey + 企业可信IP

### 阶段 1：消息收发打通（1-2 天）
- [ ] 实现 server.py 回调验签+解密+空串返回
- [ ] 实现 wecom_api.py（access_token 缓存、发文本）
- [ ] 验证：微信发文字 → Bot 收到 → 回复"收到"
- [ ] 实现文件下载（用户发 .cu → 存到本地）

### 阶段 2：接入 Agent 系统（2-3 天）
- [ ] 实现 task_queue.py（SQLite 任务表 + 状态机）
- [ ] 实现 tasks.py（Celery 异步跑 orchestrator_loop）
- [ ] 进度回调（on_progress → 主动推送）
- [ ] 验证：发 .cu → 自动优化 → 收到进度 → 收到结果

### 阶段 3：结果富文本（1-2 天）
- [ ] report_formatter.py（结果→markdown 报告 + 改进意见）
- [ ] 结果文件上传 + 发送文件消息
- [ ] 性能对比图表生成（matplotlib → 图片推送）

### 阶段 4：体验打磨（1 天）
- [ ] 命令体系：`/help` `/进度` `/历史` `/取消`
- [ ] 错误处理（编译失败/超时/资源占用时友好提示）
- [ ] 并发排队提示（GPU 占用时告知排队位置）

---

## 第七章 · 命令与交互设计

| 用户输入 | Bot 行为 |
|---------|---------|
| 上传 `.cu` 文件 | 下载 → 开始优化 → 推进度 → 推结果 |
| 发送仓库链接 | git clone → 开始优化 → ... |
| `进度` / `status` | 返回当前任务阶段 |
| `历史` / `history` | 返回最近 N 次优化结果列表 |
| `取消` / `cancel` | 终止当前任务 |
| `/help` | 显示命令帮助 |
| 发送其它内容 | 友好提示支持的操作 |

---

## 第八章 · 安全与合规

| 风险 | 应对 |
|------|------|
| 凭证泄露 | CorpID/Secret/AESKey 放环境变量，不入 git（同 MOARK_API_KEY 处理） |
| 企业可信 IP | 白名单只填 C500 出口 IP |
| 文件安全 | 用户上传文件存临时目录，任务完成清理；限制文件类型/大小 |
| 凭证用 env 引用 | `config.py` 读 `os.environ`，提交物料不含明文 |

**合规优势**（答辩亮点）：企业微信是腾讯官方 API，全程合规，无封号风险，可截图证明——这与个人微信 Hook 形成对比。

---

## 第九章 · 与比赛评分的对应

微信 Bot 是 Agent 系统的**可视化前端**，直接提升：

| 评分项 | Bot 的贡献 |
|--------|-----------|
| Agent 可复现性(20%) | 评审方可通过微信实时演示 Agent 优化全过程，复现性"可视化" |
| 文档演示(20%) | 演示视频：微信发文件→Agent优化→收报告，极具说服力 |
| 创新性(加分) | "微信对话即算子优化"是新颖的交互范式 |

---

## 第十章 · 备选方案（若企业微信受限）

若企业微信个人注册受阻，降级方案：

1. **wxauto（RPA）**：Windows 上 UI 自动化操作个人微信，封号风险较低，支持文件。需一台 Windows PC 常驻微信。
2. **WeChatFerry**：个人微信 Hook，功能完整但有封号风险（仅小号测试）。
3. **Server酱 + Web 前端**：Web 页面上传文件，Server酱单向推送结果到微信（牺牲交互性）。

详见 `docs/research/` 下两份调研报告。

---

## 附录 · 关键参考

### 企业微信官方文档
- 接收消息和事件：https://developer.work.weixin.qq.com/document/path/96426
- 发送应用消息：https://developer.work.weixin.qq.com/document/path/90236
- 获取临时素材（下载文件）：https://developer.work.weixin.qq.com/document/path/90254
- 上传临时素材：https://developer.work.weixin.qq.com/document/path/90253
- 被动回复5秒超时与异步推送：https://developer.work.weixin.qq.com/document/path/90238
- 注册（个人组建团队）：https://open.work.weixin.qq.com/help2/pc/15422

### 内网穿透
- cpolar：https://www.cpolar.com/
- frp：https://github.com/fatedier/frp

### 微信 Bot 方案对比（调研报告）
- 详见 `docs/research/` 目录两份报告（个人微信方案 + 企业微信/公众号方案）
- 个人微信 Hook 方案（WeChatFerry/wxauto/WeChatPadPro）均有封号风险，仅作备选
