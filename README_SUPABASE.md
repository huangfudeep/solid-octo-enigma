# Streamlit 问答系统：Supabase 持久化部署

## 文件替换

将本包中的 `app.py`、`db_utils.py`、`requirements.txt` 上传/覆盖到 GitHub 项目根目录。保留 `sample_qa.json` 等原有题库文件。新增 `supabase_schema.sql` 供 Supabase 初始化数据库使用。

## 1. 创建 Supabase 数据库

1. 登录 Supabase，创建一个新 Project。
2. 打开 **SQL Editor**，将 `supabase_schema.sql` 内容粘贴并运行。
3. 在 Project Settings / Database 中找到连接信息，建议优先使用 **Session pooler** 提供的连接字符串（连接方式、端口以 Supabase 控制台显示为准）。连接串需包含用户名、密码、主机、端口和数据库名；不要把密码提交到 GitHub。

## 2. 设置 Streamlit Cloud Secrets

进入 Streamlit Cloud 应用 → **Settings → Secrets**，保留现有模型与管理员配置，并新增：

```toml
SUPABASE_DB_URL = "postgresql://postgres.<项目标识>:<数据库密码>@<pooler主机>:5432/postgres?sslmode=require"
```

请以 Supabase 控制台实际给出的连接串为准，不要照抄示例中的占位符。保留现有 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`CHAT_MODEL`、`EMBED_PROVIDER`、`EMBED_MODEL`、`EMBED_DIMENSIONS`、`ADMIN_PASSWORD` 等 Secrets。Ark 设置仍使用你当前已配置的值。

如果密码中包含 `@`、`:`、`/`、`#` 等特殊字符，请使用 Supabase 提供的连接字符串，或先对密码进行 URL 编码。

## 3. 部署

1. 将以上三个文件提交到 GitHub 项目根目录。
2. 确认 Streamlit Cloud 指向正确仓库和分支，然后等待重新部署。
3. 启动后先进入管理员端，导入 `sample_qa.json` 或自己的 JSON 题库。
4. 在管理员后台的“知识库管理”标签中点击 **一键导出 JSON 完整备份**，将文件保存到自己的电脑。
5. 测试新增问答、学生提问、待处理问题入库、API 开关和 JSON 导入。
6. 重新启动/重新部署一次，再检查问答和待处理问题是否仍在。这是验证持久化是否成功的重要步骤。

## 数据保存说明

- 问答、向量、待处理问题和 API 开关均保存在 Supabase PostgreSQL。
- 不再把 ChromaDB 本地目录作为唯一检索数据源；向量保存在 PostgreSQL 的 JSONB 列中，缺失的向量会在检索时生成并保存。
- API 开启时，使用配置的 Ark/OpenAI 兼容嵌入服务；API 关闭时，使用本地 Sentence Transformers 模型，不调用 Ark 嵌入 API。
- JSON 导入的相似度阈值仍由 `SIMILARITY_THRESHOLD` 控制（默认 0.70）；学生端阈值仍为 0.60。
- JSON 备份会导出知识库、待处理/已处理问题和 API 开关状态；不会导出 API 密钥或管理员密码。恢复时可上传导出的 JSON，系统会读取其中的 `knowledge_base` 数组并跳过已存在的问题。
- 备份下载后请保存到个人电脑或其他安全位置。外部数据库仍建议定期导出备份。

## 常见问题

- `未配置 SUPABASE_DB_URL`：检查 Streamlit Cloud → Settings → Secrets 的变量名和 TOML 格式。
- `password authentication failed`：重新复制 Supabase Database 的连接串，检查用户名、密码和端口。
- 连接超时：确认使用 Supabase 控制台提供的适合当前网络的连接方式/Session pooler，并确认项目未暂停。
- 首次使用本地检索时可能需要下载 Sentence Transformers 模型，启动时间会较长。
