# 知识库 AI 助手（Streamlit）

一个基于 Streamlit、SQLite、ChromaDB 和 OpenAI 兼容模型 API 的中文知识库问答应用。

## 功能

- 学生端提问、知识库语义检索、未命中问题记录
- 管理员密码登录
- 待处理问题审核、正式答案入库
- 知识库问答新增、删除、JSON 批量导入
- 基于余弦相似度的阈值过滤

## 本地运行

推荐 Python 3.11。

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，填写 `OPENAI_API_KEY` 和 `ADMIN_PASSWORD`。然后运行：

```bash
streamlit run app.py
```

首次打开网页后进入管理员端，点击“一键导入 sample_qa.json”。也可以在终端运行 `python import_json.py` 导入。

## 部署到 Streamlit Community Cloud

1. 将本项目文件上传到你自己的 GitHub 仓库（不要上传 `.env`、API Key、`data/` 或 `secrets.toml`）。
2. 登录 Streamlit Community Cloud，选择 **Create app**，选择仓库、分支和 `app.py`。
3. 在应用的 **Settings → Secrets** 中添加：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
CHAT_MODEL = "gpt-4o-mini"
EMBED_PROVIDER = "openai"
EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = "0.70"
ADMIN_PASSWORD = "请替换为强密码"
```

4. 保存并等待构建完成。启动后进入管理员端，导入 `sample_qa.json`。

## 部署注意事项

- 代码已兼容 Streamlit Secrets 和环境变量。真实密钥只能放在 Secrets 中，不要写进代码或提交到 GitHub。
- Streamlit Community Cloud 的本地文件系统不适合保证长期持久化。SQLite 与 Chroma 数据可能在重新部署、重建容器或平台回收后丢失；正式长期使用建议迁移到托管数据库和持久化向量数据库。
- 本项目默认使用 API embedding，部署时不需要下载大型本地向量模型。若设置 `EMBED_PROVIDER=local`，需额外安装 `sentence-transformers`。
- 大模型临时回答不是实时联网检索结果，赛事政策请以官方通知为准。
