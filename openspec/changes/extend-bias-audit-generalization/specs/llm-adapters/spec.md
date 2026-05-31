## ADDED Requirements

### Requirement: 统一 LLM 调用接口

系统 SHALL 提供 `BaseLLMAdapter` 抽象类，所有具体 provider 实现 MUST 暴露相同的 `generate(prompt: str, **gen_kwargs) -> LLMResponse` 方法。`LLMResponse` MUST 包含 `text, prompt_tokens, completion_tokens, model, latency_ms, retries, finish_reason` 字段。

#### Scenario: 调用任意 provider 返回统一结构
- **WHEN** 用户用任一 adapter 实例调用 `generate("hello")`
- **THEN** 返回的 `LLMResponse` 字段集合完全一致，下游代码无需感知 provider

#### Scenario: 切换 provider 不改业务代码
- **WHEN** 实验配置把 `model` 从 `gpt-4.1-mini` 切到 `claude-3-5-sonnet-20241022`
- **THEN** 编排器调用方式不变，仅通过工厂函数返回不同 adapter

### Requirement: 多 provider 支持

系统 SHALL 至少支持 OpenAI 兼容、Anthropic、Together/SiliconFlow（OpenAI 兼容协议）三类 provider，对应实现 `OpenAIAdapter`、`AnthropicAdapter`、`TogetherAdapter`。

#### Scenario: GPT-4.1-mini 通过 OpenAI 兼容接口调用
- **WHEN** 加载 `OpenAIAdapter(model="gpt-4.1-mini", base_url="https://api.laozhang.ai/v1")`
- **THEN** 调用 `generate` 返回正常文本与 token 计数

#### Scenario: Claude 3.5 Sonnet 通过 Anthropic 接口调用
- **WHEN** 加载 `AnthropicAdapter(model="claude-3-5-sonnet-20241022")`
- **THEN** 调用 `generate` 返回 `LLMResponse`，且 `model` 字段精确反映调用模型

#### Scenario: Qwen2.5-72B 通过 SiliconFlow 调用
- **WHEN** 加载 `TogetherAdapter(model="Qwen/Qwen2.5-72B-Instruct", base_url="https://api.siliconflow.cn/v1")`
- **THEN** 返回中文 prompt 的中文回复

### Requirement: 限速、重试、退避

系统 SHALL 在所有 adapter 内置指数退避重试机制，最大重试次数 6 次，初始等待 1 秒，最大等待 60 秒。HTTP 4xx 非限流错误不重试。

#### Scenario: 429 限流自动重试成功
- **WHEN** provider 返回 HTTP 429 一次后正常返回
- **THEN** adapter 自动重试，最终返回成功响应，`retries` 字段记录为 1

#### Scenario: 401 鉴权错误立即终止
- **WHEN** provider 返回 HTTP 401
- **THEN** adapter 抛出 `AuthError` 不再重试

### Requirement: 响应缓存

系统 SHALL 基于 `(model, prompt_sha256, temperature, top_p, max_tokens)` 为 key 缓存 LLM 响应到本地 SQLite 数据库 `data/cache/llm_cache.sqlite`，命中时直接返回不调用 API。

#### Scenario: 同 prompt 二次调用走缓存
- **WHEN** 同 prompt 同参数第二次调用
- **THEN** 返回相同响应，`latency_ms < 50`，token 计费记录为 0

#### Scenario: 缓存失效条件
- **WHEN** prompt 改 1 个字符或 temperature 改变
- **THEN** 缓存未命中，重新调用 API

### Requirement: Token 计费记录

系统 SHALL 把每次实际 API 调用（非缓存命中）的 `model, prompt_tokens, completion_tokens, timestamp, experiment_id` 追加写入 `data/cache/usage_log.jsonl`，用于成本审计。

#### Scenario: 实验结束查总成本
- **WHEN** 一次实验完成
- **THEN** 通过 `usage_log.jsonl` 可以聚合出该 experiment_id 的总 prompt/completion tokens

### Requirement: API key 通过环境变量注入

所有 adapter MUST 从环境变量读取 API key，禁止硬编码。仓库 MUST 提供 `.env.example` 示例，禁止提交真实 `.env`。

#### Scenario: 缺失环境变量启动失败
- **WHEN** `OPENAI_API_KEY` 未设置且实验配置要求 OpenAI
- **THEN** 程序在加载阶段抛出明确错误，提示设置环境变量
