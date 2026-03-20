# Whisper 私聊主动消息插件

## 插件简介

Whisper 是一款基于对话感知的私聊主动消息插件，适用于 AstrBot。当用户在私聊中沉默一段时间后，插件会调用 LLM 智能判断是否应该主动发送消息，模拟更自然的聊天体验。

## 功能特性

- **智能判断**: 利用 LLM 分析对话上下文，决定是否主动发消息
- **沉默检测**: 用户沉默一段时间后自动触发主动消息
- **安静时段**: 可配置夜间免打扰时段，避免休息时间发送消息
- **防骚扰机制**: 限制连续主动消息数量，避免过度打扰
- **消息分段**: 长消息自动分段发送，带有间隔延迟
- **会话隔离**: 支持为不同会话配置不同的参数
- **状态控制**: 提供命令实时启用/禁用插件

## MCP 状态感知（可选功能）

从 v0.6.0 开始，Whisper 支持通过 MCP（Model Context Protocol）获取外部状态信息，辅助 LLM 做出更智能的主动消息决策。

### 功能说明

- **MCP 状态感知**: 插件会定期通过 MCP 服务查询外部状态（如 Spotify 播放状态），并将状态信息纳入 LLM 决策上下文
- **智能决策增强**: LLM 可以根据用户当前的外部状态（如正在听音乐、正在会议中）决定是否发送消息
- **状态检查**: 用户沉默触发检查时，插件会先通过 MCP 获取最新状态，再将状态信息发送给 LLM

### 前置要求

1. **独立运行 MCP 服务器**: 你需要自行运行 MCP 服务器（如 Spotify MCP Server）。插件本身不包含 MCP 服务器
2. **配置 MCP 服务**: 在配置中指定 MCP 服务的连接信息

### 配置示例

```yaml
mcp_enabled: true
mcp_services:
  - "spotify"
spotify_mcp_command: "node data/mcp_servers/spotify-mcp-server/build/index.js"
```

### 注意事项

- MCP 功能默认关闭，需将 `mcp_enabled` 设置为 `true` 才会启用
- 目前插件仅支持读取 MCP 状态，不支持调用 MCP 工具
- 如未运行 MCP 服务器，插件会跳过状态检查，正常执行主动消息判断

## 安装方法

1. 确保已安装 AstrBot 4.14 或更高版本
2. 将插件文件夹复制到 AstrBot 插件目录
3. 在 AstrBot 管理面板中启用插件
4. 根据需要调整配置参数

## 配置说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable` | bool | true | 启用 Whisper 主动消息功能 |
| `llm_provider_id` | string | "" | 主动消息生成使用的 LLM 提供商（留空使用当前会话模型） |
| `silence_trigger_minutes` | int | 5 | 用户沉默触发检查时间（分钟） |
| `timeout_max` | int | 30 | 最大沉默超时（分钟） |
| `max_consecutive` | int | 3 | 最大连续主动消息数 |
| `quiet_hours_enabled` | bool | true | 启用安静时段 |
| `quiet_hours_start` | string | "23:00" | 安静时段开始时间 |
| `quiet_hours_end` | string | "08:00" | 安静时段结束时间 |
| `max_history_messages` | int | 20 | 发给 LLM 的最大历史消息数 |
| `segment_enabled` | bool | true | 启用分段发送 |
| `segment_threshold` | int | 150 | 不分段字数阈值（超过此值不分段） |
| `segment_mode` | string | "regex" | 分段模式（`regex` / `words`） |
| `segment_regex` | string | `.*?[。？！~…\n]+|.+$` | 正则分段规则（`segment_mode=regex`） |
| `segment_words` | string | "。！？～…\n" | 分段词列表（`segment_mode=words`） |
| `segment_delay_ms` | int | 1500 | 分段发送间隔（毫秒） |
| `proactive_prompt` | text | (见配置页面) | 主动消息判断提示词模板 |
| `mcp_enabled` | bool | false | 启用 MCP 状态感知 |
| `mcp_services` | list | [] | MCP 服务列表（当前支持填 `spotify`） |
| `spotify_context_enabled` | bool | false | 启用 Spotify 播放状态上下文 |
| `spotify_suggest_enabled` | bool | false | 启用 Spotify 推荐建议附加 |
| `spotify_mcp_command` | string | `node data/mcp_servers/spotify-mcp-server/build/index.js` | Spotify MCP 服务器启动命令 |

兼容性说明：历史配置里的 `timeout_min` 仍被兼容读取，但推荐使用 `silence_trigger_minutes`。

## 命令说明

| 命令 | 说明 |
|------|------|
| `/whisper` | 查看当前 Whisper 状态 |
| `/whisper_on` | 启用 Whisper 主动消息 |
| `/whisper_off` | 禁用 Whisper 主动消息 |

## 使用示例

**查看状态**
```
/whisper
```
输出当前插件启用状态、沉默超时设置、未回复计数等信息。

**启用插件**
```
/whisper_on
```
在当前私聊会话中启用主动消息功能。

**禁用插件**
```
/whisper_off
```
在当前私聊会话中禁用主动消息功能。

## 工作原理

1. 当用户发送消息后，插件开始计时
2. 用户沉默超过随机超时时间后，插件触发检查
3. 将对话历史发送给 LLM，由 LLM 判断是否应该主动发消息
4. 如果 LLM 决定发送，插件会自动发送消息
5. 发送后增加未回复计数，并在达到上限前继续尝试

## 注意事项

- 首次使用建议将 `max_consecutive` 设置较小值，观察效果后再调整
- 安静时段默认设为 23:00 至次日 08:00，可根据需要修改
- `session_configs` 支持为特定会话单独配置参数，key 为会话 ID
