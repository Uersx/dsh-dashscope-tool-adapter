# dsh-dashscope-tool-adapter

DSH plugin: adapts [DashScope](https://dashscope.aliyun.com) / 阿里云百炼 DeepSeek endpoints that lack native tool calling.

## Problem

DashScope's OpenAI-compatible endpoint (`https://dashscope.aliyuncs.com/compatible-mode/v1`) does not support native tool calling (function calling) for DeepSeek models. When DSH sends tool definitions, the API either ignores them or rejects the request.

## Solution

This plugin converts tool definitions into prompt text and parses the model's text response for `<|tool_call|>` markers. It then constructs standard BlockAssembler-compatible tool-call chunks, which the DSH agent loop executes natively.

- **Non-DashScope providers**: untouched, native tool calling works as usual.
- **DashScope providers**: tools are described in the system prompt, and the model calls them via text markers.

## Installation

```bash
dsh plugin add github:Uersx/dsh-dashscope-tool-adapter
```

Or manually:

```bash
dsh plugin --profile web add github:Uersx/dsh-dashscope-tool-adapter
```

## How it works

1. A `systemPrompt.section()` injects text-based tool descriptions into the system prompt.
2. The `llm/stream` waterfall intercepts only DashScope calls.
3. The model outputs `<|tool_call|>{"name":"...","arguments":{...}}</|tool_call|>` in its text.
4. The plugin parses these markers and constructs standard `tool-call-delta` / `block-end` chunks.
5. The DSH agent loop executes the tools natively.

## License

MIT