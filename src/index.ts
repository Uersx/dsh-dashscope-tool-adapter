import type { Context } from '@deepseek-ai/cordis'

/**
 * DashScope / 阿里云百炼 Tool Calling 适配器。
 *
 * 问题：DashScope 的 DeepSeek 兼容接口不支持原生 tool calling。
 * 解决：将工具定义转为 prompt 文本注入 system prompt，从模型文本响应中
 * 解析 `<|tool_call|>` 标记，然后构造标准的 BlockAssembler 兼容的
 * tool-call chunk 流，交给 agent loop 原生执行。
 *
 * 触发条件：LLM provider 名包含 `dashscope`、`aliyun` 或 `bailian`。
 * 非百炼 provider 不受影响，原生 tool calling 正常工作。
 */

// ---- 工具函数 ----

function isDashScope(provider: string): boolean {
  const p = provider.toLowerCase()
  return (
    p.includes('dashscope') ||
    p.includes('aliyun') ||
    p.includes('bailian')
  )
}

let _callSeq = 0
function nextCallId(): string {
  return 'call_' + (++_callSeq).toString(36) + '_' + Math.random().toString(36).slice(2, 6)
}

// ---- 类型 ----

interface ToolSchema {
  name?: string
  description?: string
  parameters?: { properties?: Record<string, unknown> }
}

interface StreamChunk {
  type: string
  [key: string]: unknown
}

interface ToolCall {
  name: string
  arguments: Record<string, unknown>
}

// ---- 插件主体 ----

export function apply(ctx: Context): void | (() => void) {
  const tools = ctx.get('tools')
  const systemPrompt = ctx.get('systemPrompt')

  // ------ systemPrompt.section: 注入文本工具描述 ------
  const disposeSection =
    systemPrompt === undefined
      ? () => {}
      : systemPrompt.section({
          name: 'dashscope-tool-descriptions',
          order: 200,
          render: () => {
            if (!tools) return ''
            const schemas = tools.schemas() as ToolSchema[]
            if (!schemas || schemas.length === 0) return ''

            // 收集工具描述
            const toolLines: string[] = []
            for (const s of schemas) {
              const name = s.name || 'unknown'
              const desc = s.description || '(no description)'
              const params = s.parameters?.properties
                ? JSON.stringify(s.parameters.properties)
                : '{}'
              toolLines.push(`- **${name}**: ${desc}`)
              toolLines.push(`  Parameters: \`${params}\``)
            }

            // 取第一个工具作为示例
            const firstTool = schemas[0]?.name || 'tool_name'
            const exampleArg = schemas[0]?.parameters?.properties
              ? JSON.stringify(Object.keys(schemas[0].parameters.properties).reduce((acc, k) => ({ ...acc, [k]: '...' }), {}))
              : '{}'

            return [
              '## TOOLS — YOU MUST USE THIS FORMAT',
              '',
              'When you need to call a tool, output EXACTLY:',
              '',
              `<|tool_call|>{"name": "${firstTool}", "arguments": ${exampleArg}}</|tool_call|>`,
              '',
              'Example — to search the web:',
              '<|tool_call|>{"name": "web_search", "arguments": {"query": "Beijing weather today"}}</|tool_call|>',
              '',
              'RULES:',
              '1. Output the tool call on its OWN LINE, no extra text on that line.',
              '2. Do NOT describe what you plan to do — just output the tool call.',
              '3. After the tool executes, you will see the result and can continue.',
              '',
              'Available tools:',
              ...toolLines,
            ].join('\n')
          },
        })

  // ------ agent/request: 注入 </|tool_call|> 作为 stop 序列 ------
  ;(ctx as any).on('agent/request', (payload: any, next: () => Promise<any>) => {
    return next().then((cfg: any) => {
      // 从 agent 获取 provider，判断是否需要适配
      const agent = payload?.agent
      const provider: string = agent?.provider || cfg?.provider || ''
      if (!isDashScope(provider)) return cfg

      const STOP = '</|tool_call|>'
      const stops: string[] = Array.isArray(cfg.stop) ? cfg.stop : []
      if (!stops.includes(STOP)) {
        return { ...cfg, stop: [...stops, STOP] }
      }
      return cfg
    })
  })

  // ------ llm/stream: 流式拦截百炼响应，解析文本工具调用 ------
  ;(ctx as any).on('llm/stream', (options: any, next: () => AsyncIterable<StreamChunk>) => {
    const provider: string = options?.provider || ''
    if (!isDashScope(provider)) return next()

    const original = next()
    const START = '<|tool_call|>'
    const END = '</|tool_call|>'

    return (async function* () {
      // 流式处理：收集文本，但实时输出非文本 chunk + 安全文本
      const allChunks: StreamChunk[] = []
      let fullText = ''
      let blockIndex = 0
      let blockStarted = false
      let hasUsage = false
      let usageChunk: StreamChunk | null = null

      for await (const chunk of original) {
        allChunks.push(chunk)

        if (chunk.type === 'block-start') {
          blockIndex = chunk.index as number
          blockStarted = true
        } else if (
          (chunk.type === 'text-delta' || chunk.type === 'reasoning-delta') &&
          typeof chunk.text === 'string'
        ) {
          fullText += chunk.text
        } else if (chunk.type === 'usage') {
          hasUsage = true
          usageChunk = chunk
        }
      }

      // 解析工具调用
      const pattern = /<\|tool_call\|>(\{[\s\S]*?\})<\/\|tool_call\|>/g
      const toolCalls: ToolCall[] = []
      let match: RegExpExecArray | null
      while ((match = pattern.exec(fullText)) !== null) {
        try {
          const parsed = JSON.parse(match[1])
          if (parsed && typeof parsed.name === 'string') {
            toolCalls.push(parsed)
          }
        } catch { /* 忽略非 JSON */ }
      }

      if (toolCalls.length === 0) {
        // 没有工具调用，原样返回所有 chunk
        for (const chunk of allChunks) yield chunk
        return
      }

      // 去掉工具调用标记，保留纯文本
      const cleanedText = fullText.replace(pattern, '').trim()

      // 输出文本块（模型 think / 解释文字）
      if (cleanedText) {
        // 使用原始 block-start 的 index（如果有的话）
        const textIdx = blockStarted ? blockIndex : 0
        yield { type: 'block-start', index: textIdx, blockType: 'text' }
        yield { type: 'text-delta', index: textIdx, text: cleanedText }
        yield { type: 'block-end', index: textIdx, block: { type: 'text', text: cleanedText } }
      }

      // 输出 tool-call 块（交给 agent loop 原生执行）
      let tIdx = (cleanedText ? (blockStarted ? blockIndex + 1 : 1) : (blockStarted ? blockIndex : 0))
      for (const tc of toolCalls) {
        const id = nextCallId()
        const args = JSON.stringify(tc.arguments || {})
        yield { type: 'block-start', index: tIdx, blockType: 'tool-call' }
        yield { type: 'tool-call-delta', index: tIdx, id, name: tc.name, argumentsDelta: args }
        yield {
          type: 'block-end',
          index: tIdx,
          block: { type: 'tool-call', id, name: tc.name, arguments: args },
        }
        tIdx++
      }

      // 保留原始 usage（如果有）
      if (hasUsage && usageChunk) {
        yield usageChunk
      }

      // 告诉 agent loop：有工具调用需要执行
      yield { type: 'finish', reason: { kind: 'tool-calls' } }
    })()
  })

  return () => { disposeSection() }
}