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

/** 生成一个简单的 call id */
function callId(): string {
  return 'dash_' + Math.random().toString(36).slice(2, 10)
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
            const lines: string[] = [
              '## Available Tools',
              '',
              'To call a tool, output a SINGLE LINE in this EXACT format:',
              '',
              '<|tool_call|>{"name": "<tool_name>", "arguments": {<json_args>}}</|tool_call|>',
              '',
              'The system will execute the tool and show the result.',
              '',
            ]
            for (const s of schemas) {
              const name = s.name || 'unknown'
              const desc = s.description || '(no description)'
              const params = s.parameters?.properties
                ? JSON.stringify(s.parameters.properties)
                : '{}'
              lines.push(`- **${name}**: ${desc}`)
              lines.push(`  Parameters: \`${params}\``)
            }
            return lines.join('\n')
          },
        })

  // ------ llm/stream: 拦截百炼响应，解析文本工具调用 ------
  ;(ctx as any).on('llm/stream', (options: any, next: () => AsyncIterable<StreamChunk>) => {
    const provider: string = options?.provider || ''
    if (!isDashScope(provider)) return next()

    const original = next()
    const START = '<|tool_call|>'
    const END = '</|tool_call|>'

    return (async function* () {
      // 收集所有原始 chunk 和文本
      const chunks: StreamChunk[] = []
      let fullText = ''

      for await (const chunk of original) {
        chunks.push(chunk)
        if (chunk.type === 'text-delta' && typeof chunk.text === 'string') {
          fullText += chunk.text
        }
      }

      // 解析工具调用
      const pattern = new RegExp(
        START.replace(/[|]/g, '\\|') + '(\\{[\\s\\S]*?\\})' + END.replace(/[|]/g, '\\|'),
        'g',
      )
      const toolCalls: Array<{ name: string; arguments: Record<string, unknown> }> = []
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
        // 没有工具调用，原样返回
        for (const chunk of chunks) yield chunk
        return
      }

      // 去掉工具调用标记，保留纯文本
      const cleanedText = fullText.replace(pattern, '').trim()

      // 输出文本块（模型 think / 解释文字）
      if (cleanedText) {
        yield { type: 'block-start', index: 0, blockType: 'text' }
        yield { type: 'text-delta', index: 0, text: cleanedText }
        yield { type: 'block-end', index: 0, block: { type: 'text', text: cleanedText } }
      }

      // 输出 tool-call 块（交给 agent loop 原生执行）
      let idx = cleanedText ? 1 : 0
      for (const tc of toolCalls) {
        const id = callId()
        const args = JSON.stringify(tc.arguments || {})
        yield { type: 'block-start', index: idx, blockType: 'tool-call' }
        yield { type: 'tool-call-delta', index: idx, id, name: tc.name, argumentsDelta: args }
        yield {
          type: 'block-end',
          index: idx,
          block: { type: 'tool-call', id, name: tc.name, arguments: args },
        }
        idx++
      }

      // 告诉 agent loop：有工具调用需要执行
      yield { type: 'finish', reason: { kind: 'tool-calls' } }
    })()
  })

  return () => { disposeSection() }
}