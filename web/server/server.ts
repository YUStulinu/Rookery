// Serves the web page and connects each browser tab to its own Rookery process over a WebSocket.
//
//   node server/server.ts [--engine path/to/rookery] [--port 3021]
//
// Messages from the page:  {type: "go", fen, moves: string[], movetime?, depth?, eval: "nnue" | "classical"}
//                          {type: "stop"}
// Messages to the page:    {type: "info", ...Info}  {type: "bestmove", move}  {type: "error", message}
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { dirname, extname, join, normalize, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { WebSocketServer, type WebSocket } from 'ws'
import { parseBestMove, parseInfo } from './uci.ts'

const here = dirname(fileURLToPath(import.meta.url))
const args = process.argv.slice(2)
const arg = (name: string, fallback: string) => {
  const i = args.indexOf(name)
  return i >= 0 ? args[i + 1] : fallback
}
const exe = process.platform === 'win32' ? 'rookery.exe' : 'rookery'
const enginePath = resolve(arg('--engine', join(here, '..', '..', 'engine', 'build', exe)))
const port = Number(arg('--port', '3021'))
const dist = join(here, '..', 'dist')

const TYPES: Record<string, string> = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' }

const http = createServer((req, res) => {
  const path = normalize(decodeURIComponent((req.url ?? '/').split('?')[0])).replace(/^(\.\.[/\\])+/, '')
  const file = join(dist, path === '/' || path === '\\' ? 'index.html' : path)
  if (!file.startsWith(dist) || !existsSync(file)) {
    res.writeHead(404).end('not found')
    return
  }
  res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' }).end(readFileSync(file))
})

const wss = new WebSocketServer({ server: http, path: '/engine' })

interface GoMessage {
  type: 'go'
  fen: string
  moves?: string[]
  movetime?: number
  depth?: number
  eval?: 'nnue' | 'classical'
}

wss.on('connection', (socket: WebSocket) => {
  if (!existsSync(enginePath)) {
    socket.send(JSON.stringify({ type: 'error', message: `engine not found at ${enginePath}; build it first` }))
    socket.close()
    return
  }
  const engine: ChildProcessWithoutNullStreams = spawn(enginePath, [], { cwd: dirname(enginePath) })
  const send = (m: object) => socket.readyState === socket.OPEN && socket.send(JSON.stringify(m))
  let buffer = ''
  let evalMode: 'nnue' | 'classical' | null = null
  engine.stdout.on('data', (chunk: Buffer) => {
    buffer += chunk.toString()
    let nl: number
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim()
      buffer = buffer.slice(nl + 1)
      const info = parseInfo(line)
      if (info) send({ type: 'info', ...info })
      const best = line.startsWith('bestmove') ? parseBestMove(line) : undefined
      if (best !== undefined) send({ type: 'bestmove', move: best })
      if (line.startsWith('info string NNUE not loaded')) send({ type: 'error', message: line.slice(12) })
    }
  })
  const write = (s: string) => engine.stdin.write(s + '\n')
  write('uci')
  write('isready')

  socket.on('message', (raw) => {
    let msg: GoMessage | { type: 'stop' }
    try {
      msg = JSON.parse(raw.toString())
    } catch {
      return
    }
    if (msg.type === 'stop') {
      write('stop')
    } else if (msg.type === 'go') {
      if (msg.eval && msg.eval !== evalMode) {
        evalMode = msg.eval
        write(`setoption name UseNNUE value ${msg.eval === 'nnue'}`)
      }
      // Only what a FEN and UCI moves can contain reaches the engine.
      const fen = String(msg.fen).replace(/[^\w/ -]/g, '')
      const moves = (msg.moves ?? []).filter((m) => /^[a-h][1-8][a-h][1-8][nbrq]?$/.test(m))
      write('stop')
      write(`position fen ${fen}${moves.length ? ' moves ' + moves.join(' ') : ''}`)
      const limit = msg.depth ? `depth ${Math.min(Math.max(1, msg.depth | 0), 60)}` : `movetime ${Math.min(Math.max(50, msg.movetime ?? 1000), 60000)}`
      write(`go ${limit}`)
    }
  })
  socket.on('close', () => {
    write('quit')
    setTimeout(() => engine.kill(), 500)
  })
})

http.listen(port, () => {
  console.log(`Rookery web on http://localhost:${port} (engine: ${enginePath})`)
})
