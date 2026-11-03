// Reading the engine's UCI output.

export interface Info {
  depth: number
  seldepth?: number
  // Centipawns from the side to move's view, or moves to mate (negative: being mated).
  cp?: number
  mate?: number
  nodes?: number
  nps?: number
  time?: number
  pv: string[]
}

/** Parses an "info depth ... pv ..." line; returns null for other lines. */
export function parseInfo(line: string): Info | null {
  const words = line.trim().split(/\s+/)
  if (words[0] !== 'info' || !words.includes('depth') || !words.includes('pv')) return null
  const info: Info = { depth: 0, pv: [] }
  for (let i = 1; i < words.length; i++) {
    const w = words[i]
    const next = Number(words[i + 1])
    switch (w) {
      case 'depth': info.depth = next; i++; break
      case 'seldepth': info.seldepth = next; i++; break
      case 'nodes': info.nodes = next; i++; break
      case 'nps': info.nps = next; i++; break
      case 'time': info.time = next; i++; break
      case 'score':
        if (words[i + 1] === 'cp') info.cp = Number(words[i + 2])
        else if (words[i + 1] === 'mate') info.mate = Number(words[i + 2])
        i += 2
        break
      case 'pv':
        info.pv = words.slice(i + 1)
        i = words.length
        break
    }
  }
  return info
}

/** The best move of a "bestmove e2e4 [ponder ...]" line, or null. */
export function parseBestMove(line: string): string | null {
  const m = /^bestmove\s+(\S+)/.exec(line.trim())
  return m && m[1] !== '(none)' && m[1] !== '0000' ? m[1] : null
}

/** A score for display, from White's view: "+0.35", "-1.20", "#3", "#-2". */
export function formatScore(info: Pick<Info, 'cp' | 'mate'>, whiteToMove: boolean): string {
  const sign = whiteToMove ? 1 : -1
  if (info.mate !== undefined) return `#${info.mate * sign}`
  const v = ((info.cp ?? 0) * sign) / 100
  return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(2)
}

/** How full the evaluation bar is for White, 0..1: a win probability from the score. */
export function whiteShare(info: Pick<Info, 'cp' | 'mate'>, whiteToMove: boolean): number {
  const sign = whiteToMove ? 1 : -1
  if (info.mate !== undefined) return info.mate * sign > 0 ? 1 : 0
  return 1 / (1 + Math.exp(-((info.cp ?? 0) * sign) / 400))
}
