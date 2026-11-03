import { describe, expect, it } from 'vitest'
import { formatScore, parseBestMove, parseInfo, whiteShare } from '../server/uci.ts'

describe('parseInfo', () => {
  it('reads depth, score and the principal variation', () => {
    const i = parseInfo('info depth 12 seldepth 18 score cp -35 nodes 120000 nps 900000 hashfull 12 time 133 pv e7e5 g1f3 b8c6')
    expect(i).toEqual({ depth: 12, seldepth: 18, cp: -35, nodes: 120000, nps: 900000, time: 133, pv: ['e7e5', 'g1f3', 'b8c6'] })
  })
  it('reads mate scores', () => {
    expect(parseInfo('info depth 5 score mate -2 nodes 10 pv h7h8')?.mate).toBe(-2)
  })
  it('ignores other lines', () => {
    expect(parseInfo('info string NNUE loaded')).toBeNull()
    expect(parseInfo('readyok')).toBeNull()
  })
})

describe('parseBestMove', () => {
  it('reads the move and rejects null moves', () => {
    expect(parseBestMove('bestmove e2e4')).toBe('e2e4')
    expect(parseBestMove('bestmove e7e8q ponder a1a2')).toBe('e7e8q')
    expect(parseBestMove('bestmove 0000')).toBeNull()
  })
})

describe('scores from White’s view', () => {
  it('flips the sign when Black is to move', () => {
    expect(formatScore({ cp: 35 }, true)).toBe('+0.35')
    expect(formatScore({ cp: 35 }, false)).toBe('−0.35')
    expect(formatScore({ mate: 3 }, false)).toBe('#-3')
    expect(formatScore({ cp: 0 }, true)).toBe('0.00')
  })
  it('turns scores into a share of the bar', () => {
    expect(whiteShare({ cp: 0 }, true)).toBeCloseTo(0.5)
    expect(whiteShare({ cp: 400 }, false)).toBeLessThan(0.3)
    expect(whiteShare({ mate: 2 }, true)).toBe(1)
  })
})
