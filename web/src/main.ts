import { Chess, type Move, type Square } from 'chess.js'
import { formatScore, whiteShare, type Info } from '../server/uci.ts'

const GLYPHS: Record<string, string> = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' }
const FILES = 'abcdefgh'

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const boardEl = $('board')
const statusEl = $('status')
const modeEl = $<HTMLSelectElement>('mode')
const evalEl = $<HTMLSelectElement>('eval')
const movetimeEl = $<HTMLSelectElement>('movetime')
const fenEl = $<HTMLInputElement>('fen')

let game = new Chess()
let startFen = game.fen()
let flipped = false
let selected: Square | null = null
let thinking = false
let lastInfo: Info | null = null
// The position the shown engine output is about (it stays on screen after the engine moves).
let infoFen = game.fen()
let lastMove: Move | null = null

// The engine connection; the server starts one engine process per page.
const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/engine`)
socket.addEventListener('message', (e) => {
  const msg = JSON.parse(e.data)
  if (msg.type === 'info') {
    lastInfo = msg
    showInfo()
  } else if (msg.type === 'bestmove') {
    thinking = false
    if (msg.move && engineToMove()) {
      play({ from: msg.move.slice(0, 2), to: msg.move.slice(2, 4), promotion: msg.move[4] })
    }
  } else if (msg.type === 'error') {
    statusEl.textContent = msg.message
  }
})
socket.addEventListener('open', () => update())
socket.addEventListener('close', () => (statusEl.textContent = 'The engine server is not running (npm start).'))

function send(m: object) {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(m))
}

function humanColor(): 'w' | 'b' | null {
  return modeEl.value === 'white' ? 'w' : modeEl.value === 'black' ? 'b' : null
}

function engineToMove(): boolean {
  const human = humanColor()
  return human !== null && game.turn() !== human && !game.isGameOver()
}

function ask() {
  if (game.isGameOver()) {
    send({ type: 'stop' })
    return
  }
  const moves = game.history({ verbose: true }).map((m) => m.from + m.to + (m.promotion ?? ''))
  const analysing = modeEl.value === 'analyse'
  if (!analysing && !engineToMove()) {
    send({ type: 'stop' })
    return
  }
  lastInfo = null
  infoFen = game.fen()
  thinking = !analysing
  showInfo()
  send({ type: 'go', fen: startFen, moves, eval: evalEl.value, movetime: analysing ? 15000 : Number(movetimeEl.value) })
}

function play(m: { from: string; to: string; promotion?: string }): boolean {
  try {
    lastMove = game.move(m)
  } catch {
    return false
  }
  selected = null
  update()
  return true
}

function update() {
  drawBoard()
  drawMoves()
  fenEl.value = game.fen()
  statusEl.textContent = status()
  ask()
}

function status(): string {
  if (game.isCheckmate()) return `Checkmate: ${game.turn() === 'w' ? 'Black' : 'White'} wins.`
  if (game.isStalemate()) return 'Stalemate: a draw.'
  if (game.isThreefoldRepetition()) return 'Draw by repetition.'
  if (game.isInsufficientMaterial()) return 'Draw: not enough material to mate.'
  if (game.isDraw()) return 'Draw by the fifty-move rule.'
  const side = game.turn() === 'w' ? 'White' : 'Black'
  return `${side} to move${game.inCheck() ? ', in check' : ''}.`
}

function drawBoard() {
  boardEl.replaceChildren()
  const targets = new Set(selected ? game.moves({ square: selected, verbose: true }).map((m) => m.to) : [])
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const rank = flipped ? row + 1 : 8 - row
      const file = flipped ? 7 - col : col
      const sq = `${FILES[file]}${rank}` as Square
      const cell = document.createElement('button')
      cell.className = `sq ${(file + rank) % 2 ? 'light' : 'dark'}`
      cell.setAttribute('aria-label', sq)
      if (sq === selected) cell.classList.add('selected')
      if (lastMove && (sq === lastMove.from || sq === lastMove.to)) cell.classList.add('last')
      if (targets.has(sq)) cell.classList.add('target')
      const piece = game.get(sq)
      if (piece) {
        const span = document.createElement('span')
        span.className = `piece ${piece.color === 'w' ? 'white' : 'black'}`
        span.textContent = GLYPHS[piece.type]
        cell.append(span)
        if (piece.type === 'k' && piece.color === game.turn() && game.inCheck()) cell.classList.add('check')
      }
      if (col === 0) cell.dataset.rank = String(rank)
      if (row === 7) cell.dataset.file = FILES[file]
      cell.addEventListener('click', () => click(sq))
      boardEl.append(cell)
    }
  }
}

function click(sq: Square) {
  const human = humanColor()
  if (thinking || game.isGameOver() || (human && game.turn() !== human)) return
  const piece = game.get(sq)
  if (selected) {
    const options = game.moves({ square: selected, verbose: true }).filter((m) => m.to === sq)
    if (options.length) {
      if (options.length > 1) choosePromotion(selected, sq)
      else play({ from: selected, to: sq })
      return
    }
  }
  selected = piece && piece.color === game.turn() ? sq : null
  drawBoard()
}

function choosePromotion(from: Square, to: Square) {
  const box = $('promotion')
  box.replaceChildren()
  for (const p of ['q', 'r', 'b', 'n']) {
    const b = document.createElement('button')
    b.textContent = GLYPHS[p]
    b.className = `piece ${game.turn() === 'w' ? 'white' : 'black'}`
    b.setAttribute('aria-label', { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' }[p]!)
    b.addEventListener('click', () => {
      box.hidden = true
      play({ from, to, promotion: p })
    })
    box.append(b)
  }
  box.hidden = false
}

function drawMoves() {
  const list = $('moves')
  list.replaceChildren()
  const history = game.history()
  for (let i = 0; i < history.length; i += 2) {
    const li = document.createElement('li')
    li.textContent = `${history[i]} ${history[i + 1] ?? ''}`
    list.append(li)
  }
  list.scrollTop = list.scrollHeight
}

function showInfo() {
  const line = $('engine-line')
  const pv = $('pv')
  const bar = $('evalbar')
  const label = $('evalbar-label')
  if (!lastInfo) {
    line.textContent = thinking ? 'thinking…' : '–'
    pv.textContent = ''
    return
  }
  const white = infoFen.split(' ')[1] === 'w'
  const score = formatScore(lastInfo, white)
  const knps = lastInfo.nps ? `${Math.round(lastInfo.nps / 1000)} kn/s` : ''
  line.textContent = `${score} · depth ${lastInfo.depth} · ${knps} · ${evalEl.value === 'nnue' ? 'neural network' : 'classical'}`
  bar.style.height = `${whiteShare(lastInfo, white) * 100}%`
  label.textContent = score
  // The principal variation in standard notation, played out on a copy.
  const copy = new Chess(infoFen)
  const san: string[] = []
  for (const m of lastInfo.pv) {
    try {
      san.push(copy.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] }).san)
    } catch {
      break
    }
  }
  pv.textContent = san.join(' ')
}

$('new').addEventListener('click', () => {
  lastInfo = null
  game = new Chess()
  startFen = game.fen()
  lastMove = null
  selected = null
  flipped = modeEl.value === 'black'
  update()
})
$('undo').addEventListener('click', () => {
  // Take back the engine's reply too, so it is the player's turn again.
  game.undo()
  if (humanColor() && game.turn() !== humanColor()) game.undo()
  lastMove = null
  update()
})
$('flip').addEventListener('click', () => {
  flipped = !flipped
  drawBoard()
})
$('load').addEventListener('click', () => {
  try {
    game = new Chess(fenEl.value.trim())
    startFen = game.fen()
    lastMove = null
    update()
  } catch (e) {
    statusEl.textContent = `Not a valid FEN: ${(e as Error).message}`
  }
})
modeEl.addEventListener('change', () => {
  flipped = modeEl.value === 'black'
  update()
})
evalEl.addEventListener('change', () => ask())

// A position and mode can be given in the address: ?fen=...&mode=analyse
const params = new URLSearchParams(location.search)
if (params.get('mode')) modeEl.value = params.get('mode')!
if (params.get('fen')) {
  try {
    game = new Chess(params.get('fen')!)
    startFen = game.fen()
  } catch {
    statusEl.textContent = 'The fen in the address is not valid.'
  }
}
flipped = modeEl.value === 'black'
drawBoard()
