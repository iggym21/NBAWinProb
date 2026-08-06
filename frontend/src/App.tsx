import { useMemo, useState } from 'react'
import { GamePicker } from './components/GamePicker'
import { ScoreHeader } from './components/ScoreHeader'
import { WinProbChart } from './components/WinProbChart'
import { PlayLog } from './components/PlayLog'
import { useGameSocket } from './hooks/useGameSocket'
import './App.css'

function buildWsUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = import.meta.env.DEV ? 'localhost:8000' : window.location.host
  return `${protocol}://${host}${path}`
}

function App() {
  const [wsPath, setWsPath] = useState<string | null>(null)
  const wsUrl = useMemo(() => (wsPath ? buildWsUrl(wsPath) : null), [wsPath])
  const { messages, connected, error } = useGameSocket(wsUrl)
  const latest = messages.length > 0 ? messages[messages.length - 1] : null

  return (
    <div className="app">
      <h1>NBA Live Win Probability</h1>
      <GamePicker onSelect={setWsPath} />
      {wsPath && (
        <div className="game-view">
          <div className="connection-status">{connected ? 'Connected' : 'Connecting…'}{error && ` — ${error}`}</div>
          <ScoreHeader latest={latest} homeTeam="Home" awayTeam="Away" />
          <WinProbChart messages={messages} />
          <PlayLog messages={messages} />
        </div>
      )}
    </div>
  )
}

export default App
