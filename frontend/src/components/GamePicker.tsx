import { useEffect, useState } from 'react'
import type { GamesResponse, GameSummary } from '../types'

interface GamePickerProps {
  onSelect: (wsPath: string) => void
}

export function GamePicker({ onSelect }: GamePickerProps) {
  const [games, setGames] = useState<GameSummary[]>([])
  const [liveAvailable, setLiveAvailable] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/games')
      .then((res) => res.json())
      .then((data: GamesResponse) => {
        setGames(data.replay_games)
        setLiveAvailable(data.live_available)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  if (loading) {
    return <div>Loading games…</div>
  }

  return (
    <div className="game-picker">
      <button disabled={!liveAvailable} onClick={() => onSelect('/live')}>
        Watch Live
      </button>
      <ul>
        {games.map((g) => (
          <li key={g.game_id}>
            <button onClick={() => onSelect(`/replay/${g.game_id}`)}>
              {g.home_team} vs {g.away_team}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
