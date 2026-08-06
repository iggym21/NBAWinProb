import type { WinProbMessage } from '../types'

interface ScoreHeaderProps {
  latest: WinProbMessage | null
  homeTeam: string
  awayTeam: string
}

export function ScoreHeader({ latest, homeTeam, awayTeam }: ScoreHeaderProps) {
  if (!latest) {
    return <div className="score-header">Waiting for tip-off…</div>
  }
  const homeWinPct = Math.round(latest.win_prob * 100)
  return (
    <div className="score-header">
      <div className="score-header-teams">
        <span>{awayTeam} {latest.away_score}</span>
        <span className="score-header-divider">—</span>
        <span>{latest.home_score} {homeTeam}</span>
      </div>
      <div className="score-header-clock">Q{latest.period} · {latest.clock.replace('PT', '').replace('S', '').replace('M', ':')}</div>
      <div className="score-header-winprob">{homeTeam} win probability: {homeWinPct}%</div>
    </div>
  )
}
