import type { WinProbMessage } from '../types'

interface PlayLogProps {
  messages: WinProbMessage[]
}

export function PlayLog({ messages }: PlayLogProps) {
  const newestFirst = [...messages].reverse()
  return (
    <ul className="play-log" data-testid="play-log">
      {newestFirst.map((m) => (
        <li key={m.event_index}>
          <span className="play-log-clock">Q{m.period} {m.clock.replace('PT', '').replace('S', '').replace('M', ':')}</span>
          <span className="play-log-desc">{m.description}</span>
        </li>
      ))}
    </ul>
  )
}
