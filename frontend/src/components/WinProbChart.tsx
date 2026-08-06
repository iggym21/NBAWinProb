import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import type { WinProbMessage } from '../types'

interface WinProbChartProps {
  messages: WinProbMessage[]
}

export function WinProbChart({ messages }: WinProbChartProps) {
  if (messages.length === 0) {
    return <div data-testid="win-prob-chart-empty">Waiting for game data…</div>
  }

  const data = messages.map((m) => ({ event_index: m.event_index, win_prob: m.win_prob }))

  return (
    <div data-testid="win-prob-chart" style={{ width: '100%', height: 320 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="event_index" label={{ value: 'Event', position: 'insideBottom', offset: -4 }} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <ReferenceLine y={0.5} stroke="#999" strokeDasharray="4 4" />
          <Tooltip formatter={(value) => `${Math.round(Number(value) * 100)}%`} />
          <Line type="monotone" dataKey="win_prob" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
