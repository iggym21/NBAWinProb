import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { WinProbChart } from '../components/WinProbChart'
import type { WinProbMessage } from '../types'

const sampleMessages: WinProbMessage[] = [
  { event_index: 0, period: 1, clock: 'PT12M00.00S', home_score: 0, away_score: 0, event_type: 'Jump Ball', description: 'Tip', win_prob: 0.5 },
  { event_index: 1, period: 1, clock: 'PT11M42.00S', home_score: 2, away_score: 0, event_type: 'Made Shot', description: 'Dunk', win_prob: 0.55 },
]

describe('WinProbChart', () => {
  it('renders an empty state with no messages', () => {
    render(<WinProbChart messages={[]} />)
    expect(screen.getByTestId('win-prob-chart-empty')).toBeInTheDocument()
  })

  it('renders a chart container when messages are present', () => {
    render(<WinProbChart messages={sampleMessages} />)
    expect(screen.getByTestId('win-prob-chart')).toBeInTheDocument()
  })
})
