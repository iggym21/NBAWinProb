import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { GamePicker } from '../components/GamePicker'

beforeEach(() => {
  // @ts-expect-error test override
  global.fetch = vi.fn(async () => ({
    ok: true,
    json: async () => ({
      replay_games: [{ game_id: 'g1', home_team: 'DEN', away_team: 'LAL' }],
      live_available: false,
    }),
  })) as unknown as typeof fetch
})

describe('GamePicker', () => {
  it('fetches and renders replay games, disables live when unavailable', async () => {
    const onSelect = vi.fn()
    render(<GamePicker onSelect={onSelect} />)

    await waitFor(() => expect(screen.getByText(/DEN vs LAL/i)).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /watch live/i })).toBeDisabled()

    fireEvent.click(screen.getByText(/DEN vs LAL/i))
    expect(onSelect).toHaveBeenCalledWith('/replay/g1')
  })
})
