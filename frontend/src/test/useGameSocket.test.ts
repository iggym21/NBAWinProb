// frontend/src/test/useGameSocket.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useGameSocket } from '../hooks/useGameSocket'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  onopen: (() => void) | null = null
  onmessage: ((ev: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  readyState = 0
  url: string
  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }
  close() {
    this.readyState = 3
    this.onclose?.()
  }
}

beforeEach(() => {
  MockWebSocket.instances = []
  // @ts-expect-error test override
  global.WebSocket = MockWebSocket
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useGameSocket', () => {
  it('starts disconnected with no messages when wsUrl is null', () => {
    const { result } = renderHook(() => useGameSocket(null))
    expect(result.current.connected).toBe(false)
    expect(result.current.messages).toEqual([])
  })

  it('appends parsed messages as they arrive', async () => {
    const { result } = renderHook(() => useGameSocket('ws://localhost:8000/replay/g1'))
    const socket = MockWebSocket.instances[0]
    socket.onopen?.()
    await waitFor(() => expect(result.current.connected).toBe(true))

    const message = {
      event_index: 0, period: 1, clock: 'PT12M00.00S', home_score: 0, away_score: 0,
      event_type: 'Jump Ball', description: 'Tip', win_prob: 0.5,
    }
    socket.onmessage?.({ data: JSON.stringify(message) })

    await waitFor(() => expect(result.current.messages).toHaveLength(1))
    expect(result.current.messages[0]).toEqual(message)
  })

  it('resets messages and connected state when the socket closes', async () => {
    const { result } = renderHook(() => useGameSocket('ws://localhost:8000/replay/g1'))
    const socket = MockWebSocket.instances[0]
    socket.onopen?.()
    await waitFor(() => expect(result.current.connected).toBe(true))
    socket.onclose?.()
    await waitFor(() => expect(result.current.connected).toBe(false))
  })
})
