import { useEffect, useRef, useState } from 'react'
import type { WinProbMessage } from '../types'

export interface UseGameSocketResult {
  messages: WinProbMessage[]
  connected: boolean
  error: string | null
}

export function useGameSocket(wsUrl: string | null): UseGameSocketResult {
  const [messages, setMessages] = useState<WinProbMessage[]>([])
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    setMessages([])
    setConnected(false)
    setError(null)

    if (!wsUrl) {
      return
    }

    const socket = new WebSocket(wsUrl)
    socketRef.current = socket

    socket.onopen = () => setConnected(true)
    socket.onmessage = (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data) as WinProbMessage
        setMessages((prev) => [...prev, parsed])
      } catch {
        setError('Failed to parse server message')
      }
    }
    socket.onerror = () => setError('WebSocket error')
    socket.onclose = () => setConnected(false)

    return () => {
      socket.close()
    }
  }, [wsUrl])

  return { messages, connected, error }
}
