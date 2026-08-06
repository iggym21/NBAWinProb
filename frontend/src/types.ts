export interface GameSummary {
  game_id: string
  home_team: string
  away_team: string
}

export interface GamesResponse {
  replay_games: GameSummary[]
  live_available: boolean
}

export interface WinProbMessage {
  event_index: number
  period: number
  clock: string
  home_score: number
  away_score: number
  event_type: string
  description: string
  win_prob: number
}
