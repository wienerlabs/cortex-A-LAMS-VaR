import type { RegimeStreamMessage } from "./types";

export type RegimeHandler = (msg: RegimeStreamMessage) => void;
export type ErrorHandler = (err: Error) => void;

export interface RegimeStreamConfig {
  baseUrl: string;
  token: string;
  onRegime: RegimeHandler;
  onError?: ErrorHandler;
  onClose?: () => void;
  reconnect?: boolean;
  reconnectDelay?: number;
  maxReconnects?: number;
}

export class RegimeStreamClient {
  private ws: WebSocket | null = null;
  private reconnectCount = 0;
  private closed = false;
  private readonly url: string;
  private readonly reconnect: boolean;
  private readonly reconnectDelay: number;
  private readonly maxReconnects: number;

  constructor(private readonly config: RegimeStreamConfig) {
    const base = config.baseUrl.replace(/^http/, "ws").replace(/\/+$/, "");
    this.url = `${base}/api/v1/stream/regime?token=${encodeURIComponent(config.token)}`;
    this.reconnect = config.reconnect ?? true;
    this.reconnectDelay = config.reconnectDelay ?? 3000;
    this.maxReconnects = config.maxReconnects ?? 10;
  }

  connect(): void {
    this.closed = false;
    this.open();
  }

  private open(): void {
    this.ws = new WebSocket(this.url);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(String(event.data)) as RegimeStreamMessage;
        this.reconnectCount = 0;
        this.config.onRegime(data);
      } catch (err) {
        this.config.onError?.(err as Error);
      }
    };

    this.ws.onerror = (event) => {
      const err = new Error(`WebSocket error: ${(event as ErrorEvent).message ?? "unknown"}`);
      this.config.onError?.(err);
    };

    this.ws.onclose = () => {
      if (this.closed) {
        this.config.onClose?.();
        return;
      }
      if (this.reconnect && this.reconnectCount < this.maxReconnects) {
        this.reconnectCount++;
        setTimeout(() => this.open(), this.reconnectDelay);
      } else {
        this.config.onClose?.();
      }
    };
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
    this.ws = null;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

