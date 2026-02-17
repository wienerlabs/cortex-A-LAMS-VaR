import { Request, Response, NextFunction } from "express";
import { PublicKey } from "@solana/web3.js";
import crypto from "node:crypto";
import bs58 from "bs58";

const REPLAY_WINDOW_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Solana signature-verification middleware for mutating endpoints.
 *
 * Protocol:
 *   1. Caller builds message = `${method}:${path}:${timestamp}:${bodyJSON}`
 *   2. Signs with their Solana keypair (Ed25519)
 *   3. Sends headers:
 *      - x-solana-pubkey  (base58 public key)
 *      - x-solana-signature (base58 signature)
 *      - x-timestamp (unix ms string)
 *
 * The middleware verifies the signature matches AUTHORIZED_PUBLIC_KEY.
 * If AUTHORIZED_PUBLIC_KEY is not set, the middleware is a no-op (dev mode).
 */
export function solanaAuth(req: Request, res: Response, next: NextFunction): void {
  // Skip auth for read-only methods (GET, HEAD, OPTIONS)
  if (req.method === "GET" || req.method === "HEAD" || req.method === "OPTIONS") {
    next();
    return;
  }

  const authorizedKey = process.env.AUTHORIZED_PUBLIC_KEY;

  // Dev mode: no auth required when key not configured
  if (!authorizedKey) {
    next();
    return;
  }

  const pubkeyHeader = req.headers["x-solana-pubkey"] as string | undefined;
  const signatureHeader = req.headers["x-solana-signature"] as string | undefined;
  const timestampHeader = req.headers["x-timestamp"] as string | undefined;

  if (!pubkeyHeader || !signatureHeader || !timestampHeader) {
    res.status(401).json({
      success: false,
      error: "Missing authentication headers: x-solana-pubkey, x-solana-signature, x-timestamp",
    });
    return;
  }

  // Replay protection
  const timestamp = parseInt(timestampHeader, 10);
  if (isNaN(timestamp) || Math.abs(Date.now() - timestamp) > REPLAY_WINDOW_MS) {
    res.status(401).json({ success: false, error: "Request timestamp expired or invalid" });
    return;
  }

  // Verify caller is the authorized key
  if (pubkeyHeader !== authorizedKey) {
    res.status(403).json({ success: false, error: "Unauthorized public key" });
    return;
  }

  // Verify Ed25519 signature using Node crypto
  try {
    const pubkey = new PublicKey(pubkeyHeader);
    const bodyStr = JSON.stringify(req.body ?? {});
    const message = `${req.method}:${req.path}:${timestampHeader}:${bodyStr}`;
    const messageBytes = Buffer.from(message, "utf-8");
    const signatureBytes = Buffer.from(bs58.decode(signatureHeader));

    const ed25519Key = crypto.createPublicKey({
      key: Buffer.concat([
        // Ed25519 DER prefix for a 32-byte public key
        Buffer.from("302a300506032b6570032100", "hex"),
        pubkey.toBuffer(),
      ]),
      format: "der",
      type: "spki",
    });

    const valid = crypto.verify(null, messageBytes, ed25519Key, signatureBytes);

    if (!valid) {
      res.status(403).json({ success: false, error: "Invalid signature" });
      return;
    }
  } catch {
    res.status(403).json({ success: false, error: "Signature verification failed" });
    return;
  }

  next();
}
