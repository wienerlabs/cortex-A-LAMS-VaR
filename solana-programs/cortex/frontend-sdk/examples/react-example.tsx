import React, { useEffect, useState } from "react";
import { useConnection, useWallet } from "@solana/wallet-adapter-react";
import { CortexSDK } from "../index";

export function TokenBalance() {
  const { connection } = useConnection();
  const wallet = useWallet();
  const [balance, setBalance] = useState<number>(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!wallet.publicKey) return;

    const fetchBalance = async () => {
      setLoading(true);
      try {
        const sdk = CortexSDK.createWithConnection(connection, wallet);
        const balance = await sdk.token.getTokenBalance(wallet.publicKey!);
        setBalance(balance);
      } catch (error) {
        console.error("Error fetching balance:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchBalance();
    const interval = setInterval(fetchBalance, 10000);
    return () => clearInterval(interval);
  }, [wallet.publicKey, connection]);

  if (!wallet.connected) {
    return <div>Please connect your wallet</div>;
  }

  return (
    <div>
      <h3>CORTEX Balance</h3>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <p>{balance.toLocaleString()} CORTEX</p>
      )}
    </div>
  );
}

export function PrivateSaleInfo() {
  const { connection } = useConnection();
  const wallet = useWallet();
  const [saleInfo, setSaleInfo] = useState<any>(null);
  const [whitelistInfo, setWhitelistInfo] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!wallet.publicKey) return;

    const fetchInfo = async () => {
      setLoading(true);
      try {
        const sdk = CortexSDK.createWithConnection(connection, wallet);
        const sale = await sdk.privateSale.getSaleInfo();
        const whitelist = await sdk.privateSale.getWhitelistInfo(wallet.publicKey!);
        setSaleInfo(sale);
        setWhitelistInfo(whitelist);
      } catch (error) {
        console.error("Error fetching sale info:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchInfo();
  }, [wallet.publicKey, connection]);

  if (!wallet.connected) {
    return <div>Please connect your wallet</div>;
  }

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h3>Private Sale</h3>
      {saleInfo && (
        <div>
          <p>Price: {saleInfo.pricePerToken} SOL per token</p>
          <p>Tokens Sold: {saleInfo.tokensSold.toLocaleString()}</p>
          <p>Total for Sale: {saleInfo.totalTokensForSale.toLocaleString()}</p>
          <p>Status: {saleInfo.isActive ? "Active" : "Inactive"}</p>
        </div>
      )}
      {whitelistInfo && (
        <div>
          <h4>Your Allocation</h4>
          <p>Total: {whitelistInfo.allocation.toLocaleString()}</p>
          <p>Purchased: {whitelistInfo.purchased.toLocaleString()}</p>
          <p>Remaining: {(whitelistInfo.allocation - whitelistInfo.purchased).toLocaleString()}</p>
        </div>
      )}
    </div>
  );
}

export function VestingInfo() {
  const { connection } = useConnection();
  const wallet = useWallet();
  const [vestingInfo, setVestingInfo] = useState<any>(null);
  const [claimable, setClaimable] = useState<number>(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!wallet.publicKey) return;

    const fetchInfo = async () => {
      setLoading(true);
      try {
        const sdk = CortexSDK.createWithConnection(connection, wallet);
        const schedule = await sdk.vesting.getVestingSchedule(wallet.publicKey!);
        const claimableAmount = await sdk.vesting.getClaimableAmount(wallet.publicKey!);
        setVestingInfo(schedule);
        setClaimable(claimableAmount);
      } catch (error) {
        console.error("Error fetching vesting info:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchInfo();
    const interval = setInterval(fetchInfo, 30000);
    return () => clearInterval(interval);
  }, [wallet.publicKey, connection]);

  const handleClaim = async () => {
    if (!wallet.publicKey) return;
    
    setLoading(true);
    try {
      const sdk = CortexSDK.createWithConnection(connection, wallet);
      const tx = await sdk.vesting.claim();
      console.log("Claim successful:", tx);
      alert("Tokens claimed successfully!");
    } catch (error) {
      console.error("Error claiming tokens:", error);
      alert("Failed to claim tokens");
    } finally {
      setLoading(false);
    }
  };

  if (!wallet.connected) {
    return <div>Please connect your wallet</div>;
  }

  if (loading && !vestingInfo) {
    return <div>Loading...</div>;
  }

  if (!vestingInfo) {
    return <div>No vesting schedule found</div>;
  }

  return (
    <div>
      <h3>Vesting Schedule</h3>
      <p>Total Amount: {vestingInfo.totalAmount.toLocaleString()}</p>
      <p>Released: {vestingInfo.releasedAmount.toLocaleString()}</p>
      <p>Claimable: {claimable.toLocaleString()}</p>
      <button onClick={handleClaim} disabled={loading || claimable === 0}>
        {loading ? "Claiming..." : "Claim Tokens"}
      </button>
    </div>
  );
}

