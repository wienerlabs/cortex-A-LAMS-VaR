/**
 * PM (Portfolio Manager) Approval Workflow Integration Tests
 *
 * Tests the complete pipeline:
 * Strategy Analysis → PM Approval → Guardian Validation → Risk Manager → Execution
 *
 * Run with: npx vitest run src/services/pm/__tests__/pmIntegration.test.ts
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  pmDecisionEngine,
  approvalQueue,
  pmLogger,
  ApprovalStatusEnum as ApprovalStatus,
  type QueueTradeParams,
  type ApprovalResult,
} from '../index.js';

// ============= TEST HELPERS =============

function createTradeParams(overrides: Partial<QueueTradeParams> = {}): QueueTradeParams {
  return {
    strategy: 'spot',
    action: 'BUY',
    asset: 'SOL',
    assetMint: 'So11111111111111111111111111111111111111112',
    amount: 100,
    amountUsd: 100,
    confidence: 0.75,
    risk: {
      volatility: 20,
      liquidityScore: 70,
      riskScore: 30,
    },
    reasoning: 'Test trade',
    protocol: 'jupiter',
    ...overrides,
  };
}

// ============= TEST SUITE =============

describe('PM Approval Workflow Integration', () => {
  beforeEach(() => {
    // Clear the approval queue before each test
    approvalQueue.clearAll();
  });

  afterEach(() => {
    approvalQueue.clearAll();
  });

  // ============= SCENARIO 1: Small Trade (Auto-Approve) =============
  describe('Scenario 1: Small Trade (Auto-Approve)', () => {
    it('should auto-approve trades below $1000 threshold', () => {
      const params = createTradeParams({
        strategy: 'spot',
        asset: 'SOL',
        amountUsd: 50, // Below $1000 threshold
      });

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      // $50 is below $1000 threshold AND below 10% of portfolio
      expect(needsApproval).toBe(false);
    });

    it('should not queue trades that do not need approval', () => {
      const params = createTradeParams({
        amountUsd: 50,
      });

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      if (!needsApproval) {
        // Trade proceeds directly without queuing
        const pendingTrades = approvalQueue.getPendingTrades();
        expect(pendingTrades.length).toBe(0);
      }
    });
  });

  // ============= SCENARIO 2: Large Trade (Needs Approval) =============
  describe('Scenario 2: Large Trade (Needs Approval)', () => {
    it('should require approval for trades >= $1000', () => {
      const params = createTradeParams({
        strategy: 'spot',
        asset: 'SOL',
        amountUsd: 1500, // Above $1000 threshold
      });

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      expect(needsApproval).toBe(true);
    });

    it('should queue trade and create trade ID', () => {
      const params = createTradeParams({
        amountUsd: 1500,
      });

      const tradeId = approvalQueue.queueTrade(params);

      expect(tradeId).toBeDefined();
      expect(typeof tradeId).toBe('string');
      expect(tradeId.length).toBeGreaterThan(0);

      const pendingTrades = approvalQueue.getPendingTrades();
      expect(pendingTrades.length).toBe(1);
      expect(pendingTrades[0].id).toBe(tradeId);
      expect(pendingTrades[0].status).toBe(ApprovalStatus.PENDING);
    });

    it('should show PENDING status in queue', () => {
      const params = createTradeParams({
        amountUsd: 1500,
      });

      const tradeId = approvalQueue.queueTrade(params);
      const trade = approvalQueue.getTrade(tradeId);

      expect(trade).toBeDefined();
      expect(trade?.status).toBe(ApprovalStatus.PENDING);
      expect(trade?.amountUsd).toBe(1500);
    });
  });

  // ============= SCENARIO 3: Perps Trade (Always Needs Approval) =============
  describe('Scenario 3: Perps Trade (Always Needs Approval)', () => {
    it('should require approval for perps even if amount < $1000', () => {
      const params = createTradeParams({
        strategy: 'perps',
        action: 'OPEN',
        asset: 'SOL-PERP',
        amountUsd: 500, // Below $1000 but perps always needs approval
        leverage: 5,
      });

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      // Perps is in "always_approve" list
      expect(needsApproval).toBe(true);
    });

    it('should queue perps trade with correct metadata', () => {
      const params = createTradeParams({
        strategy: 'perps',
        action: 'OPEN',
        asset: 'SOL-PERP',
        amountUsd: 500,
        leverage: 5,
        protocol: 'drift',
      });

      const tradeId = approvalQueue.queueTrade(params);
      const trade = approvalQueue.getTrade(tradeId);

      expect(trade?.strategy).toBe('perps');
      expect(trade?.action).toBe('OPEN');
      expect(trade?.leverage).toBe(5);
      expect(trade?.protocol).toBe('drift');
    });
  });

  // ============= SCENARIO 4: Lending Trade (Never Needs Approval) =============
  describe('Scenario 4: Lending Trade (Never Needs Approval)', () => {
    it('should auto-approve lending trades even if amount >= $1000', () => {
      const params = createTradeParams({
        strategy: 'lending',
        action: 'DEPOSIT',
        asset: 'USDC',
        amountUsd: 1500,
        protocol: 'marginfi',
      });

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      // Lending is in "never_approve" list
      expect(needsApproval).toBe(false);
    });

    it('should auto-approve lending borrow even with high amount', () => {
      const params = createTradeParams({
        strategy: 'lending',
        action: 'BORROW',
        asset: 'SOL',
        amountUsd: 2000, // 20% of portfolio
        protocol: 'kamino',
      });

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      expect(needsApproval).toBe(false);
    });
  });

  // ============= SCENARIO 5: High Risk Trade (Multiple Triggers) =============
  describe('Scenario 5: High Risk Trade (Multiple Triggers)', () => {
    it('should require approval when percentage threshold exceeded', () => {
      const params = createTradeParams({
        strategy: 'spot',
        asset: 'SOL',
        amountUsd: 2000, // 20% of $10K portfolio (> 10% threshold)
      });

      const portfolioValueUsd = 10000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      expect(needsApproval).toBe(true);
    });

    it('should require approval when volatility exceeds threshold', () => {
      const params = createTradeParams({
        strategy: 'spot',
        amountUsd: 500,
        risk: { volatility: 60, liquidityScore: 70, riskScore: 30 },
      });

      const portfolioValueUsd = 100000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      expect(needsApproval).toBe(true);
    });

    it('should require approval when confidence below threshold', () => {
      const params = createTradeParams({
        strategy: 'spot',
        amountUsd: 500,
        confidence: 0.55, // Below 70% threshold
      });

      const portfolioValueUsd = 100000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      expect(needsApproval).toBe(true);
    });

    it('should require approval when risk score exceeds threshold', () => {
      const params = createTradeParams({
        strategy: 'spot',
        amountUsd: 500,
        risk: { volatility: 20, liquidityScore: 70, riskScore: 80 },
      });

      const portfolioValueUsd = 100000;
      const needsApproval = pmDecisionEngine.needsApproval(params, portfolioValueUsd);

      expect(needsApproval).toBe(true);
    });
  });

  // ============= SCENARIO 6: Approval/Rejection Workflow =============
  describe('Scenario 6: Approval/Rejection Workflow', () => {
    it('should approve trade and update status', () => {
      const params = createTradeParams({ amountUsd: 1500 });
      const tradeId = approvalQueue.queueTrade(params);

      const result = approvalQueue.approveTrade(tradeId, 'test-pm');

      expect(result.approved).toBe(true);
      expect(result.status).toBe(ApprovalStatus.APPROVED);
      expect(result.approver).toBe('test-pm');

      const trade = approvalQueue.getTrade(tradeId);
      expect(trade?.status).toBe(ApprovalStatus.APPROVED);
    });

    it('should reject trade with reason', () => {
      const params = createTradeParams({ amountUsd: 1500 });
      const tradeId = approvalQueue.queueTrade(params);

      const result = approvalQueue.rejectTrade(tradeId, 'Too risky', 'test-pm');

      expect(result.approved).toBe(false);
      expect(result.status).toBe(ApprovalStatus.REJECTED);
      expect(result.rejectionReason).toBe('Too risky');

      const trade = approvalQueue.getTrade(tradeId);
      expect(trade?.status).toBe(ApprovalStatus.REJECTED);
    });

    it('should auto-approve trade with reason', () => {
      const params = createTradeParams({ amountUsd: 1500 });
      const tradeId = approvalQueue.queueTrade(params);

      const result = approvalQueue.autoApprove(tradeId, 'Below threshold');

      expect(result.approved).toBe(true);
      expect(result.status).toBe(ApprovalStatus.AUTO_APPROVED);
    });

    it('should fail to approve non-existent trade', () => {
      const result = approvalQueue.approveTrade('non-existent-id', 'test-pm');

      expect(result.approved).toBe(false);
      expect(result.status).toBe(ApprovalStatus.REJECTED);
    });

    it('should list all pending trades', () => {
      approvalQueue.queueTrade(createTradeParams({ amountUsd: 1500, asset: 'SOL' }));
      approvalQueue.queueTrade(createTradeParams({ amountUsd: 2000, asset: 'ETH' }));
      approvalQueue.queueTrade(createTradeParams({ amountUsd: 3000, asset: 'BTC' }));

      const pending = approvalQueue.getPendingTrades();

      expect(pending.length).toBe(3);
      expect(pending.every(t => t.status === ApprovalStatus.PENDING)).toBe(true);
    });

    it('should clear all trades from queue', () => {
      approvalQueue.queueTrade(createTradeParams({ amountUsd: 1500 }));
      approvalQueue.queueTrade(createTradeParams({ amountUsd: 2000 }));

      approvalQueue.clearAll();

      const pending = approvalQueue.getPendingTrades();
      expect(pending.length).toBe(0);
    });
  });

  // ============= PM Decision Engine State =============
  describe('PM Decision Engine State', () => {
    it('should report enabled status', () => {
      const enabled = pmDecisionEngine.isEnabled();
      expect(typeof enabled).toBe('boolean');
    });

    it('should have loaded config', () => {
      // If PM is enabled, config should be loaded
      if (pmDecisionEngine.isEnabled()) {
        const params = createTradeParams({ amountUsd: 50 });
        const needsApproval = pmDecisionEngine.needsApproval(params, 10000);
        // Small trade should not need approval
        expect(needsApproval).toBe(false);
      }
    });
  });
});
