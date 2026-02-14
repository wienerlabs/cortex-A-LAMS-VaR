use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer, Mint};

declare_id!("G2uYZTetdVmo2Q4upMSYoF2ya8GpaKKtRakEWYv7PgHn");

pub const JUPITER_PROGRAM_ID: &str = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4";
pub const BUYBACK_COOLDOWN: i64 = 24 * 60 * 60;  // 24 hours
pub const MIN_BUYBACK_AMOUNT: u64 = 1_000_000;   // Min 1 USDC (6 decimals)

#[program]
pub mod cortex_treasury {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let treasury = &mut ctx.accounts.treasury;
        treasury.authority = ctx.accounts.authority.key();
        treasury.guardian = ctx.accounts.authority.key();
        treasury.total_fees_collected = 0;
        treasury.total_distributed = 0;
        treasury.total_buyback = 0;
        treasury.last_buyback = 0;
        treasury.bump = ctx.bumps.treasury;
        Ok(())
    }

    pub fn collect_fee(ctx: Context<CollectFee>, amount: u64) -> Result<()> {
        require!(amount > 0, TreasuryError::ZeroAmount);

        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.from_account.to_account_info(),
                    to: ctx.accounts.treasury_vault.to_account_info(),
                    authority: ctx.accounts.from_authority.to_account_info(),
                },
            ),
            amount,
        )?;

        ctx.accounts.treasury.total_fees_collected += amount;
        emit!(FeeCollected { mint: ctx.accounts.mint.key(), amount });
        Ok(())
    }

    pub fn distribute(ctx: Context<Distribute>, amount: u64) -> Result<()> {
        require!(amount > 0, TreasuryError::ZeroAmount);
        require!(ctx.accounts.treasury_vault.amount >= amount, TreasuryError::InsufficientFunds);

        let seeds = &[b"treasury".as_ref(), &[ctx.accounts.treasury.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.treasury_vault.to_account_info(),
                    to: ctx.accounts.to_account.to_account_info(),
                    authority: ctx.accounts.treasury.to_account_info(),
                },
                signer,
            ),
            amount,
        )?;

        ctx.accounts.treasury.total_distributed += amount;
        emit!(FundsDistributed { to: ctx.accounts.to_account.key(), amount });
        Ok(())
    }

    pub fn set_guardian(ctx: Context<SetGuardian>, new_guardian: Pubkey) -> Result<()> {
        ctx.accounts.treasury.guardian = new_guardian;
        emit!(GuardianChanged { new_guardian });
        Ok(())
    }

    pub fn emergency_withdraw(ctx: Context<EmergencyWithdraw>) -> Result<()> {
        let amount = ctx.accounts.treasury_vault.amount;
        require!(amount > 0, TreasuryError::ZeroAmount);

        let seeds = &[b"treasury".as_ref(), &[ctx.accounts.treasury.bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.treasury_vault.to_account_info(),
                    to: ctx.accounts.to_account.to_account_info(),
                    authority: ctx.accounts.treasury.to_account_info(),
                },
                signer,
            ),
            amount,
        )?;

        emit!(EmergencyWithdrawal { amount });
        Ok(())
    }

    /// Execute buyback via Jupiter swap
    /// Route data is computed off-chain via Jupiter API
    pub fn execute_buyback(ctx: Context<ExecuteBuyback>, route_data: Vec<u8>, min_output: u64) -> Result<()> {
        let treasury = &mut ctx.accounts.treasury;
        let clock = Clock::get()?;

        // Validate route_data size to prevent excessive compute
        require!(route_data.len() <= 1024, TreasuryError::InvalidRouteData);
        require!(min_output > 0, TreasuryError::ZeroAmount);

        // Cooldown check with overflow protection
        let time_since_buyback = clock.unix_timestamp.saturating_sub(treasury.last_buyback);
        require!(time_since_buyback >= BUYBACK_COOLDOWN, TreasuryError::BuybackCooldown);

        let input_amount = ctx.accounts.input_vault.amount;
        require!(input_amount >= MIN_BUYBACK_AMOUNT, TreasuryError::AmountTooSmall);

        let output_before = ctx.accounts.output_vault.amount;

        // CRITICAL: Validate Jupiter program ID
        require!(!ctx.remaining_accounts.is_empty(), TreasuryError::InvalidRouteData);
        let jupiter_program = &ctx.remaining_accounts[0];
        let expected_jupiter = JUPITER_PROGRAM_ID.parse::<Pubkey>().map_err(|_| TreasuryError::InvalidRouteData)?;
        require!(jupiter_program.key() == expected_jupiter, TreasuryError::InvalidJupiterProgram);

        let seeds = &[b"treasury".as_ref(), &[treasury.bump]];
        let signer_seeds = &[&seeds[..]];

        // Build account metas for Jupiter CPI
        let accounts_meta: Vec<AccountMeta> = ctx.remaining_accounts[1..]
            .iter()
            .map(|acc| {
                if acc.is_writable {
                    AccountMeta::new(*acc.key, acc.is_signer)
                } else {
                    AccountMeta::new_readonly(*acc.key, acc.is_signer)
                }
            })
            .collect();

        let ix = anchor_lang::solana_program::instruction::Instruction {
            program_id: *jupiter_program.key,
            accounts: accounts_meta,
            data: route_data,
        };

        anchor_lang::solana_program::program::invoke_signed(
            &ix,
            ctx.remaining_accounts,
            signer_seeds,
        )?;

        // Verify slippage with reload
        ctx.accounts.output_vault.reload()?;
        let output_received = ctx.accounts.output_vault.amount
            .checked_sub(output_before)
            .ok_or(TreasuryError::MathOverflow)?;
        require!(output_received >= min_output, TreasuryError::SlippageExceeded);

        // Update state
        treasury.last_buyback = clock.unix_timestamp;
        treasury.total_buyback = treasury.total_buyback
            .checked_add(output_received)
            .ok_or(TreasuryError::MathOverflow)?;

        emit!(BuybackExecuted {
            input_mint: ctx.accounts.input_mint.key(),
            output_mint: ctx.accounts.output_mint.key(),
            input_amount,
            output_amount: output_received
        });
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = authority, space = 8 + Treasury::INIT_SPACE, seeds = [b"treasury"], bump)]
    pub treasury: Account<'info, Treasury>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CollectFee<'info> {
    #[account(mut, seeds = [b"treasury"], bump = treasury.bump)]
    pub treasury: Account<'info, Treasury>,
    pub mint: Account<'info, Mint>,
    #[account(
        init_if_needed, payer = from_authority,
        token::mint = mint, token::authority = treasury,
        seeds = [b"treasury_vault", mint.key().as_ref()], bump
    )]
    pub treasury_vault: Account<'info, TokenAccount>,
    #[account(mut, token::mint = mint)]
    pub from_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub from_authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct Distribute<'info> {
    #[account(mut, seeds = [b"treasury"], bump = treasury.bump, has_one = authority)]
    pub treasury: Account<'info, Treasury>,
    pub mint: Account<'info, Mint>,
    #[account(mut, seeds = [b"treasury_vault", mint.key().as_ref()], bump)]
    pub treasury_vault: Account<'info, TokenAccount>,
    #[account(mut, token::mint = mint)]
    pub to_account: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct SetGuardian<'info> {
    #[account(mut, seeds = [b"treasury"], bump = treasury.bump, has_one = authority)]
    pub treasury: Account<'info, Treasury>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct EmergencyWithdraw<'info> {
    #[account(mut, seeds = [b"treasury"], bump = treasury.bump,
        constraint = treasury.guardian == authority.key() || treasury.authority == authority.key())]
    pub treasury: Account<'info, Treasury>,
    pub mint: Account<'info, Mint>,
    #[account(mut, seeds = [b"treasury_vault", mint.key().as_ref()], bump)]
    pub treasury_vault: Account<'info, TokenAccount>,
    #[account(mut, token::mint = mint)]
    pub to_account: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct ExecuteBuyback<'info> {
    #[account(mut, seeds = [b"treasury"], bump = treasury.bump, has_one = authority)]
    pub treasury: Account<'info, Treasury>,

    pub input_mint: Account<'info, Mint>,   // e.g., USDC
    pub output_mint: Account<'info, Mint>,  // e.g., CORTEX

    #[account(mut, seeds = [b"treasury_vault", input_mint.key().as_ref()], bump)]
    pub input_vault: Account<'info, TokenAccount>,

    #[account(mut, seeds = [b"treasury_vault", output_mint.key().as_ref()], bump)]
    pub output_vault: Account<'info, TokenAccount>,

    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    // remaining_accounts: Jupiter program + all accounts needed for the swap
}

#[account]
#[derive(InitSpace)]
pub struct Treasury {
    pub authority: Pubkey,
    pub guardian: Pubkey,
    pub total_fees_collected: u64,
    pub total_distributed: u64,
    pub total_buyback: u64,
    pub last_buyback: i64,
    pub bump: u8,
}

#[event]
pub struct FeeCollected { pub mint: Pubkey, pub amount: u64 }
#[event]
pub struct FundsDistributed { pub to: Pubkey, pub amount: u64 }
#[event]
pub struct GuardianChanged { pub new_guardian: Pubkey }
#[event]
pub struct EmergencyWithdrawal { pub amount: u64 }
#[event]
pub struct BuybackExecuted {
    pub input_mint: Pubkey,
    pub output_mint: Pubkey,
    pub input_amount: u64,
    pub output_amount: u64
}

#[error_code]
pub enum TreasuryError {
    #[msg("Zero amount")] ZeroAmount,
    #[msg("Insufficient funds")] InsufficientFunds,
    #[msg("Unauthorized")] Unauthorized,
    #[msg("Buyback cooldown active")] BuybackCooldown,
    #[msg("Amount too small for buyback")] AmountTooSmall,
    #[msg("Slippage exceeded")] SlippageExceeded,
    #[msg("Invalid route data")] InvalidRouteData,
    #[msg("Invalid Jupiter program")] InvalidJupiterProgram,
    #[msg("Math overflow")] MathOverflow,
}
