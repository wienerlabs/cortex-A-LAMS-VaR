use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer, Mint};

declare_id!("5Rkn4B2CAcAiizUyHrxxBTRcAsZcRaLSMi8gdzXUW1nX");

pub const MAX_PERFORMANCE_FEE: u16 = 2000; // 20%
pub const FEE_DENOMINATOR: u16 = 10000;
pub const MAX_STRATEGIES: usize = 10;
pub const MAX_DEPOSIT: u64 = 1_000_000_000_000_000; // 1B tokens (with 6 decimals)
pub const MIN_SHARES: u64 = 1000; // Minimum shares to prevent dust attacks

#[program]
pub mod cortex_vault {
    use super::*;

    pub fn create_vault(
        ctx: Context<CreateVault>,
        name: String,
        performance_fee: u16,
    ) -> Result<()> {
        require!(name.len() <= 16, VaultError::NameTooLong);
        require!(performance_fee <= MAX_PERFORMANCE_FEE, VaultError::FeeTooHigh);

        let vault = &mut ctx.accounts.vault;
        vault.authority = ctx.accounts.authority.key();
        vault.guardian = ctx.accounts.authority.key();
        vault.agent = Pubkey::default();
        vault.asset_mint = ctx.accounts.asset_mint.key();
        vault.share_mint = Pubkey::default();
        vault.asset_vault = Pubkey::default();
        vault.treasury = ctx.accounts.treasury.key();
        vault.name = name;
        vault.total_assets = 0;
        vault.total_shares = 0;
        vault.performance_fee = performance_fee;
        vault.state = VaultState::Initializing;
        vault.strategy_count = 0;
        vault.bump = ctx.bumps.vault;
        Ok(())
    }

    pub fn init_vault_accounts(ctx: Context<InitVaultAccounts>) -> Result<()> {
        let vault = &mut ctx.accounts.vault;
        require!(vault.state == VaultState::Initializing, VaultError::VaultNotActive);

        vault.share_mint = ctx.accounts.share_mint.key();
        vault.asset_vault = ctx.accounts.asset_vault.key();
        vault.state = VaultState::Active;
        Ok(())
    }

    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        require!(amount > 0, VaultError::ZeroAmount);
        require!(amount <= MAX_DEPOSIT, VaultError::DepositTooLarge);

        let vault = &mut ctx.accounts.vault;
        require!(vault.state == VaultState::Active, VaultError::VaultNotActive);

        // Calculate shares with rounding protection
        let shares = if vault.total_shares == 0 {
            // First deposit: require minimum to prevent share price manipulation
            require!(amount >= MIN_SHARES, VaultError::DepositTooSmall);
            amount
        } else {
            // Prevent division by zero (shouldn't happen but defensive)
            require!(vault.total_assets > 0, VaultError::InvalidState);
            // Round down to protect existing depositors
            let shares = (amount as u128)
                .checked_mul(vault.total_shares as u128)
                .ok_or(VaultError::MathOverflow)?
                .checked_div(vault.total_assets as u128)
                .ok_or(VaultError::MathOverflow)? as u64;
            require!(shares > 0, VaultError::DepositTooSmall);
            shares
        };

        // CEI Pattern: Update state BEFORE external calls
        vault.total_assets = vault.total_assets.checked_add(amount).ok_or(VaultError::MathOverflow)?;
        vault.total_shares = vault.total_shares.checked_add(shares).ok_or(VaultError::MathOverflow)?;

        // Transfer tokens from user
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.user_asset_account.to_account_info(),
                    to: ctx.accounts.asset_vault.to_account_info(),
                    authority: ctx.accounts.user.to_account_info(),
                },
            ),
            amount,
        )?;

        // Mint shares to user
        let asset_mint = vault.asset_mint;
        let bump = vault.bump;
        let seeds = &[b"vault".as_ref(), asset_mint.as_ref(), &[bump]];
        let signer = &[&seeds[..]];
        token::mint_to(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                token::MintTo {
                    mint: ctx.accounts.share_mint.to_account_info(),
                    to: ctx.accounts.user_share_account.to_account_info(),
                    authority: vault.to_account_info(),
                },
                signer,
            ),
            shares,
        )?;

        emit!(DepositEvent { user: ctx.accounts.user.key(), assets: amount, shares });
        Ok(())
    }

    pub fn withdraw(ctx: Context<Withdraw>, shares: u64) -> Result<()> {
        require!(shares > 0, VaultError::ZeroAmount);

        let vault = &mut ctx.accounts.vault;

        // Allow withdrawals in Active, Paused, or Emergency states
        // Only Initializing state blocks withdrawals
        require!(vault.state != VaultState::Initializing, VaultError::VaultNotActive);

        // Validate shares don't exceed total
        require!(shares <= vault.total_shares, VaultError::InsufficientShares);
        require!(vault.total_shares > 0, VaultError::InvalidState);

        // Calculate assets with checked math - round down to protect vault
        let assets = (shares as u128)
            .checked_mul(vault.total_assets as u128)
            .ok_or(VaultError::MathOverflow)?
            .checked_div(vault.total_shares as u128)
            .ok_or(VaultError::MathOverflow)? as u64;

        require!(assets > 0, VaultError::WithdrawTooSmall);
        require!(assets <= ctx.accounts.asset_vault.amount, VaultError::InsufficientFunds);

        // CEI Pattern: Update state BEFORE external calls
        vault.total_assets = vault.total_assets.checked_sub(assets).ok_or(VaultError::MathOverflow)?;
        vault.total_shares = vault.total_shares.checked_sub(shares).ok_or(VaultError::MathOverflow)?;

        // Burn shares first (user gives up shares)
        token::burn(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                token::Burn {
                    mint: ctx.accounts.share_mint.to_account_info(),
                    from: ctx.accounts.user_share_account.to_account_info(),
                    authority: ctx.accounts.user.to_account_info(),
                },
            ),
            shares,
        )?;

        // Transfer assets to user
        let asset_mint = vault.asset_mint;
        let bump = vault.bump;
        let seeds = &[b"vault".as_ref(), asset_mint.as_ref(), &[bump]];
        let signer = &[&seeds[..]];
        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.asset_vault.to_account_info(),
                    to: ctx.accounts.user_asset_account.to_account_info(),
                    authority: vault.to_account_info(),
                },
                signer,
            ),
            assets,
        )?;

        emit!(WithdrawEvent { user: ctx.accounts.user.key(), assets, shares });
        Ok(())
    }

    pub fn set_agent(ctx: Context<SetAgent>, agent: Pubkey) -> Result<()> {
        ctx.accounts.vault.agent = agent;
        emit!(AgentSet { vault: ctx.accounts.vault.key(), agent });
        Ok(())
    }

    pub fn pause(ctx: Context<Pause>) -> Result<()> {
        let vault = &mut ctx.accounts.vault;
        vault.state = VaultState::Paused;
        emit!(VaultStateChanged { vault: vault.key(), state: VaultState::Paused });
        Ok(())
    }

    pub fn unpause(ctx: Context<Unpause>) -> Result<()> {
        let vault = &mut ctx.accounts.vault;
        vault.state = VaultState::Active;
        emit!(VaultStateChanged { vault: vault.key(), state: VaultState::Active });
        Ok(())
    }

    pub fn emergency(ctx: Context<Emergency>) -> Result<()> {
        let vault = &mut ctx.accounts.vault;
        vault.state = VaultState::Emergency;
        emit!(VaultStateChanged { vault: vault.key(), state: VaultState::Emergency });
        Ok(())
    }
}


#[derive(Accounts)]
pub struct CreateVault<'info> {
    #[account(
        init, payer = authority, space = 8 + Vault::INIT_SPACE,
        seeds = [b"vault", asset_mint.key().as_ref()], bump
    )]
    pub vault: Box<Account<'info, Vault>>,
    pub asset_mint: Account<'info, Mint>,
    /// CHECK: Treasury receives fees
    pub treasury: UncheckedAccount<'info>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitVaultAccounts<'info> {
    #[account(mut, seeds = [b"vault", vault.asset_mint.as_ref()], bump = vault.bump, has_one = authority)]
    pub vault: Box<Account<'info, Vault>>,
    pub asset_mint: Account<'info, Mint>,
    #[account(
        init, payer = authority,
        mint::decimals = asset_mint.decimals,
        mint::authority = vault,
        seeds = [b"share_mint", vault.key().as_ref()], bump
    )]
    pub share_mint: Box<Account<'info, Mint>>,
    #[account(
        init, payer = authority,
        token::mint = asset_mint, token::authority = vault,
        seeds = [b"asset_vault", vault.key().as_ref()], bump
    )]
    pub asset_vault: Box<Account<'info, TokenAccount>>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct Deposit<'info> {
    #[account(mut, seeds = [b"vault", vault.asset_mint.as_ref()], bump = vault.bump)]
    pub vault: Box<Account<'info, Vault>>,
    #[account(mut, address = vault.share_mint)]
    pub share_mint: Box<Account<'info, Mint>>,
    #[account(mut, address = vault.asset_vault)]
    pub asset_vault: Box<Account<'info, TokenAccount>>,
    #[account(mut, token::mint = vault.asset_mint, token::authority = user)]
    pub user_asset_account: Box<Account<'info, TokenAccount>>,
    #[account(mut, token::mint = vault.share_mint, token::authority = user)]
    pub user_share_account: Box<Account<'info, TokenAccount>>,
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct Withdraw<'info> {
    #[account(mut, seeds = [b"vault", vault.asset_mint.as_ref()], bump = vault.bump)]
    pub vault: Box<Account<'info, Vault>>,
    #[account(mut, address = vault.share_mint)]
    pub share_mint: Box<Account<'info, Mint>>,
    #[account(mut, address = vault.asset_vault)]
    pub asset_vault: Box<Account<'info, TokenAccount>>,
    #[account(mut, token::mint = vault.asset_mint, token::authority = user)]
    pub user_asset_account: Box<Account<'info, TokenAccount>>,
    #[account(mut, token::mint = vault.share_mint, token::authority = user)]
    pub user_share_account: Box<Account<'info, TokenAccount>>,
    pub user: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct SetAgent<'info> {
    #[account(mut, seeds = [b"vault", vault.asset_mint.as_ref()], bump = vault.bump, has_one = authority)]
    pub vault: Account<'info, Vault>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct Pause<'info> {
    #[account(mut, seeds = [b"vault", vault.asset_mint.as_ref()], bump = vault.bump,
        constraint = vault.guardian == authority.key() || vault.authority == authority.key())]
    pub vault: Account<'info, Vault>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct Unpause<'info> {
    #[account(mut, seeds = [b"vault", vault.asset_mint.as_ref()], bump = vault.bump, has_one = authority)]
    pub vault: Account<'info, Vault>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct Emergency<'info> {
    #[account(mut, seeds = [b"vault", vault.asset_mint.as_ref()], bump = vault.bump,
        constraint = vault.guardian == authority.key() || vault.authority == authority.key())]
    pub vault: Account<'info, Vault>,
    pub authority: Signer<'info>,
}

#[account]
#[derive(InitSpace)]
pub struct Vault {
    pub authority: Pubkey,
    pub guardian: Pubkey,
    pub agent: Pubkey,
    pub asset_mint: Pubkey,
    pub share_mint: Pubkey,
    pub asset_vault: Pubkey,
    pub treasury: Pubkey,
    #[max_len(16)]
    pub name: String,
    pub total_assets: u64,
    pub total_shares: u64,
    pub performance_fee: u16,
    pub state: VaultState,
    pub strategy_count: u8,
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, InitSpace)]
pub enum VaultState { Initializing, Active, Paused, Emergency }

#[event]
pub struct DepositEvent { pub user: Pubkey, pub assets: u64, pub shares: u64 }
#[event]
pub struct WithdrawEvent { pub user: Pubkey, pub assets: u64, pub shares: u64 }
#[event]
pub struct AgentSet { pub vault: Pubkey, pub agent: Pubkey }
#[event]
pub struct VaultStateChanged { pub vault: Pubkey, pub state: VaultState }

#[error_code]
pub enum VaultError {
    #[msg("Name too long")] NameTooLong,
    #[msg("Fee too high")] FeeTooHigh,
    #[msg("Zero amount")] ZeroAmount,
    #[msg("Vault not active")] VaultNotActive,
    #[msg("Unauthorized")] Unauthorized,
    #[msg("Deposit too large")] DepositTooLarge,
    #[msg("Deposit too small")] DepositTooSmall,
    #[msg("Withdraw too small")] WithdrawTooSmall,
    #[msg("Math overflow")] MathOverflow,
    #[msg("Invalid vault state")] InvalidState,
    #[msg("Insufficient shares")] InsufficientShares,
    #[msg("Insufficient funds in vault")] InsufficientFunds,
}
