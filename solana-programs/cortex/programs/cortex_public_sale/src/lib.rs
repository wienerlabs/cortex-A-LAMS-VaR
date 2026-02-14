use anchor_lang::prelude::*;
use anchor_lang::system_program;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("7TEbrJ6YRedPxxf9Syza94UdsJ6nUDZwoKAxEsijmjW8");

// Tokenomics: 22M tokens for public sale
const PUBLIC_SALE_ALLOCATION: u64 = 22_000_000_000_000_000; // 22M with 9 decimals
const MIN_PURCHASE: u64 = 50_000_000_000; // 50 CRTX minimum
const MAX_PURCHASE_PER_WALLET: u64 = 500_000_000_000_000; // 500K CRTX max per wallet

// Bonding curve: price = base_price + (slope * tokens_sold)
// Base price: 0.01 SOL per 1000 tokens = 10_000 lamports per 1000 tokens
// At 22M sold: ~0.015 SOL per 1000 tokens (50% increase)
const BASE_PRICE_LAMPORTS: u64 = 10_000; // per 1000 tokens (with 9 decimals = 1_000_000_000_000)
const PRICE_UNIT: u64 = 1_000_000_000_000; // 1000 tokens
const SLOPE_NUMERATOR: u64 = 5_000; // price increases 0.5 lamports per PRICE_UNIT sold
const SLOPE_DENOMINATOR: u64 = 1_000_000_000_000_000; // normalized
const SELL_FEE_BPS: u64 = 100; // 1% sell fee

#[program]
pub mod cortex_public_sale {
    use super::*;

    pub fn initialize(
        ctx: Context<Initialize>,
        start_time: i64,
        end_time: i64,
    ) -> Result<()> {
        let sale = &mut ctx.accounts.sale;
        sale.authority = ctx.accounts.authority.key();
        sale.token_mint = ctx.accounts.token_mint.key();
        sale.sale_vault = ctx.accounts.sale_vault.key();
        sale.treasury = ctx.accounts.treasury.key();
        sale.total_allocation = PUBLIC_SALE_ALLOCATION;
        sale.tokens_sold = 0;
        sale.sol_raised = 0;
        sale.start_time = start_time;
        sale.end_time = end_time;
        sale.is_active = true;
        sale.bump = *ctx.bumps.get("sale").unwrap();
        Ok(())
    }

    pub fn register_user(ctx: Context<RegisterUser>) -> Result<()> {
        let user_account = &mut ctx.accounts.user_account;
        user_account.total_purchased = 0;
        user_account.total_sol_spent = 0;
        user_account.bump = *ctx.bumps.get("user_account").unwrap();
        Ok(())
    }

    pub fn buy(ctx: Context<Buy>, token_amount: u64) -> Result<()> {
        let clock = Clock::get()?;
        let sale = &mut ctx.accounts.sale;
        let user_account = &mut ctx.accounts.user_account;

        require!(sale.is_active, PublicSaleError::SaleNotActive);
        require!(clock.unix_timestamp >= sale.start_time, PublicSaleError::SaleNotStarted);
        require!(clock.unix_timestamp <= sale.end_time, PublicSaleError::SaleEnded);
        require!(token_amount >= MIN_PURCHASE, PublicSaleError::BelowMinimum);
        require!(
            user_account.total_purchased + token_amount <= MAX_PURCHASE_PER_WALLET,
            PublicSaleError::ExceedsMaxPerWallet
        );
        require!(
            sale.tokens_sold + token_amount <= sale.total_allocation,
            PublicSaleError::ExceedsTotalAllocation
        );

        let sol_cost = calculate_buy_cost(sale.tokens_sold, token_amount)?;

        system_program::transfer(
            CpiContext::new(
                ctx.accounts.system_program.to_account_info(),
                system_program::Transfer {
                    from: ctx.accounts.buyer.to_account_info(),
                    to: ctx.accounts.treasury.to_account_info(),
                },
            ),
            sol_cost,
        )?;

        let bump = sale.bump;
        let seeds = &[b"public_sale".as_ref(), &[bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.sale_vault.to_account_info(),
                    to: ctx.accounts.buyer_token_account.to_account_info(),
                    authority: sale.to_account_info(),
                },
                signer,
            ),
            token_amount,
        )?;

        sale.tokens_sold += token_amount;
        sale.sol_raised += sol_cost;
        user_account.total_purchased += token_amount;
        user_account.total_sol_spent += sol_cost;

        emit!(TokensBought {
            buyer: ctx.accounts.buyer.key(),
            token_amount,
            sol_paid: sol_cost,
            new_price: get_current_price(sale.tokens_sold),
        });

        Ok(())
    }

    pub fn sell(ctx: Context<Sell>, token_amount: u64) -> Result<()> {
        let clock = Clock::get()?;
        let sale = &mut ctx.accounts.sale;
        let user_account = &mut ctx.accounts.user_account;

        require!(sale.is_active, PublicSaleError::SaleNotActive);
        require!(clock.unix_timestamp >= sale.start_time, PublicSaleError::SaleNotStarted);
        require!(clock.unix_timestamp <= sale.end_time, PublicSaleError::SaleEnded);
        require!(token_amount >= MIN_PURCHASE, PublicSaleError::BelowMinimum);
        require!(token_amount <= sale.tokens_sold, PublicSaleError::InsufficientLiquidity);

        let sol_return = calculate_sell_return(sale.tokens_sold, token_amount)?;
        let fee = sol_return * SELL_FEE_BPS / 10_000;
        let net_return = sol_return - fee;

        require!(net_return <= sale.sol_raised, PublicSaleError::InsufficientTreasuryBalance);

        // Transfer tokens from seller to sale vault
        token::transfer(
            CpiContext::new(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.seller_token_account.to_account_info(),
                    to: ctx.accounts.sale_vault.to_account_info(),
                    authority: ctx.accounts.seller.to_account_info(),
                },
            ),
            token_amount,
        )?;

        // Transfer SOL from treasury to seller
        **ctx.accounts.treasury.to_account_info().try_borrow_mut_lamports()? -= net_return;
        **ctx.accounts.seller.to_account_info().try_borrow_mut_lamports()? += net_return;

        sale.tokens_sold -= token_amount;
        sale.sol_raised -= sol_return;

        if user_account.total_purchased >= token_amount {
            user_account.total_purchased -= token_amount;
        }

        emit!(TokensSold {
            seller: ctx.accounts.seller.key(),
            token_amount,
            sol_received: net_return,
            fee_paid: fee,
            new_price: get_current_price(sale.tokens_sold),
        });

        Ok(())
    }

    pub fn get_price(ctx: Context<GetPrice>) -> Result<u64> {
        Ok(get_current_price(ctx.accounts.sale.tokens_sold))
    }

    pub fn close_sale(ctx: Context<CloseSale>) -> Result<()> {
        ctx.accounts.sale.is_active = false;
        Ok(())
    }

    pub fn withdraw_unsold(ctx: Context<WithdrawUnsold>) -> Result<()> {
        let sale = &ctx.accounts.sale;
        require!(!sale.is_active, PublicSaleError::SaleStillActive);

        let unsold = sale.total_allocation - sale.tokens_sold;
        require!(unsold > 0, PublicSaleError::NoTokensToWithdraw);

        let bump = sale.bump;
        let seeds = &[b"public_sale".as_ref(), &[bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.sale_vault.to_account_info(),
                    to: ctx.accounts.authority_token_account.to_account_info(),
                    authority: sale.to_account_info(),
                },
                signer,
            ),
            unsold,
        )?;

        Ok(())
    }
}

fn calculate_buy_cost(tokens_sold: u64, amount: u64) -> Result<u64> {
    let units = amount / PRICE_UNIT;
    if units == 0 {
        return Ok(BASE_PRICE_LAMPORTS);
    }
    let start_units = tokens_sold / PRICE_UNIT;
    let end_units = start_units + units;
    let avg_units = (start_units + end_units) / 2;
    let slope_addition = (avg_units * SLOPE_NUMERATOR) / (SLOPE_DENOMINATOR / PRICE_UNIT);
    let avg_price = BASE_PRICE_LAMPORTS + slope_addition;
    let total_cost = avg_price * units;
    Ok(total_cost)
}

fn calculate_sell_return(tokens_sold: u64, amount: u64) -> Result<u64> {
    let units = amount / PRICE_UNIT;
    if units == 0 {
        return Ok(BASE_PRICE_LAMPORTS);
    }
    let start_units = tokens_sold / PRICE_UNIT;
    let end_units = start_units.saturating_sub(units);
    let avg_units = (start_units + end_units) / 2;
    let slope_addition = (avg_units * SLOPE_NUMERATOR) / (SLOPE_DENOMINATOR / PRICE_UNIT);
    let avg_price = BASE_PRICE_LAMPORTS + slope_addition;
    let total_return = avg_price * units;
    Ok(total_return)
}

fn get_current_price(tokens_sold: u64) -> u64 {
    let units = tokens_sold / PRICE_UNIT;
    let slope_addition = (units * SLOPE_NUMERATOR) / (SLOPE_DENOMINATOR / PRICE_UNIT);
    BASE_PRICE_LAMPORTS + slope_addition
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + PublicSale::INIT_SPACE,
        seeds = [b"public_sale"],
        bump
    )]
    pub sale: Account<'info, PublicSale>,

    /// CHECK: Token mint
    pub token_mint: AccountInfo<'info>,

    #[account(mut)]
    pub sale_vault: Account<'info, TokenAccount>,

    /// CHECK: Treasury wallet
    pub treasury: AccountInfo<'info>,

    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct RegisterUser<'info> {
    #[account(seeds = [b"public_sale"], bump = sale.bump)]
    pub sale: Account<'info, PublicSale>,

    #[account(
        init,
        payer = user,
        space = 8 + UserAccount::INIT_SPACE,
        seeds = [b"user_account", user.key().as_ref()],
        bump
    )]
    pub user_account: Account<'info, UserAccount>,

    #[account(mut)]
    pub user: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Buy<'info> {
    #[account(
        mut,
        seeds = [b"public_sale"],
        bump = sale.bump
    )]
    pub sale: Account<'info, PublicSale>,

    #[account(
        mut,
        seeds = [b"user_account", buyer.key().as_ref()],
        bump = user_account.bump
    )]
    pub user_account: Account<'info, UserAccount>,

    #[account(mut)]
    pub sale_vault: Account<'info, TokenAccount>,

    #[account(mut)]
    pub buyer_token_account: Account<'info, TokenAccount>,

    /// CHECK: Treasury wallet
    #[account(mut)]
    pub treasury: AccountInfo<'info>,

    #[account(mut)]
    pub buyer: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Sell<'info> {
    #[account(
        mut,
        seeds = [b"public_sale"],
        bump = sale.bump
    )]
    pub sale: Account<'info, PublicSale>,

    #[account(
        mut,
        seeds = [b"user_account", seller.key().as_ref()],
        bump = user_account.bump
    )]
    pub user_account: Account<'info, UserAccount>,

    #[account(mut)]
    pub sale_vault: Account<'info, TokenAccount>,

    #[account(mut)]
    pub seller_token_account: Account<'info, TokenAccount>,

    /// CHECK: Treasury wallet - must have enough SOL
    #[account(mut, constraint = treasury.key() == sale.treasury)]
    pub treasury: AccountInfo<'info>,

    #[account(mut)]
    pub seller: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct GetPrice<'info> {
    #[account(seeds = [b"public_sale"], bump = sale.bump)]
    pub sale: Account<'info, PublicSale>,
}

#[derive(Accounts)]
pub struct CloseSale<'info> {
    #[account(
        mut,
        seeds = [b"public_sale"],
        bump = sale.bump,
        has_one = authority
    )]
    pub sale: Account<'info, PublicSale>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct WithdrawUnsold<'info> {
    #[account(
        seeds = [b"public_sale"],
        bump = sale.bump,
        has_one = authority
    )]
    pub sale: Account<'info, PublicSale>,

    #[account(mut)]
    pub sale_vault: Account<'info, TokenAccount>,

    #[account(mut)]
    pub authority_token_account: Account<'info, TokenAccount>,

    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[account]
#[derive(InitSpace)]
pub struct PublicSale {
    pub authority: Pubkey,
    pub token_mint: Pubkey,
    pub sale_vault: Pubkey,
    pub treasury: Pubkey,
    pub total_allocation: u64,
    pub tokens_sold: u64,
    pub sol_raised: u64,
    pub start_time: i64,
    pub end_time: i64,
    pub is_active: bool,
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct UserAccount {
    pub total_purchased: u64,
    pub total_sol_spent: u64,
    pub bump: u8,
}

#[event]
pub struct TokensBought {
    pub buyer: Pubkey,
    pub token_amount: u64,
    pub sol_paid: u64,
    pub new_price: u64,
}

#[event]
pub struct TokensSold {
    pub seller: Pubkey,
    pub token_amount: u64,
    pub sol_received: u64,
    pub fee_paid: u64,
    pub new_price: u64,
}

#[error_code]
pub enum PublicSaleError {
    #[msg("Sale is not active")]
    SaleNotActive,
    #[msg("Sale has not started yet")]
    SaleNotStarted,
    #[msg("Sale has ended")]
    SaleEnded,
    #[msg("Below minimum purchase")]
    BelowMinimum,
    #[msg("Exceeds max per wallet")]
    ExceedsMaxPerWallet,
    #[msg("Exceeds total allocation")]
    ExceedsTotalAllocation,
    #[msg("Sale still active")]
    SaleStillActive,
    #[msg("No tokens to withdraw")]
    NoTokensToWithdraw,
    #[msg("Math overflow")]
    MathOverflow,
    #[msg("Insufficient liquidity for sell")]
    InsufficientLiquidity,
    #[msg("Insufficient treasury balance")]
    InsufficientTreasuryBalance,
}

