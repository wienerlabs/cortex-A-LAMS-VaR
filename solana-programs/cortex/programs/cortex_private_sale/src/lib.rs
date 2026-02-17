use anchor_lang::prelude::*;
use anchor_lang::system_program;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("Cr3msfrK46Mx4id7thTGeihCwK2dpaXWioEWNYgETJGU");

const PRIVATE_SALE_ALLOCATION: u64 = 3_000_000_000_000_000; // 3M tokens
const PRICE_PER_TOKEN_LAMPORTS: u64 = 10_000; // $0.01 in lamports (assuming 1 SOL = $100)

#[program]
pub mod cortex_private_sale {
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
        sale.total_allocation = PRIVATE_SALE_ALLOCATION;
        sale.sold_amount = 0;
        sale.start_time = start_time;
        sale.end_time = end_time;
        sale.is_active = true;
        sale.bump = *ctx.bumps.get("sale").unwrap();
        Ok(())
    }

    pub fn add_to_whitelist(
        ctx: Context<AddToWhitelist>,
        max_allocation: u64,
    ) -> Result<()> {
        let whitelist = &mut ctx.accounts.whitelist;
        whitelist.user = ctx.accounts.user.key();
        whitelist.max_allocation = max_allocation;
        whitelist.purchased = 0;
        whitelist.bump = *ctx.bumps.get("whitelist").unwrap();
        Ok(())
    }

    pub fn purchase(ctx: Context<Purchase>, token_amount: u64) -> Result<()> {
        let clock = Clock::get()?;

        require!(ctx.accounts.sale.is_active, PrivateSaleError::SaleNotActive);
        require!(
            clock.unix_timestamp >= ctx.accounts.sale.start_time,
            PrivateSaleError::SaleNotStarted
        );
        require!(
            clock.unix_timestamp <= ctx.accounts.sale.end_time,
            PrivateSaleError::SaleEnded
        );
        require!(
            ctx.accounts.whitelist.purchased + token_amount <= ctx.accounts.whitelist.max_allocation,
            PrivateSaleError::ExceedsAllocation
        );
        require!(
            ctx.accounts.sale.sold_amount + token_amount <= ctx.accounts.sale.total_allocation,
            PrivateSaleError::ExceedsTotalAllocation
        );

        let sol_amount = (token_amount / 1_000_000_000)
            .checked_mul(PRICE_PER_TOKEN_LAMPORTS)
            .ok_or(PrivateSaleError::MathOverflow)?;

        system_program::transfer(
            CpiContext::new(
                ctx.accounts.system_program.to_account_info(),
                system_program::Transfer {
                    from: ctx.accounts.buyer.to_account_info(),
                    to: ctx.accounts.treasury.to_account_info(),
                },
            ),
            sol_amount,
        )?;

        let bump = ctx.accounts.sale.bump;
        let seeds = &[b"sale".as_ref(), &[bump]];
        let signer = &[&seeds[..]];

        token::transfer(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                Transfer {
                    from: ctx.accounts.sale_vault.to_account_info(),
                    to: ctx.accounts.buyer_token_account.to_account_info(),
                    authority: ctx.accounts.sale.to_account_info(),
                },
                signer,
            ),
            token_amount,
        )?;

        ctx.accounts.sale.sold_amount += token_amount;
        ctx.accounts.whitelist.purchased += token_amount;

        emit!(TokensPurchased {
            buyer: ctx.accounts.buyer.key(),
            amount: token_amount,
            sol_paid: sol_amount,
        });

        Ok(())
    }

    pub fn close_sale(ctx: Context<CloseSale>) -> Result<()> {
        let sale = &mut ctx.accounts.sale;
        sale.is_active = false;
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + PrivateSale::INIT_SPACE,
        seeds = [b"sale"],
        bump
    )]
    pub sale: Account<'info, PrivateSale>,
    
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
pub struct AddToWhitelist<'info> {
    #[account(
        mut,
        seeds = [b"sale"],
        bump = sale.bump,
        has_one = authority
    )]
    pub sale: Account<'info, PrivateSale>,
    
    #[account(
        init,
        payer = authority,
        space = 8 + Whitelist::INIT_SPACE,
        seeds = [b"whitelist", user.key().as_ref()],
        bump
    )]
    pub whitelist: Account<'info, Whitelist>,
    
    /// CHECK: User to whitelist
    pub user: AccountInfo<'info>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Purchase<'info> {
    #[account(
        mut,
        seeds = [b"sale"],
        bump = sale.bump
    )]
    pub sale: Account<'info, PrivateSale>,

    #[account(
        mut,
        seeds = [b"whitelist", buyer.key().as_ref()],
        bump = whitelist.bump
    )]
    pub whitelist: Account<'info, Whitelist>,

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
pub struct CloseSale<'info> {
    #[account(
        mut,
        seeds = [b"sale"],
        bump = sale.bump,
        has_one = authority
    )]
    pub sale: Account<'info, PrivateSale>,
    pub authority: Signer<'info>,
}

#[account]
#[derive(InitSpace)]
pub struct PrivateSale {
    pub authority: Pubkey,
    pub token_mint: Pubkey,
    pub sale_vault: Pubkey,
    pub treasury: Pubkey,
    pub total_allocation: u64,
    pub sold_amount: u64,
    pub start_time: i64,
    pub end_time: i64,
    pub is_active: bool,
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct Whitelist {
    pub user: Pubkey,
    pub max_allocation: u64,
    pub purchased: u64,
    pub bump: u8,
}

#[event]
pub struct TokensPurchased {
    pub buyer: Pubkey,
    pub amount: u64,
    pub sol_paid: u64,
}

#[error_code]
pub enum PrivateSaleError {
    #[msg("Sale is not active")]
    SaleNotActive,
    #[msg("Sale has not started yet")]
    SaleNotStarted,
    #[msg("Sale has ended")]
    SaleEnded,
    #[msg("Exceeds user allocation")]
    ExceedsAllocation,
    #[msg("Exceeds total allocation")]
    ExceedsTotalAllocation,
    #[msg("Math overflow")]
    MathOverflow,
}
