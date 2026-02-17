use anchor_lang::prelude::*;
use anchor_spl::token::{self, Mint, Token, TokenAccount, MintTo};

declare_id!("HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg");

const TOTAL_SUPPLY: u64 = 100_000_000_000_000_000; // 100M tokens with 9 decimals

#[program]
pub mod cortex_token {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let token_data = &mut ctx.accounts.token_data;
        token_data.authority = ctx.accounts.authority.key();
        token_data.mint = ctx.accounts.mint.key();
        token_data.total_minted = 0;
        token_data.bump = *ctx.bumps.get("token_data").unwrap();
        Ok(())
    }

    pub fn mint_to_treasury(ctx: Context<MintToTreasury>, amount: u64) -> Result<()> {
        require!(
            ctx.accounts.token_data.total_minted + amount <= TOTAL_SUPPLY,
            TokenError::ExceedsTotalSupply
        );

        let bump = ctx.accounts.token_data.bump;
        let seeds = &[b"token_data".as_ref(), &[bump]];
        let signer = &[&seeds[..]];

        token::mint_to(
            CpiContext::new_with_signer(
                ctx.accounts.token_program.to_account_info(),
                MintTo {
                    mint: ctx.accounts.mint.to_account_info(),
                    to: ctx.accounts.treasury_token_account.to_account_info(),
                    authority: ctx.accounts.token_data.to_account_info(),
                },
                signer,
            ),
            amount,
        )?;

        ctx.accounts.token_data.total_minted += amount;
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + TokenData::INIT_SPACE,
        seeds = [b"token_data"],
        bump
    )]
    pub token_data: Account<'info, TokenData>,
    
    #[account(
        init,
        payer = authority,
        mint::decimals = 9,
        mint::authority = token_data,
        seeds = [b"mint"],
        bump
    )]
    pub mint: Account<'info, Mint>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct MintToTreasury<'info> {
    #[account(
        mut,
        seeds = [b"token_data"],
        bump = token_data.bump,
        has_one = authority
    )]
    pub token_data: Account<'info, TokenData>,
    
    #[account(
        mut,
        seeds = [b"mint"],
        bump
    )]
    pub mint: Account<'info, Mint>,
    
    #[account(mut)]
    pub treasury_token_account: Account<'info, TokenAccount>,
    
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}

#[account]
#[derive(InitSpace)]
pub struct TokenData {
    pub authority: Pubkey,
    pub mint: Pubkey,
    pub total_minted: u64,
    pub bump: u8,
}

#[error_code]
pub enum TokenError {
    #[msg("Exceeds total supply of 100M tokens")]
    ExceedsTotalSupply,
}

