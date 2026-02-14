use anchor_lang::prelude::*;

declare_id!("H2LwVDsPxadCWznZjfqFEbWRpdiKNtB37KcHAN7Xg1c8");

#[program]
pub mod cortex {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let registry = &mut ctx.accounts.registry;
        registry.authority = ctx.accounts.authority.key();
        registry.guardian = ctx.accounts.authority.key();
        registry.agent_count = 0;
        registry.vault_count = 0;
        registry.is_paused = false;
        registry.bump = ctx.bumps.registry;
        Ok(())
    }

    pub fn register_agent(ctx: Context<RegisterAgent>, name: String) -> Result<()> {
        require!(!ctx.accounts.registry.is_paused, CortexError::ProtocolPaused);
        require!(name.len() <= 32, CortexError::NameTooLong);

        let agent = &mut ctx.accounts.agent;
        agent.authority = ctx.accounts.authority.key();
        agent.name = name;
        agent.is_active = true;
        agent.actions_count = 0;
        agent.last_action_slot = 0;
        agent.registered_at = Clock::get()?.unix_timestamp;
        agent.bump = ctx.bumps.agent;

        let registry = &mut ctx.accounts.registry;
        registry.agent_count += 1;

        emit!(AgentRegistered {
            agent: agent.key(),
            authority: agent.authority,
            name: agent.name.clone(),
        });

        Ok(())
    }

    pub fn deactivate_agent(ctx: Context<DeactivateAgent>) -> Result<()> {
        let agent = &mut ctx.accounts.agent;
        agent.is_active = false;

        emit!(AgentDeactivated { agent: agent.key() });
        Ok(())
    }

    pub fn set_guardian(ctx: Context<SetGuardian>, new_guardian: Pubkey) -> Result<()> {
        ctx.accounts.registry.guardian = new_guardian;
        Ok(())
    }

    pub fn pause(ctx: Context<Pause>) -> Result<()> {
        ctx.accounts.registry.is_paused = true;
        emit!(ProtocolPaused {});
        Ok(())
    }

    pub fn unpause(ctx: Context<Unpause>) -> Result<()> {
        ctx.accounts.registry.is_paused = false;
        emit!(ProtocolUnpaused {});
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + Registry::INIT_SPACE,
        seeds = [b"registry"],
        bump
    )]
    pub registry: Account<'info, Registry>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(name: String)]
pub struct RegisterAgent<'info> {
    #[account(
        mut,
        seeds = [b"registry"],
        bump = registry.bump
    )]
    pub registry: Account<'info, Registry>,
    #[account(
        init,
        payer = authority,
        space = 8 + Agent::INIT_SPACE,
        seeds = [b"agent", authority.key().as_ref()],
        bump
    )]
    pub agent: Account<'info, Agent>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct DeactivateAgent<'info> {
    #[account(
        seeds = [b"registry"],
        bump = registry.bump,
        constraint = registry.authority == authority.key() || registry.guardian == authority.key()
    )]
    pub registry: Account<'info, Registry>,
    #[account(
        mut,
        seeds = [b"agent", agent.authority.as_ref()],
        bump = agent.bump
    )]
    pub agent: Account<'info, Agent>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct SetGuardian<'info> {
    #[account(
        mut,
        seeds = [b"registry"],
        bump = registry.bump,
        constraint = registry.authority == authority.key()
    )]
    pub registry: Account<'info, Registry>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct Pause<'info> {
    #[account(
        mut,
        seeds = [b"registry"],
        bump = registry.bump,
        constraint = registry.guardian == authority.key() || registry.authority == authority.key()
    )]
    pub registry: Account<'info, Registry>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct Unpause<'info> {
    #[account(
        mut,
        seeds = [b"registry"],
        bump = registry.bump,
        constraint = registry.authority == authority.key()
    )]
    pub registry: Account<'info, Registry>,
    pub authority: Signer<'info>,
}

#[account]
#[derive(InitSpace)]
pub struct Registry {
    pub authority: Pubkey,
    pub guardian: Pubkey,
    pub agent_count: u64,
    pub vault_count: u64,
    pub is_paused: bool,
    pub bump: u8,
}

#[account]
#[derive(InitSpace)]
pub struct Agent {
    pub authority: Pubkey,
    #[max_len(32)]
    pub name: String,
    pub is_active: bool,
    pub actions_count: u64,
    pub last_action_slot: u64,
    pub registered_at: i64,
    pub bump: u8,
}

#[event]
pub struct AgentRegistered {
    pub agent: Pubkey,
    pub authority: Pubkey,
    pub name: String,
}

#[event]
pub struct AgentDeactivated {
    pub agent: Pubkey,
}

#[event]
pub struct ProtocolPaused {}

#[event]
pub struct ProtocolUnpaused {}

#[error_code]
pub enum CortexError {
    #[msg("Protocol is paused")]
    ProtocolPaused,
    #[msg("Name too long, max 32 characters")]
    NameTooLong,
    #[msg("Unauthorized")]
    Unauthorized,
}
