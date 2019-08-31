module TicTacToe

export TicTacToeEnv,
       reset!, observe, interact!, get_legal_actions

using ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments:reset!, observe, interact!, get_legal_actions


const REWARD_LOSE = 0.
const REWARD_UNFINISH = 0.
const REWARD_WIN = 1.
const REWARD_TIE = 0.5

struct Offensive end
const offensive = Offensive()
struct Defensive end
const defensive = Defensive()

const ROLES = (offensive, defensive)
const Board = Array{Union{Offensive, Defensive, Nothing}}
const Roles = Union{Offensive, Defensive}

const CHECK_INDS = vcat([(i,i+3,i+6) for i in 1:3],     # rows
                        [(3i-2,3i-1,3i) for i in 1:3],  # cols
                        [(1,5,9), (3,5,7)])             # diag

function get_winner(board)
    for inds in CHECK_INDS
        w = get_winner(board, inds)
        w ≢ nothing && return w
    end
    nothing
end

get_winner(board, inds) = board[inds[1]] == board[inds[2]] == board[inds[3]] ? board[inds[1]] : nothing

get_legal_actions(state::Board) = state .=== nothing

function get_next_state(state::Board, role::Roles, action)
    s = copy(state)
    s[action] = role
    s
end

get_next_states(state::Board, role::Roles, actions=findall(get_legal_actions(state))) = (get_next_state(state, role, action) for action in actions)

get_next_role(::Offensive) = defensive
get_next_role(::Defensive) = offensive
get_next_role(::Nothing) = offensive  # offensive role starts the game

function get_states_info()
    init_state, states_info, unfinished_states  = Board(nothing, 3,3), Dict(), Set()
    states_info[init_state] = (isdone=false, winner=nothing)
    push!(unfinished_states, (init_state, offensive))
    while length(unfinished_states) > 0
        state, role = pop!(unfinished_states)
        for s in get_next_states(state, role)
            if !haskey(states_info, s)
                winner = get_winner(s)
                isdone = winner != nothing || sum(get_legal_actions(s)) == 0
                states_info[s] = (isdone=isdone, winner=winner)
                if !isdone
                    push!(unfinished_states, (s, get_next_role(role)))
                end
            end
        end
    end
    states_info
end

const STATES_INFO = get_states_info()
const STATE2ID = Dict(s => i for (i, s) in enumerate(keys(STATES_INFO)))
const ID2STATE =  Dict(i => s for (i, s) in enumerate(keys(STATES_INFO)))

#####
# TicTacToeEnv
#####

"""
    TicTacToeEnv()

Using a 3 * 3 Array to simulate the [tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe) game.
"""
mutable struct TicTacToeEnv <: AbstractEnv
    role::Union{Nothing, Roles}
    board::Board
    action_space::DiscreteSpace
    state_space::DiscreteSpace
    function TicTacToeEnv()
        init_board = Board(nothing, 3,3)
        new(nothing, init_board, DiscreteSpace(length(STATES_INFO)), DiscreteSpace(length(init_board)))
    end
end

get_winner(env::TicTacToeEnv) = STATES_INFO[env.board].winner
is_done(env::TicTacToeEnv) = STATES_INFO[env.board].isdone

RLEnvs.observe(env::TicTacToeEnv) = RLEnvs.observe(env, get_next_role(env))

function RLEnvs.observe(env::TicTacToeEnv, role::Roles)
    isdone, winner = STATES_INFO[env.board]
    if isdone
        if winner == nothing
            reward = REWARD_TIE
        elseif winner == role
            reward = REWARD_WIN
        else
            reward = REWARD_LOSE
        end
    else
        reward = REWARD_UNFINISH
    end

    Observation(
        reward = reward,
        terminal = isdone,
        state = STATE2ID[env.board],
        legal_actions = get_legal_actions(env.board)
    )
end

function RLEnvs.reset!(env::TicTacToeEnv) 
    fill!(env.board, nothing)
    env.role = nothing
    nothing
end

get_next_role(env::TicTacToeEnv) = is_done(env) ? nothing : get_next_role(env.role)

RLEnvs.interact!(env::TicTacToeEnv, action::Int) = interact!(env, get_next_role(env) => action)

function RLEnvs.interact!(env::TicTacToeEnv, act_info::Pair{<:Roles, Int}) 
    role, action = act_info

    is_done(env) && throw(ArgumentError("env is already done!"))

    nextrole = get_next_role(env)
    role ≢ nextrole && throw(ArgumentError("invalid role of $role, should be $nextrole"))

    get_legal_actions(env.board)[action] || throw(ArgumentError("invalid action: $action"))

    env.role = role
    env.board[action] = role
    nothing
end

get_roles(::TicTacToeEnv) = ROLES
RLEnvs.get_legal_actions(env::TicTacToeEnv) = get_legal_actions(env.board)

function Base.show(io::IO, env::TicTacToeEnv)
    for r in 1:3
        for c in 1:3
            s = env.board[r,c]
            print(io, s === nothing ? "_" : s)
        end
        println(io)
    end
    println(io, "isdone = [$(is_done(env))], winner = [$(get_winner(env))]")
end

Base.show(io::IO, ::Offensive) = print(io, "X")
Base.show(io::IO, ::Defensive) = print(io, "O")

end
