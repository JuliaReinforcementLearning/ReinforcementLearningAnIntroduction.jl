@reexport module TicTacToe

export TicTacToeEnv

using ReinforcementLearningBase


const REWARD_LOSE = 0.0
const REWARD_UNFINISH = 0.0
const REWARD_WIN = 1.0
const REWARD_TIE = 0.5
const IDLE_ACTION = 10

struct Offensive end
const offensive = Offensive()
struct Defensive end
const defensive = Defensive()

const ROLES = (offensive, defensive)
const Board = Array{Union{Offensive,Defensive,Nothing}}
const Roles = Union{Offensive,Defensive}

const CHECK_INDS = vcat(
    [(i, i + 3, i + 6) for i = 1:3],     # rows
    [(3i - 2, 3i - 1, 3i) for i = 1:3],  # cols
    [(1, 5, 9), (3, 5, 7)],
)             # diag

function get_winner(board)
    for inds in CHECK_INDS
        w = get_winner(board, inds)
        w ≢ nothing && return w
    end
    nothing
end

get_winner(board, inds) =
    board[inds[1]] == board[inds[2]] == board[inds[3]] ? board[inds[1]] : nothing

RLBase.get_legal_actions_mask(state::Board) = state .=== nothing

function get_next_state(state::Board, role::Roles, action)
    s = copy(state)
    if action != IDLE_ACTION
        s[action] = role
    end
    s
end

get_next_states(state::Board, role::Roles, actions = findall(get_legal_actions_mask(state))) =
    (get_next_state(state, role, action) for action in actions)

get_next_state_id(id::Int, role::Roles, action) = STATE2ID[get_next_state(ID2STATE[id], role, action)]

get_next_role(::Offensive) = defensive
get_next_role(::Defensive) = offensive
get_next_role(::Nothing) = offensive  # offensive role starts the game

function get_states_info()
    init_state, states_info, unfinished_states = Board(nothing, 3, 3), Dict(), Set()
    states_info[init_state] = (isdone = false, winner = nothing)
    push!(unfinished_states, (init_state, offensive))
    while length(unfinished_states) > 0
        state, role = pop!(unfinished_states)
        for s in get_next_states(state, role)
            if !haskey(states_info, s)
                winner = get_winner(s)
                isdone = winner != nothing || sum(get_legal_actions_mask(s)) == 0
                states_info[s] = (isdone = isdone, winner = winner)
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
const ID2STATE = Dict(i => s for (i, s) in enumerate(keys(STATES_INFO)))

#####
# TicTacToeEnv
#####

"""
    TicTacToeEnv()

Using a 3 * 3 Array to simulate the [tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe) game.
"""
mutable struct TicTacToeEnv <: AbstractEnv
    role::Union{Nothing,Roles}
    board::Board
    observation_space::DiscreteSpace
    action_space::DiscreteSpace
    function TicTacToeEnv()
        init_board = Board(nothing, 3, 3)
        new(
            nothing,
            init_board,
            DiscreteSpace(length(STATES_INFO)),
            DiscreteSpace(length(init_board) + 1),
        )
    end
end

RLBase.get_observation_space(env::TicTacToeEnv) = env.observation_space
RLBase.get_action_space(env::TicTacToeEnv) = env.action_space
RLBase.get_current_player(env::TicTacToeEnv) = get_next_role(env)
RLBase.get_num_players(env::TicTacToeEnv) = 2

get_winner(env::TicTacToeEnv) = STATES_INFO[env.board].winner
is_done(env::TicTacToeEnv) = STATES_INFO[env.board].isdone

RLBase.observe(env::TicTacToeEnv) = RLBase.observe(env, get_next_role(env))

function RLBase.observe(env::TicTacToeEnv, role::Roles)
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

    (
        reward = reward,
        terminal = isdone,
        state = STATE2ID[env.board],
        legal_actions_mask = get_legal_actions_mask(env, role),
    )
end

function RLBase.reset!(env::TicTacToeEnv)
    fill!(env.board, nothing)
    env.role = nothing
    nothing
end

get_next_role(env::TicTacToeEnv) = is_done(env) ? nothing : get_next_role(env.role)

function (env::TicTacToeEnv)(action::Int)
    next_role = get_next_role(env)
    another_role = get_next_role(next_role)
    env([next_role, another_role] => [action, IDLE_ACTION])
end

function (env::TicTacToeEnv)(act_info::Pair{<:Vector,<:Vector})
    is_done(env) && throw(ArgumentError("env is already done!"))
    roles, actions = act_info
    nextrole = get_next_role(env)

    for (role, action) in zip(roles, actions)
        if role === nextrole
            env.board[action] === nothing || throw(ArgumentError("invalid action: $action"))
            env.role = role
            env.board[action] = role
        else
            action != IDLE_ACTION && throw(ArgumentError("invalid action [$action] for role [$role]"))
        end
    end

    nothing
end

get_roles(::TicTacToeEnv) = ROLES

function RLBase.get_legal_actions_mask(env::TicTacToeEnv, role)
    legal_actions = fill(false, IDLE_ACTION)
    if role === get_next_role(env)
        for i = 1:9
            legal_actions[i] = env.board[i] === nothing
        end
    else
        legal_actions[end] = true
    end
    legal_actions
end

function Base.show(io::IO, env::TicTacToeEnv)
    for r = 1:3
        for c = 1:3
            s = env.board[r, c]
            print(io, s === nothing ? "_" : s)
        end
        println(io)
    end
    println(io, "isdone = [$(is_done(env))], winner = [$(get_winner(env))]")
end

Base.show(io::IO, ::Offensive) = print(io, "X")
Base.show(io::IO, ::Defensive) = print(io, "O")

end
