module TicTacToe

using Ju

import Ju:AbstractSyncEnvironment,
          reset!, render, observe, observationspace, actionspace, get_next_role, get_idle_action

export TicTacToeEnv

const REWARD_INVALID = -1.
const REWARD_LOSE = 0.
const REWARD_UNFINISH = 0.
const REWARD_WIN = 1.
const REWARD_TIE = 0.5

const Board = Array{Union{Symbol, Nothing}, 2}
const Roles = [:offensive, :defensive]
get_next_role(role::Union{Symbol, Nothing}) =  (role == :defensive || role == nothing) ? :offensive : :defensive

## judge the winner

getwinner(xs, idxs) = xs[idxs[1]] == xs[idxs[2]] == xs[idxs[3]] ? xs[idxs[1]] : nothing

const CHECK_INDS = vcat([(i,i+3,i+6) for i in 1:3],     # rows
                        [(3i-2,3i-1,3i) for i in 1:3],  # cols
                        [(1,5,9), (3,5,7)])             # diag

function getwinner(state::Board)
    for i in eachindex(CHECK_INDS)
        w = getwinner(state, CHECK_INDS[i])
        w != nothing && return w
    end
    nothing
end

## pre calculate all valid states

validactions(state::Board) = Set([i for (i, x) in enumerate(state) if x == nothing])

function get_next_state(state::Board, role::Symbol, action::Int)
    s = copy(state)
    s[action] = role
    s
end

function get_next_states(state::Board, role::Symbol, actions::Set{Int}=validactions(state))
    (get_next_state(state, role, action) for action in actions)
end

function get_states_info()
    init_state, states_info, unfinished_states  = Board(nothing, 3,3), Dict(), Set()
    states_info[init_state] = (isdone=false, winner=nothing)
    push!(unfinished_states, (init_state, :offensive))
    while length(unfinished_states) > 0
        state, role = pop!(unfinished_states)
        for s in get_next_states(state, role)
            if !haskey(states_info, s)
                winner = getwinner(s)
                isdone = winner != nothing || length(validactions(s)) == 0
                states_info[s] = (isdone=isdone, winner=winner)
                if !isdone
                    push!(unfinished_states, (s, get_next_role(role)))
                end
            end
        end
    end
    states_info
end

const allstates = get_states_info()
const state_ids = Dict(s=>i for (i, s) in enumerate(keys(allstates)))
const id2state =  Dict(i=>s for (i, s) in enumerate(keys(allstates)))
const INVALID_STATE = length(allstates) + 1
const IDLE_ACTION = 10  # board size (3 * 3) + idle action (1)

function estimation_of(role)
    est = zeros(length(allstates) + 1)
    est[INVALID_STATE] = -1.
    for i in 1:length(id2state)
        isdone, winner = allstates[id2state[i]]
        if isdone
            if winner == nothing
                est[i] = 0.5
            elseif winner == role
                est[i] = 1.
            else
                est[i] = 0.
            end
        else
            est[i] = 0.5
        end
    end
    est
end

function get_possible_action_state(s, role)
    board = id2state[s]
    actions = validactions(board)
    states = get_next_states(board, role, actions)
    collect(actions), (state_ids[s] for s in states)
end

"""
    TicTacToeEnv()

Using a 3 * 3 Array to simulate the [tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe) game.
"""
mutable struct TicTacToeEnv <: AbstractSyncEnvironment{DiscreteSpace, DiscreteSpace, 2}
    role::Union{Nothing, Symbol}
    board::Board
    state::Int
    function TicTacToeEnv()
        init_board = Board(nothing, 3,3)
        new(nothing, init_board, state_ids[init_board])
    end
end

getwinner(env::TicTacToeEnv) = allstates[env.board].winner
isdone(env::TicTacToeEnv) = env.state == INVALID_STATE ?  true : allstates[env.board].isdone

## interfaces
observationspace(env::TicTacToeEnv) = DiscreteSpace(length(allstates) + 1)  # here the 1 is for `INVALID_STATE`
actionspace(env::TicTacToeEnv) = DiscreteSpace(length(env.board)+1)  # here the 1 is used for idle action

function observe(env::TicTacToeEnv, role)
    if env.state == INVALID_STATE
        if role == env.role
            reward = REWARD_INVALID
        else
            reward = REWARD_UNFINISH
        end
    else
        done, winner = allstates[env.board]
        if done
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
    end

    (observation = env.state,
     isdone      = isdone(env),
     reward      = reward)
end

function reset!(env::TicTacToeEnv) 
    fill!(env.board, nothing)
    env.role = nothing
    env.state = state_ids[env.board]
end

get_next_role(env::TicTacToeEnv) = isdone(env) ? nothing : get_next_role(env.role)

function (env::TicTacToeEnv)(action::Int, role::Symbol) 
    isdone(env) && throw(ArgumentError("env is already done!"))
    nextrole = get_next_role(env)
    role != nextrole && throw(ArgumentError("invalid role of $role, should be $nextrole"))
    env.role = role

    if action in validactions(env.board)
        env.board[action] = role
        env.state = state_ids[env.board]
        (observation = env.state,
        reward = isdone(env) ? (getwinner(env) == nothing ? 0.5 : Float64(getwinner(env) == role)) : 0.,
        isdone = isdone(env))
    else
        env.state = INVALID_STATE
        fill!(env.board, nothing)
        (observation = INVALID_STATE,
        reward = REWARD_INVALID,
        isdone = true)
    end
end

get_idle_action(env::TicTacToeEnv) = IDLE_ACTION

function render(env::TicTacToeEnv)
    for r in 1:3
        for c in 1:3
            s = env.board[r,c]
            print(s == nothing ? "_" : (s == :offensive ? "X" : "O"))
        end
        println()
    end
    println("isdone = [$(isdone(env))], winner = $(repr(getwinner(env)))\n")
end
end