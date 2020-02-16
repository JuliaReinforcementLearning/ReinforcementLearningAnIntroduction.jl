@reexport module BlackJack

export BlackJackEnv

using ReinforcementLearningBase
using Random

const ACTIONS = [:hit, :stick]

const INDS = LinearIndices((
    2,  # player has ace or not
    11, # player's hands sum, 12 ~ 21, plus other cases that the sum is greater than 21
    10,  # dealer's hands sum, 2~11
))

deal_card() = rand(1:13)
is_ace(x) = x == 1
value(x) = is_ace(x) ? 11 : min(x, 10)

mutable struct Hands
    sum::Int
    cards::Vector{Int}
    n_usable_ace::Int
end

Hands() = Hands(0, [], false)

is_busted(h::Hands) = h.sum > 21

function Base.push!(h::Hands, x)
    is_busted(h) && throw(ArgumentError("cards in hand are already busted!"))
    push!(h.cards, x)
    h.sum += value(x)
    h.n_usable_ace += is_ace(x)

    if h.n_usable_ace > 0 && is_busted(h)
        h.sum -= 10
        h.n_usable_ace -= 1
    end
end

mutable struct BlackJackEnv <: AbstractEnv
    player_hands::Hands
    dealer_hands::Hands
    is_end::Bool
    reward::Float64
    is_exploring_start::Bool
    init::Union{Nothing,Tuple{Hands,Hands}}
end

RLBase.get_observation_space(env::BlackJackEnv) = DiscreteSpace(length(INDS))
RLBase.get_action_space(env::BlackJackEnv) = DiscreteSpace(2)

function BlackJackEnv(; is_exploring_start = false, init = nothing)
    env = BlackJackEnv(
        Hands(),
        Hands(),
        false,
        0.0,
        is_exploring_start,
        init,
    )
    init_hands!(env)
    env
end

function init_hands!(env::BlackJackEnv)
    player_hands, dealer_hands = Hands(), Hands()

    if env.is_exploring_start
        player_hands.sum = rand(12:21)
        player_hands.n_usable_ace = rand(Bool)
        dealer_hands.sum = rand(2:11)
        dealer_hands.n_usable_ace = dealer_hands.sum == 11 ? true : false
    elseif env.init === nothing
        while player_hands.sum <= 11
            push!(player_hands, deal_card())
        end
        push!(dealer_hands, deal_card())
    else
        player_hands, dealer_hands = deepcopy(env.init)
    end

    env.player_hands, env.dealer_hands = player_hands, dealer_hands
end

function (env::BlackJackEnv)(a::Int)
    if ACTIONS[a] == :hit
        push!(env.player_hands, deal_card())
        if is_busted(env.player_hands)
            env.is_end = true
            env.reward = -1.0
        else
            env.is_end = false
            env.reward = 0.0
        end
    elseif ACTIONS[a] == :stick
        while env.dealer_hands.sum < 17
            push!(env.dealer_hands, deal_card())
        end
        if is_busted(env.dealer_hands)
            env.reward = 1.0
        else
            if env.player_hands.sum > env.dealer_hands.sum
                env.reward = 1.0
            elseif env.player_hands.sum < env.dealer_hands.sum
                env.reward = -1.0
            else
                env.reward = 0.0
            end
        end
        env.is_end = true
    end
    nothing
end

function RLBase.reset!(env::BlackJackEnv)
    env.is_end = false
    env.reward = 0.0

    init_hands!(env)

    nothing
end

encode(env) =
    INDS[
        env.player_hands.n_usable_ace > 0 ? 1 : 2,
        12 <= env.player_hands.sum <= 21 ? env.player_hands.sum - 10 : 1,
        2 <= env.dealer_hands.sum <= 10 ? env.dealer_hands.sum : 1,
    ]

RLBase.observe(env::BlackJackEnv) =
    (reward = env.reward, terminal = env.is_end, state = encode(env))

end