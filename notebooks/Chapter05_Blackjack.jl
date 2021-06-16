### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 2697620e-5e5e-11eb-3568-ab2248561ebd
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
end

# ╔═╡ fb3c6816-5e5d-11eb-06dd-b5ba442b05ed
md"""
# Chapter 5 The Blackjack Environment

In this notebook, we'll study monte carlo based methods to play the **Blackjack** game.
"""

# ╔═╡ 32125940-5e5e-11eb-3a93-a30c1afca000
md"""
As usual, let's define the environment first. The implementation of the **Blackjack** environment is mainly taken from [openai/gym](https://github.com/openai/gym/blob/c4d0af393ef9fba641bd3ebbbf1f60d291c8475d/gym/envs/toy_text/blackjack.py) with some necessary modifications for our following up experiments.
"""

# ╔═╡ 9e0bc938-5e5e-11eb-1665-c331d08ac768
begin
	# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
	DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

	mutable struct BlackjackEnv <: AbstractEnv
		dealer_hand::Vector{Int}
		player_hand::Vector{Int}
		done::Bool
		reward::Int
		init::Union{Tuple{Vector{Int}, Vector{Int}}, Nothing}
	end

	function BlackjackEnv(;init=nothing)
		env = BlackjackEnv([], [], false, 0., init)
		reset!(env)
		env
	end

	function RLBase.reset!(env::BlackjackEnv)
		empty!(env.dealer_hand)
		empty!(env.player_hand)
		if isnothing(env.init)
			push!(env.dealer_hand, rand(DECK))
			push!(env.dealer_hand, rand(DECK))
			while sum_hand(env.player_hand) < 12
				push!(env.player_hand, rand(DECK))
			end
		else
			append!(env.player_hand, env.init[1])
			append!(env.dealer_hand, env.init[2])
		end
		env.done=false
		env.reward = 0.
	end

	RLBase.state_space(env::BlackjackEnv) = Space([Base.OneTo(31), Base.OneTo(10), Base.OneTo(2)])
	RLBase.action_space(env::BlackjackEnv) = Base.OneTo(2)

	usable_ace(hand) = (1 in hand) && (sum(hand) + 10 <= 21)
	sum_hand(hand) = usable_ace(hand) ? sum(hand) + 10 : sum(hand)
	is_bust(hand) = sum_hand(hand) > 21
	score(hand) = is_bust(hand) ? 0 : sum_hand(hand)

	RLBase.state(env::BlackjackEnv) = (sum_hand(env.player_hand), env.dealer_hand[1], usable_ace(env.player_hand)+1)
	RLBase.reward(env::BlackjackEnv) = env.reward
	RLBase.is_terminated(env::BlackjackEnv) = env.done

	function (env::BlackjackEnv)(action)
		if action == 1
			push!(env.player_hand, rand(DECK))
			if is_bust(env.player_hand)
				env.done = true
				env.reward = -1
			else
				env.done = false
				env.reward = 0
			end
		elseif action == 2
			env.done = true
			while sum_hand(env.dealer_hand) < 17
				push!(env.dealer_hand, rand(DECK))
			end
			env.reward = cmp(score(env.player_hand), score(env.dealer_hand))
		else
			@error "unknown action"
		end
	end
end

# ╔═╡ f0454136-5e5e-11eb-01ad-2fda9f6d0a78
game = BlackjackEnv()

# ╔═╡ f6d2fbe0-5e5e-11eb-0cae-7b9f5d343ae3
state_space(game)

# ╔═╡ dfebc2b8-5e5e-11eb-2030-4766a533b521
md"""
As you can see, the `state_space` of the **Blackjack** environment has 3 discrete features. To reuse the tabular algorithms in `ReinforcementLearning.jl`, we need to flatten the state and wrap it in a `StateOverriddenEnv`.
"""

# ╔═╡ 651a9d74-5e5f-11eb-251d-1bab0d8d3384
STATE_MAPPING = s ->  LinearIndices((31, 10, 2))[CartesianIndex(s)]

# ╔═╡ 6f95e1a0-5e5f-11eb-1203-21668fdb4d1d
world = StateOverriddenEnv(
    BlackjackEnv(),
    STATE_MAPPING
)

# ╔═╡ 82391e12-5e5f-11eb-1eb9-f1239a781589
RLBase.state_space(x::typeof(world)) = Base.OneTo(31* 10*2)

# ╔═╡ 7e18ee70-5e5f-11eb-00aa-877362e303b3
NS = state_space(world)

# ╔═╡ cffe4ec4-5e5f-11eb-2f7d-03deb8eb3680
md"""
## Figure 5.1
"""

# ╔═╡ e56fbcca-5e5f-11eb-0c04-d3b2aa8a634e
agent = Agent(
    policy = VBasedPolicy(
        learner=MonteCarloLearner(;
            approximator=TabularVApproximator(;n_state=NS, opt=InvDecay(1.0)),
            γ = 1.0
            ),
        mapping= (env, V) -> sum_hand(env.env.player_hand) in (20, 21) ? 2 : 1
        ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ f032cbf2-5e5f-11eb-0bca-457ebd9e2f01
run(agent, world, StopAfterEpisode(10_000))

# ╔═╡ fd16d76e-5e5f-11eb-06f3-ebfe8c142e13
VT = agent.policy.learner.approximator.table

# ╔═╡ 0b4702e4-5e60-11eb-1d2a-7528302acbfe
X, Y = 12:21, 1:10

# ╔═╡ 129d507c-5e60-11eb-05de-75e719976129
plot(X, Y, [VT[STATE_MAPPING((i,j,1))] for i in X, j in Y],linetype=:wireframe)

# ╔═╡ 1d37cfda-5e60-11eb-0923-3d5f8a53521a
plot(X, Y, [VT[STATE_MAPPING((i,j,2))] for i in X, j in Y],linetype=:wireframe)

# ╔═╡ 2e18d632-5e60-11eb-3988-8de51cb1706a
# now run more simulations
run(agent, world, StopAfterEpisode(500_000))

# ╔═╡ 3b0f8098-5e60-11eb-035c-7339e9c07157
plot(X, Y, [VT[STATE_MAPPING((i,j,1))] for i in X, j in Y],linetype=:wireframe)

# ╔═╡ 2b67b004-5e60-11eb-0c57-9b94ffc1799e
plot(X, Y, [VT[STATE_MAPPING((i,j,2))] for i in X, j in Y],linetype=:wireframe)

# ╔═╡ 475e9956-5e60-11eb-1108-13deb8642ac0
md"""
## Figure 5.2
"""

# ╔═╡ 54ac6d18-5e60-11eb-030c-190885e37497
md"""
In Chapter 5.3, a **Monte Carlo Exploring Start** method is used to solve the **Blackjack** game. Although several variants of monte carlo methods are supported in `ReinforcementLearning.jl` package, they do not support the *exploring start*. Nevertheless, we can define it very easily.
"""

# ╔═╡ d33e736a-5e60-11eb-3765-69a54bbf6049
begin
	Base.@kwdef mutable struct ExploringStartPolicy{P} <: AbstractPolicy
		policy::P
		is_start::Bool = true
	end

	function (p::ExploringStartPolicy)(env::AbstractEnv)
		if p.is_start
			p.is_start = false
			rand(action_space(env))
		else
			p.policy(env)
		end
	end

	(p::ExploringStartPolicy)(s::AbstractStage, env::AbstractEnv) = p.policy(s, env)

	function (p::ExploringStartPolicy)(s::PreEpisodeStage, env::AbstractEnv)
		p.is_start = true
		p.policy(s, env)
	end

	function (p::ExploringStartPolicy)(s::PreActStage, env::AbstractEnv, action)
		p.policy(s, env, action)
	end
end

# ╔═╡ d9a5a368-5e60-11eb-0a0d-313dc4697d0d
solver = Agent(
    policy = QBasedPolicy(
        learner=MonteCarloLearner(;
            approximator=TabularQApproximator(
				;n_state=NS,
				n_action=2, 
				opt=InvDecay(1.0)),
            γ = 1.0,
            ),
        explorer=GreedyExplorer()
        ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ e904b632-5e60-11eb-05f3-abca0adeac69
run(ExploringStartPolicy(policy=solver), world, StopAfterEpisode(10_000_000))

# ╔═╡ 10750292-5e61-11eb-0a2f-51c6de1737b4
QT = solver.policy.learner.approximator.table

# ╔═╡ 114b9d7e-5e61-11eb-3579-f9fd7d0a4bd3
heatmap([argmax(QT[:,STATE_MAPPING((i,j,1))]) for i in 11:21, j in Y])

# ╔═╡ 1aec9c8c-5e61-11eb-2b12-6b9dbf4b61db
heatmap([argmax(QT[:,STATE_MAPPING((i,j,2))]) for i in 11:21, j in Y])

# ╔═╡ 15f4a8fa-5e61-11eb-2313-81caa4ed00e6
V_agent = Agent(
    policy = VBasedPolicy(
        learner=MonteCarloLearner(;
            approximator=TabularVApproximator(;n_state=NS, opt=InvDecay(1.0)),
            γ = 1.0
            ),
        mapping=(env, V) -> solver.policy(env)
        ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ 8c366ee0-5e61-11eb-31b9-6722b4dfc9a3
run(V_agent, world, StopAfterEpisode(500_000))

# ╔═╡ 937068f0-5e61-11eb-0326-4f2e7e313b83
V_agent_T = V_agent.policy.learner.approximator.table;

# ╔═╡ 9b0617b0-5e61-11eb-27be-15bcf24d11f5
plot(X, Y, [V_agent_T[STATE_MAPPING((i,j,1))] for i in X, j in Y],linetype=:wireframe)

# ╔═╡ a0a52486-5e61-11eb-212a-45ca32bfb987
plot(X, Y, [V_agent_T[STATE_MAPPING((i,j,2))] for i in X, j in Y],linetype=:wireframe)

# ╔═╡ aa68a5ec-5e61-11eb-07c8-9103a0ada51c
md"""
## Figure 5.3
"""

# ╔═╡ d874fc22-5e61-11eb-2fee-37742f910182
static_env = StateOverriddenEnv(
    BlackjackEnv(;init=([1,2], [2])),
    STATE_MAPPING
);

# ╔═╡ e31a1da6-5e61-11eb-09ac-1f29247434ca
INIT_STATE = state(static_env)

# ╔═╡ f1437b16-5e61-11eb-2f14-976483f8d924
GOLD_VAL = -0.27726

# ╔═╡ ecade5b4-5e61-11eb-0fd5-d1b149adbece
begin
	Base.@kwdef struct StoreMSE <: AbstractHook
		mse::Vector{Float64} = []
	end
	(f::StoreMSE)(::PostEpisodeStage, agent, env) = push!(f.mse, (GOLD_VAL - agent.policy.π_target.learner.approximator[1](INIT_STATE))^2)
end

# ╔═╡ 4f8a0a80-5e64-11eb-13db-3b117cdd35b6
target_policy_mapping = (env, V) -> sum_hand(env.env.player_hand) in (20, 21) ? 2 : 1

# ╔═╡ 5f5473ba-5e64-11eb-1b37-7706eb629b45
function RLBase.prob(
	p::VBasedPolicy{<:Any,typeof(target_policy_mapping)},
	env::AbstractEnv,
	a
)
	s = sum_hand(env.env.player_hand)
    if s in (20, 21)
        Int(a == 2)
    else
        Int(a == 1)
    end
end

# ╔═╡ bceb8f5a-5e63-11eb-25fb-1d73dc919323
function ordinary_mse()
	agent = Agent(
		policy=OffPolicy(
			π_target = VBasedPolicy(
				learner=MonteCarloLearner(
					approximator=(
						TabularVApproximator(;n_state=NS, opt=Descent(1.0)), 
						TabularVApproximator(;n_state=NS, opt=InvDecay(1.0))
						),
					kind=FIRST_VISIT,
					sampling=ORDINARY_IMPORTANCE_SAMPLING
					),
				mapping=target_policy_mapping
			),
			π_behavior = RandomPolicy(Base.OneTo(2))
			),
		trajectory=VectorWSARTTrajectory()
	)
	h = StoreMSE()
	run(agent, static_env, StopAfterEpisode(10_000), h)
	h.mse
end

# ╔═╡ 0d5ef608-5e65-11eb-2881-d72c5478e1ce
function weighted_mse()
	agent = Agent(
		policy=OffPolicy(
			π_target = VBasedPolicy(
				learner=MonteCarloLearner(
					approximator=(
						TabularVApproximator(;n_state=NS, opt=Descent(1.0)),
						TabularVApproximator(;n_state=NS, opt=InvDecay(1.0)),
						TabularVApproximator(;n_state=NS, opt=InvDecay(1.0))),
					kind=FIRST_VISIT,
					sampling=WEIGHTED_IMPORTANCE_SAMPLING
					),
				mapping=target_policy_mapping
			),
			π_behavior = RandomPolicy(Base.OneTo(2))
			),
		trajectory=VectorWSARTTrajectory()
	)
	h = StoreMSE()
	run(agent, static_env, StopAfterEpisode(10_000), h)
	h.mse
end

# ╔═╡ d054b474-5e64-11eb-281d-c75d62dcad55
begin
	fig_5_3 = plot()
	plot!(fig_5_3, mean(ordinary_mse() for _ in 1:100), xscale=:log10,label="Ordinary Importance Sampling")
	plot!(fig_5_3, mean(weighted_mse() for _ in 1:100), xscale=:log10,label="Weighted Importance Sampling")
	fig_5_3
end

# ╔═╡ Cell order:
# ╟─fb3c6816-5e5d-11eb-06dd-b5ba442b05ed
# ╠═2697620e-5e5e-11eb-3568-ab2248561ebd
# ╟─32125940-5e5e-11eb-3a93-a30c1afca000
# ╠═9e0bc938-5e5e-11eb-1665-c331d08ac768
# ╠═f0454136-5e5e-11eb-01ad-2fda9f6d0a78
# ╠═f6d2fbe0-5e5e-11eb-0cae-7b9f5d343ae3
# ╟─dfebc2b8-5e5e-11eb-2030-4766a533b521
# ╠═651a9d74-5e5f-11eb-251d-1bab0d8d3384
# ╠═6f95e1a0-5e5f-11eb-1203-21668fdb4d1d
# ╠═82391e12-5e5f-11eb-1eb9-f1239a781589
# ╠═7e18ee70-5e5f-11eb-00aa-877362e303b3
# ╟─cffe4ec4-5e5f-11eb-2f7d-03deb8eb3680
# ╠═e56fbcca-5e5f-11eb-0c04-d3b2aa8a634e
# ╠═f032cbf2-5e5f-11eb-0bca-457ebd9e2f01
# ╠═fd16d76e-5e5f-11eb-06f3-ebfe8c142e13
# ╠═0b4702e4-5e60-11eb-1d2a-7528302acbfe
# ╠═129d507c-5e60-11eb-05de-75e719976129
# ╠═1d37cfda-5e60-11eb-0923-3d5f8a53521a
# ╠═2e18d632-5e60-11eb-3988-8de51cb1706a
# ╠═3b0f8098-5e60-11eb-035c-7339e9c07157
# ╠═2b67b004-5e60-11eb-0c57-9b94ffc1799e
# ╟─475e9956-5e60-11eb-1108-13deb8642ac0
# ╟─54ac6d18-5e60-11eb-030c-190885e37497
# ╠═d33e736a-5e60-11eb-3765-69a54bbf6049
# ╠═d9a5a368-5e60-11eb-0a0d-313dc4697d0d
# ╠═e904b632-5e60-11eb-05f3-abca0adeac69
# ╠═10750292-5e61-11eb-0a2f-51c6de1737b4
# ╠═114b9d7e-5e61-11eb-3579-f9fd7d0a4bd3
# ╠═1aec9c8c-5e61-11eb-2b12-6b9dbf4b61db
# ╠═15f4a8fa-5e61-11eb-2313-81caa4ed00e6
# ╠═8c366ee0-5e61-11eb-31b9-6722b4dfc9a3
# ╠═937068f0-5e61-11eb-0326-4f2e7e313b83
# ╠═9b0617b0-5e61-11eb-27be-15bcf24d11f5
# ╠═a0a52486-5e61-11eb-212a-45ca32bfb987
# ╟─aa68a5ec-5e61-11eb-07c8-9103a0ada51c
# ╠═d874fc22-5e61-11eb-2fee-37742f910182
# ╠═e31a1da6-5e61-11eb-09ac-1f29247434ca
# ╠═f1437b16-5e61-11eb-2f14-976483f8d924
# ╠═ecade5b4-5e61-11eb-0fd5-d1b149adbece
# ╠═4f8a0a80-5e64-11eb-13db-3b117cdd35b6
# ╠═5f5473ba-5e64-11eb-1b37-7706eb629b45
# ╠═bceb8f5a-5e63-11eb-25fb-1d73dc919323
# ╠═0d5ef608-5e65-11eb-2881-d72c5478e1ce
# ╠═d054b474-5e64-11eb-281d-c75d62dcad55
