### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ dab179ae-4a5a-11eb-317c-c7fa9d9ccf8f
using ReinforcementLearning

# ╔═╡ 109c4fb2-4a5b-11eb-08d5-bd6b1eb0ebe9
using Plots

# ╔═╡ 1fcd93f0-4a5c-11eb-252d-9da5bc78b08b
using StatsPlots

# ╔═╡ 1fbc2952-4b1b-11eb-3b65-75c1058a9537
using Flux

# ╔═╡ db64341a-4b1b-11eb-3f7b-f11b26f442f4
using Statistics

# ╔═╡ 538080e4-4a5a-11eb-0570-65614c5797f0
md"""
# Ten Armed Bandits Environment

In this chapter, we'll use the `MultiArmBanditsEnv` to study two main concepts in reinforcement learning: **exploration** and **exploitation**.

Let's take a look at the environment first.
"""

# ╔═╡ dfe484d4-4a5a-11eb-0224-573d091b3d08
env = MultiArmBanditsEnv()

# ╔═╡ 1427b132-4a5b-11eb-3506-e744a9a5595c
violin(
	[
		[
			begin 
				reset!(env)
				env(a)
				reward(env)
			end
			for _ in 1:100
		]
		for a in action_space(env)
	],
	leg=false
)

# ╔═╡ 69bc9e66-4a5c-11eb-0288-1930cdb31d9d
md"""
The above figure shows the reward distribution of each action. （Figure 2.1）
"""

# ╔═╡ c0ca4172-4aac-11eb-255d-8b0005441fb0
md"""
Now we create a testbed to calulate the average reward and perfect action percentage.
"""

# ╔═╡ 4bf0f782-4aad-11eb-291c-afa853f150a3
"""
A customized hook to record whether the action to take is the best action or not.
"""
Base.@kwdef struct CollectBestActions <: AbstractHook
	best_action::Int
    isbest::Vector{Bool} = []
end

# ╔═╡ d0186892-4aad-11eb-080c-d985066abbc6
function (h::CollectBestActions)(::PreActStage, agent, env, action)
	push!(h.isbest, h.best_action==action)
end

# ╔═╡ 1ff8d726-4aad-11eb-0d88-c7f6080c4072
md"""
Writing a customized hook is easy.

1. Define your `struct` and make it inherit from `AbstractHook` (optional).
1. Write your customized runtime logic by overwriting some of the following functions. By default, they will do nothing if your hook inherits from `AbstractHook`.
    - `(h::YourHook)(::PreActStage, agent, env, action)`
    - `(h::YourHook)(::PostActStage, agent, env)`
    - `(h::YourHook)(::PreEpisodeStage, agent, env)`
    - `(h::YourHook)(::PostEpisodeStage, agent, env)`
"""

# ╔═╡ 1dfcd040-4b1a-11eb-248e-990c0e029f43
function bandit_testbed(
	;explorer=EpsilonGreedyExplorer(0.1),
	true_reward=0.0,
	init=0.,
	opt=InvDecay(1.0)
)
   env = MultiArmBanditsEnv(;true_reward=true_reward)
   agent = Agent(
	   policy=QBasedPolicy(
		   learner = TDLearner(
			   approximator = TabularQApproximator(
				   n_state=length(state_space(env)),
				   n_action=length(action_space(env)),
				   init=init,
				   opt = opt
			   ),
			   γ = 1.0,
			   method=:SARSA,
			   n = 0,
		   ),
		   explorer = explorer
	   ),
	   trajectory=VectorSARTTrajectory()
	)
	h1 = CollectBestActions(;best_action=findmax(env.true_values)[2])
	h2 = TotalRewardPerEpisode()
	run(agent, env, StopAfterStep(1000), ComposedHook(h1, h2))
    h1.isbest, h2.rewards
end

# ╔═╡ 15e65f7e-4b1b-11eb-26b7-ef85168b5112
begin
	p = plot(layout=(2, 1))
	for ϵ in [0.1, 0.01, 0.0]
    	stats = [
			bandit_testbed(;explorer=EpsilonGreedyExplorer(ϵ))
			for _ in 1:2000
		]
    	plot!(p, mean(x[2] for x in stats);
			subplot=1, legend=:bottomright, label="epsilon=$ϵ")
    	plot!(p, mean(x[1] for x in stats);
			subplot=2, legend=:bottomright, label="epsilon=$ϵ")
	end
	p
end

# ╔═╡ b302e1e6-4b1c-11eb-1cb8-91be2a24d5a7
begin
	# now compare the effect of setting init value
	p_2_3 = plot(legend=:bottomright)
	
	v1 = mean(
		bandit_testbed(
			;explorer=EpsilonGreedyExplorer(0.),
			init=5., 
			opt=Descent(0.1)
		)[1]
		for _ in 1:2000
	)
	plot!(p_2_3, v1, label="Q_1=5, epsilon=0.")
	
	v2 = mean(
		bandit_testbed(
			;explorer=EpsilonGreedyExplorer(0.1),
			init=0., 
			opt=Descent(0.1)
		)[1]
		for _ in 1:2000
	)
	plot!(p_2_3, v2, label="Q_1=0, epsilon=0.1")
	p_2_3
end

# ╔═╡ 21c988d8-4b80-11eb-15aa-550e091315f5
begin
	p_2_4 = plot(legend=:bottomright)
	plot!(p_2_4, mean(bandit_testbed(;explorer=UCBExplorer(10), opt=Descent(0.1))[2] for _ in 1:5000), label="UpperConfidenceBound, c=2")
	plot!(p_2_4, mean(bandit_testbed(;explorer=EpsilonGreedyExplorer(0.1), opt=Descent(0.1))[2] for _ in 1:5000), label="epsilon-greedy, epsilon=0.1")

	p_2_4
end

# ╔═╡ 5b3f7b36-4b1e-11eb-0035-2bced9aead26
md"""
Similar to the `bandit_testbed` function, we'll create a new function to test the performance of `GradientBanditLearner`.
"""

# ╔═╡ 6d93c3d0-4b1e-11eb-2b41-af6689ba72f4
function gb_bandit_testbed(
	;baseline=0.,
	explorer=WeightedExplorer(is_normalized=true),
	true_reward=0.0,
	init=0.,
	opt=InvDecay(1.0)
)
    env = MultiArmBanditsEnv(;true_reward=true_reward)
	agent = Agent(
	   policy=QBasedPolicy(
		   learner = GradientBanditLearner(
			   approximator = TabularQApproximator(
				   n_state=length(state_space(env)),
				   n_action=length(action_space(env)),
				   init=init,
				   opt = opt
			   ),
			   baseline=baseline
		   ),
		   explorer = explorer
	   ),
	   trajectory=VectorSARTTrajectory()
	)

	h1 = CollectBestActions(;best_action=findmax(env.true_values)[2])
	h2 = TotalRewardPerEpisode()
	run(agent, env, StopAfterStep(1000), ComposedHook(h1, h2))
    h1.isbest, h2.rewards
end

# ╔═╡ 04e8320c-4b1f-11eb-3340-47f7392a8282
md"""
Note that there's a keyword argument named `baseline` in the `GradientBanditLearner`. It can be either a number or a callable function (`reward -> value`). One of such functions mentioned in the book is to calculate the average of seen rewards.
"""

# ╔═╡ b291cb0c-4b1f-11eb-3ee5-cfcdfdcae00b
Base.@kwdef mutable struct SampleAvg
    t::Int = 0
    avg::Float64 = 0.0
end

# ╔═╡ d61f3168-4b1f-11eb-2a20-2f3d1bb69cd9
function (s::SampleAvg)(x)
    s.t += 1
    s.avg += (x - s.avg) / s.t
    s.avg
end

# ╔═╡ e0e72a60-4b1f-11eb-1001-89777fd3d0f7
begin
	baseline = SampleAvg()
	[baseline(x) for x in 1:10]
end

# ╔═╡ 42525d24-4b20-11eb-099c-b10c90af166e
begin
	true_reward = 4.0

	p_2_5 = plot(legend=:bottomright)

	plot!(p_2_5, mean(gb_bandit_testbed(;opt=Descent(0.1), baseline=SampleAvg(), true_reward=true_reward)[1] for _ in 1:2000), label="alpha = 0.1, with baseline")
	plot!(p_2_5, mean(gb_bandit_testbed(;opt=Descent(0.4), baseline=SampleAvg(), true_reward=true_reward)[1] for _ in 1:2000), label="alpha = 0.4, with baseline")
	plot!(p_2_5, mean(gb_bandit_testbed(;opt=Descent(0.1), true_reward=true_reward)[1] for _ in 1:2000), label="alpha = 0.1, without baseline")
	plot!(p_2_5, mean(gb_bandit_testbed(;opt=Descent(0.4), true_reward=true_reward)[1] for _ in 1:2000), label="alpha = 0.4, without baseline")

	p_2_5
end

# ╔═╡ aad675d2-4b80-11eb-3d78-a1ef731d7d8b
begin

	p_2_6 = plot(legend=:topleft)

	plot!(p_2_6, -7:-2, [mean(mean(bandit_testbed(;explorer=EpsilonGreedyExplorer(2.0^i))[2] for _ in 1:2000)) for i in -7:-2], label="epsilon greedy")
	plot!(p_2_6, -5:1, [mean(mean(gb_bandit_testbed(;explorer=WeightedExplorer(is_normalized=true), opt=Descent(2.0^i))[2] for _ in 1:2000)) for i in -5:1], label="gradient")
	plot!(p_2_6, -4:2, [mean(mean(bandit_testbed(;explorer=UCBExplorer(10; c=2.0^i))[2] for _ in 1:2000)) for i in -4:2], label="UCB")
	plot!(p_2_6, -2:2, [mean(mean(bandit_testbed(;explorer=EpsilonGreedyExplorer(0.), init=(2.0^i))[2] for _ in 1:2000)) for i in -2:2], label="greedy with initialization")

	p_2_6
end

# ╔═╡ Cell order:
# ╟─538080e4-4a5a-11eb-0570-65614c5797f0
# ╠═dab179ae-4a5a-11eb-317c-c7fa9d9ccf8f
# ╠═dfe484d4-4a5a-11eb-0224-573d091b3d08
# ╠═109c4fb2-4a5b-11eb-08d5-bd6b1eb0ebe9
# ╠═1fcd93f0-4a5c-11eb-252d-9da5bc78b08b
# ╠═1427b132-4a5b-11eb-3506-e744a9a5595c
# ╟─69bc9e66-4a5c-11eb-0288-1930cdb31d9d
# ╟─c0ca4172-4aac-11eb-255d-8b0005441fb0
# ╠═4bf0f782-4aad-11eb-291c-afa853f150a3
# ╠═d0186892-4aad-11eb-080c-d985066abbc6
# ╟─1ff8d726-4aad-11eb-0d88-c7f6080c4072
# ╠═1fbc2952-4b1b-11eb-3b65-75c1058a9537
# ╠═db64341a-4b1b-11eb-3f7b-f11b26f442f4
# ╠═1dfcd040-4b1a-11eb-248e-990c0e029f43
# ╠═15e65f7e-4b1b-11eb-26b7-ef85168b5112
# ╠═b302e1e6-4b1c-11eb-1cb8-91be2a24d5a7
# ╠═21c988d8-4b80-11eb-15aa-550e091315f5
# ╠═5b3f7b36-4b1e-11eb-0035-2bced9aead26
# ╠═6d93c3d0-4b1e-11eb-2b41-af6689ba72f4
# ╟─04e8320c-4b1f-11eb-3340-47f7392a8282
# ╠═b291cb0c-4b1f-11eb-3ee5-cfcdfdcae00b
# ╠═d61f3168-4b1f-11eb-2a20-2f3d1bb69cd9
# ╠═e0e72a60-4b1f-11eb-1001-89777fd3d0f7
# ╠═42525d24-4b20-11eb-099c-b10c90af166e
# ╠═aad675d2-4b80-11eb-3d78-a1ef731d7d8b
