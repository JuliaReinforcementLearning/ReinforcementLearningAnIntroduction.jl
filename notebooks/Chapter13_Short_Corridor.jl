### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ a0c988de-f84c-11ec-3cb2-55683572ae09
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
	using LinearAlgebra: dot
	using LaTeXStrings
end

# ╔═╡ 2299a7d3-0fbd-4e13-9790-e314aeadf5fd
md"""
First, let's define the environment.
"""

# ╔═╡ b88af788-7af1-4705-a568-d905b1929b83
begin
	Base.@kwdef mutable struct ShortCorridorEnv <: AbstractEnv
		position::Int = 1
	end

	RLBase.state_space(env::ShortCorridorEnv) = Base.OneTo(4)
	RLBase.action_space(env::ShortCorridorEnv) = Base.OneTo(2)

	function (env::ShortCorridorEnv)(a)
		if env.position == 1 && a == 2
			env.position += 1
		elseif env.position == 2
			env.position += a == 1 ? 1 : -1
		elseif env.position == 3
			env.position += a == 1 ? -1 : 1
		end
		nothing
	end

	function RLBase.reset!(env::ShortCorridorEnv)
		env.position = 1
		nothing
	end

	RLBase.state(env::ShortCorridorEnv) = env.position
	RLBase.is_terminated(env::ShortCorridorEnv) = env.position == 4
	RLBase.reward(env::ShortCorridorEnv) =  -1 # reward is -1 at every step
end

# ╔═╡ 9944cbb4-b55a-4759-bafb-e0e8b3f45d85
begin
	world = ShortCorridorEnv()
	ns, na = length(state_space(world)), length(action_space(world))
end

# ╔═╡ 85404ae9-f621-4469-8855-31d307863123
md"""
For Example 13.1, create a random agent that takes the `right` action with probability `ϵ`.  We can use `TabularRandomPolicy` for this purpose.
"""

# ╔═╡ 66ac0ea9-7860-4bc2-b9b9-834025ec414e
function run_random_ϵ_agent(A, number_of_iterations::Int)
    avg_rewards = []
    for ϵ in A
        p = TabularRandomPolicy(;table=Dict(s => [1-ϵ, ϵ] for s in 1:ns))
        env = ShortCorridorEnv()
        hook=TotalRewardPerEpisode(is_display_on_exit=false)
        run(p, env, StopAfterEpisode(number_of_iterations),hook)
        push!(avg_rewards, mean(hook.rewards))
    end
    avg_rewards
end

# ╔═╡ 2a198dda-f347-4c37-aafb-178cd5f9b5f5
begin
	X = 0.05:0.01:0.95
    plot(X, run_random_ϵ_agent(X, Int(1e5)), legend=nothing, xlabel="probability of right action", ylabel=L"J(\theta)=v_{\pi_\theta}(S)")
end

# ╔═╡ ac1d2202-ee67-4b69-8de9-4d60e8b7ca35
md"""
## REINFORCE Policy

Based on descriptions in Chapter 13.1, we need to define a new customized approximator.
"""

# ╔═╡ d949b958-c8c1-4c0c-b1e7-1f90e3b8620f
begin
	Base.@kwdef struct LinearPreferenceApproximator{F,O,app<:Union{AbstractApproximator, Nothing}} <: AbstractApproximator
		weight::Vector{Float64}
		feature_func::F
		actions::Int
		opt::O
		baseline::app # estimate state values
	end

	function (A::LinearPreferenceApproximator)(s)
		h = [dot(A.weight, A.feature_func(s, a)) for a in 1:A.actions]
		softmax(h)
	end

	function EligibilityVector(app::LinearPreferenceApproximator, s, a)
		app.feature_func(s, a) .- sum(app(s) .* [app.feature_func(s, b) for b in 1:app.actions])
	end

	function RLBase.update!(A::LinearPreferenceApproximator{F,O,App}, correction::Pair) where {F, O, App<:Nothing}
		(s, a), Δ = correction
		w, x = A.weight, A.feature_func
		w̄ = -Δ .* EligibilityVector(A, s, a)
		Flux.Optimise.update!(A.opt, w, w̄)
	end

	function RLBase.update!(A::LinearPreferenceApproximator{F,O,App}, correction::Pair) where {F, O, App<:AbstractApproximator}
		(s, a), Δ = correction
		δ = Δ - A.baseline(1) # use one state to approximate all states
		update!(A.baseline, 1 => -δ)
		w, x = A.weight, A.feature_func
		w̄ = -δ .* EligibilityVector(A, s, a)
        Flux.Optimise.update!(A.opt, w, w̄)
	end
end

# ╔═╡ 90908b2d-c86b-4598-9df4-6d429f4d4880
begin
	Base.@kwdef struct ReinforcePolicy{A<:AbstractApproximator} <: AbstractPolicy
		approximator::A
		γ::Float64
	end

	(p::ReinforcePolicy)(env::AbstractEnv) = prob(p, state(env)) |> WeightedExplorer(;is_normalized=true)

	RLBase.prob(p::ReinforcePolicy, s) = p.approximator(s)

	function RLBase.update!(
		p::ReinforcePolicy,
		t::AbstractTrajectory,
		::AbstractEnv,
		::PostEpisodeStage
	)
		S, A, R = t[:state], t[:action], t[:reward]
		Q, γ = p.approximator, p.γ
		Gs = discount_rewards(R, γ)

		for i in length(R):-1:1
			s, a, r, G = S[i], A[i], R[i], Gs[i]
			update!(Q, (s, a) => G)
		end
	end
	
	# clear the trajectory before each episode
	function RLBase.update!(
		t::AbstractTrajectory,
		::ReinforcePolicy,
		::AbstractEnv,
		::PreEpisodeStage
	)
		empty!(t)
	end
end

# ╔═╡ 7addf744-7402-4c52-b742-4948edbc3d7f
function run_once_reinforce(α)
    agent = Agent(
        policy=ReinforcePolicy(
            approximator=LinearPreferenceApproximator(
				# init_weight such that the agent goes left 95%
                weight=[-log(19), 0],  
                feature_func=(s,a) -> a == 1 ? [0, 1] : [1, 0],
                actions=na,
                opt=Descent(α),
				baseline=nothing
            ),
            γ=1.0
        ),
        trajectory=VectorSARTTrajectory()
    )
    
    env = MaxTimeoutEnv(world, 300) # in case of MC failure and divergence
    hook = TotalRewardPerEpisode(is_display_on_exit=false)
    run(agent,env,StopAfterEpisode(1000;is_show_progress=false),hook)
    hook.rewards
end

# ╔═╡ 5df212b0-c084-417b-9164-12f28287b5b0
begin
	fig_13_1 = plot(legend=:bottomright)
	for x in [-12, -13, -14]
		plot!(fig_13_1, mean(run_once_reinforce(2. ^ x) for _ in 1:100), label=L"\alpha = 2^{%$x}", xlabel="Episode", ylabel="Total reward on episode")
	end
	fig_13_1
end

# ╔═╡ 84956d9c-4c6c-4c4e-832a-e14c1ea9b5e4
function run_once_reinforce_baseline(α_θ,α_w)
    agent = Agent(
        policy=ReinforcePolicy(
            approximator=LinearPreferenceApproximator(
				# init_weight such that the agent goes left 95%
                weight=[-log(19), 0],  
                feature_func=(s,a) -> a == 1 ? [0, 1] : [1, 0],
                actions=na,
                opt=Descent(α_θ),
				baseline=TabularVApproximator(
					n_state=1,
					opt=Descent(α_w)
				)
            ),
            γ=1.0
        ),
        trajectory=VectorSARTTrajectory()
    )
    
    env = MaxTimeoutEnv(world, 300) # in case of MC failure and divergence
    hook = TotalRewardPerEpisode(is_display_on_exit=false)
    run(agent,env,StopAfterEpisode(1000;is_show_progress=false),hook)
    hook.rewards
end

# ╔═╡ 441def46-ac9b-4af8-a9ed-f12611b9b428
begin
	fig_13_2 = plot(legend=:bottomright)
    plot!(fig_13_2, mean(run_once_reinforce_baseline(2^-10, 2^-6) for _ in 1:100), label="REINFORCE with baseline")
	plot!(fig_13_2, mean(run_once_reinforce(2^-13) for _ in 1:100), label="REINFORCE")
	hline!([-11.6568542495], label=L"v_*(s_0)")
	fig_13_2
end

# ╔═╡ Cell order:
# ╠═a0c988de-f84c-11ec-3cb2-55683572ae09
# ╟─2299a7d3-0fbd-4e13-9790-e314aeadf5fd
# ╠═b88af788-7af1-4705-a568-d905b1929b83
# ╠═9944cbb4-b55a-4759-bafb-e0e8b3f45d85
# ╠═85404ae9-f621-4469-8855-31d307863123
# ╠═66ac0ea9-7860-4bc2-b9b9-834025ec414e
# ╠═2a198dda-f347-4c37-aafb-178cd5f9b5f5
# ╠═ac1d2202-ee67-4b69-8de9-4d60e8b7ca35
# ╠═d949b958-c8c1-4c0c-b1e7-1f90e3b8620f
# ╠═90908b2d-c86b-4598-9df4-6d429f4d4880
# ╠═7addf744-7402-4c52-b742-4948edbc3d7f
# ╠═5df212b0-c084-417b-9164-12f28287b5b0
# ╠═84956d9c-4c6c-4c4e-832a-e14c1ea9b5e4
# ╠═441def46-ac9b-4af8-a9ed-f12611b9b428
