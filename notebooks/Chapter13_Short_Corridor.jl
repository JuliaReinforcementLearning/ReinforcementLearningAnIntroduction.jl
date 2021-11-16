### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 6e487246-5d73-11eb-2f43-1f159557c68c
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
	using LinearAlgebra:dot
	using LaTeXStrings
end

# ╔═╡ 41796f22-5d73-11eb-0365-f35b80ba9bbb
md"""
# Chapter 13 Short Corridor
"""

# ╔═╡ 8fa18536-5d73-11eb-398d-b93cbd63aedc
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
	RLBase.reward(env::ShortCorridorEnv) =  env.position == 4 ? 0.0 : -1.0
end

# ╔═╡ 9a269140-5d73-11eb-01a6-a78c2b583887
world = ShortCorridorEnv()

# ╔═╡ aab4885a-5d73-11eb-33c7-b97e45de4016
ns, na = length(state_space(world)), length(action_space(world))

# ╔═╡ ae7c57c4-5d73-11eb-1734-f531d5b4cc6e
function run_once(A)
    avg_rewards = []
    for ϵ in A
        p = TabularRandomPolicy(;table=Dict(s => [1-ϵ, ϵ] for s in 1:ns))
        env = ShortCorridorEnv()
        hook=TotalRewardPerEpisode()
        run(p, env, StopAfterEpisode(1000),hook)
        push!(avg_rewards, mean(hook.rewards[end-100:end]))
    end
    avg_rewards
end

# ╔═╡ 2159d8de-5d74-11eb-0f9b-9d4fbb340ad6
X = 0.05:0.05:0.95

# ╔═╡ c2a474fe-5d73-11eb-0073-2f4afc5a1ada
plot(X, mean([run_once(X) for _ in 1:10]), legend=nothing,
	xlabel="probability of right action", ylabel=L"J(\theta)=v_{\pi_\theta}(S)")

# ╔═╡ d39bd212-5d73-11eb-0baf-b54fd44c2337
md"""
## REINFORCE Policy

Based on descriptions in Chapter 13.1, we need to define a new customized approximator.
"""

# ╔═╡ 33c22aaa-5d74-11eb-2271-e1c21336c28a
begin
	Base.@kwdef struct LinearPreferenceApproximator{F,O} <: AbstractApproximator
		weight::Vector{Float64}
		feature_func::F
		actions::Int
		opt::O
	end

	function (A::LinearPreferenceApproximator)(s)
		h = [dot(A.weight, A.feature_func(s, a)) for a in 1:A.actions]
		softmax(h)
	end

	function RLBase.update!(A::LinearPreferenceApproximator, correction::Pair)
		(s, a), Δ = correction
		w, x = A.weight, A.feature_func
		w̄ = -Δ .* (x(s,a) .- sum(A(s) .* [x(s, b) for b in 1:A.actions]))
		Flux.Optimise.update!(A.opt, w, w̄)
	end
end

# ╔═╡ 4d00b372-5d74-11eb-0760-11feb1439567
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
		G = 0.

		for i in 1:length(R)
			s,a,r = S[end-i], A[end-i], R[end-i+1]
			G = r + γ*G

			update!(Q, (s, a) => G)
		end
	end

	function RLBase.update!(
		t::AbstractTrajectory,
		::ReinforcePolicy,
		::AbstractEnv,
		::PreEpisodeStage
	)
		empty!(t)
	end
end

# ╔═╡ 5f0926f8-5d74-11eb-114c-77cfe8c57a11
function run_once_RL(α)
    agent = Agent(
        policy=ReinforcePolicy(
            approximator=LinearPreferenceApproximator(
                weight=[-1.47, 1.47],  # init_weight
                feature_func=(s,a) -> a == 1 ? [0, 1] : [1, 0],
                actions=na,
                opt=Descent(α)
            ),
            γ=1.0
        ),
        trajectory=VectorSARTTrajectory()
    )
    
    env = ShortCorridorEnv()
    hook = TotalRewardPerEpisode()
    run(agent,env,StopAfterEpisode(1000;is_show_progress=false),hook)
    hook.rewards
end

# ╔═╡ 6310c5c4-5d74-11eb-331a-8339e03cd161
begin
	fig_13_1 = plot(legend=:bottomright)
	for x in [-13, -14]  # for -12, it seems not that easy to converge in short time
		plot!(fig_13_1, mean(run_once_RL(2. ^ x) for _ in 1:50), label=L"alpha = 2^{%$x}", xlabel="Episode", ylabel="Total reward on episode")
	end
	fig_13_1
end

# ╔═╡ 9bf6ff86-5d74-11eb-2235-1bada5c181d4
md"""
Interested in how to reproduce figure 13.2? A PR is warmly welcomed! See you there!
"""

# ╔═╡ Cell order:
# ╟─41796f22-5d73-11eb-0365-f35b80ba9bbb
# ╠═6e487246-5d73-11eb-2f43-1f159557c68c
# ╠═8fa18536-5d73-11eb-398d-b93cbd63aedc
# ╠═9a269140-5d73-11eb-01a6-a78c2b583887
# ╠═aab4885a-5d73-11eb-33c7-b97e45de4016
# ╠═ae7c57c4-5d73-11eb-1734-f531d5b4cc6e
# ╠═2159d8de-5d74-11eb-0f9b-9d4fbb340ad6
# ╠═c2a474fe-5d73-11eb-0073-2f4afc5a1ada
# ╟─d39bd212-5d73-11eb-0baf-b54fd44c2337
# ╠═33c22aaa-5d74-11eb-2271-e1c21336c28a
# ╠═4d00b372-5d74-11eb-0760-11feb1439567
# ╠═5f0926f8-5d74-11eb-114c-77cfe8c57a11
# ╠═6310c5c4-5d74-11eb-331a-8339e03cd161
# ╟─9bf6ff86-5d74-11eb-2235-1bada5c181d4
