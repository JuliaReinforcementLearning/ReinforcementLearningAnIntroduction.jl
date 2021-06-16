### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 1f43297c-5e20-11eb-1ce3-5b75c0b5c368
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
	using StatsBase
end

# ╔═╡ 5818518e-5e20-11eb-3a74-c3f0b78d8098
md"""
## Left Right Environment
"""

# ╔═╡ 6fe63566-5e20-11eb-056c-1f513ae5d161
begin
	Base.@kwdef mutable struct LeftRightEnv <: AbstractEnv
		reward::Float64 = 0.
		current_state::Int = 1
	end

	RLBase.state_space(env::LeftRightEnv) = Base.OneTo(2)
	RLBase.action_space(env::LeftRightEnv) = Base.OneTo(2)

	function (env::LeftRightEnv)(a::Int)
		if a == 2
			env.reward = 0.
			env.current_state = 2
		else
			s = sample(Weights([0.9, 0.1], 1.0))
			if s == 1
				env.reward = 0.
				env.current_state = 1
			else
				env.reward = 1.0
				env.current_state = 2
			end
		end
	end

	function RLBase.reset!(env::LeftRightEnv)
		env.current_state = 1
		env.reward = 0.
	end

	RLBase.reward(env::LeftRightEnv) = env.reward
	RLBase.is_terminated(env::LeftRightEnv) = env.current_state == 2
	RLBase.state(env::LeftRightEnv) = env.current_state
end

# ╔═╡ 54f876d4-5e21-11eb-14be-67265069c8ae
world = LeftRightEnv()

# ╔═╡ 62022db4-5e21-11eb-220d-d163a3aa5c81
begin
	ns = length(state_space(world))
	na = length(action_space(world))
end 

# ╔═╡ 597bbcd2-5e21-11eb-10c9-7db2f46cfffb
 π_t = VBasedPolicy(
	learner=MonteCarloLearner(
		approximator=(
			TabularVApproximator(;n_state=ns, opt=Descent(1.0)), # V
			TabularVApproximator(;n_state=ns, opt=InvDecay(1.0)) # Returns
			),
		kind=FIRST_VISIT,
		sampling=ORDINARY_IMPORTANCE_SAMPLING,
		γ=1.0
		),
	mapping= (env, V) -> 1
)

# ╔═╡ 9cdef3ba-5e21-11eb-1cfd-fb1abbf1d608
# A little ad-hoc here
RLBase.prob(::typeof(π_t), s, a) = a == 1 ? 1.0 : 0.

# ╔═╡ 88b48522-5e20-11eb-00df-65834ec124b2
begin
	struct CollectValue <: AbstractHook
		values::Vector{Float64}
		CollectValue() = new([])
	end
	(f::CollectValue)(::PostEpisodeStage, agent, env) = push!(f.values, agent.policy.π_target.learner.approximator[2](1))
end

# ╔═╡ 4e0a26b0-5e21-11eb-01bf-85db236b9bf8
begin
	p = plot()
	for _ in 1:10
		fill!(π_t.learner.approximator[1].table, 0.)
		fill!(π_t.learner.approximator[2].table, 0.)
		empty!(π_t.learner.approximator[2].optimizer.state)
		agent = Agent(
			policy=OffPolicy(
				π_t,
				RandomPolicy(1:na)
				),
			trajectory=VectorWSARTTrajectory()
		)
		hook = CollectValue()
		run(agent, world, StopAfterEpisode(100_000, is_show_progress=true),hook)
		plot!(p, hook.values, xscale = :log10)
	end
	p
end

# ╔═╡ Cell order:
# ╠═1f43297c-5e20-11eb-1ce3-5b75c0b5c368
# ╟─5818518e-5e20-11eb-3a74-c3f0b78d8098
# ╠═6fe63566-5e20-11eb-056c-1f513ae5d161
# ╠═54f876d4-5e21-11eb-14be-67265069c8ae
# ╠═62022db4-5e21-11eb-220d-d163a3aa5c81
# ╠═597bbcd2-5e21-11eb-10c9-7db2f46cfffb
# ╠═9cdef3ba-5e21-11eb-1cfd-fb1abbf1d608
# ╠═88b48522-5e20-11eb-00df-65834ec124b2
# ╠═4e0a26b0-5e21-11eb-01bf-85db236b9bf8
