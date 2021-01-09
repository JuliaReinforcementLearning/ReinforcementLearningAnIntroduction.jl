### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ d9097a02-51d0-11eb-304e-cb6b5940b4c7
begin
	using ReinforcementLearning
	using Flux

	const NX = 7
	const NY = 10
	const Wind = [CartesianIndex(w, 0) for w in [0, 0, 0, -1, -1, -1, -2, -2, -1, 0]]
	const StartPosition = CartesianIndex(4, 1)
	const Goal = CartesianIndex(4, 8)
	const Actions = [
		CartesianIndex(0, -1),  # left
		CartesianIndex(0, 1),   # right
		CartesianIndex(-1, 0),  # up
		CartesianIndex(1, 0),   # down
	]

	const LinearInds = LinearIndices((NX, NY))

	Base.@kwdef mutable struct WindyGridWorldEnv <: AbstractEnv
		position::CartesianIndex{2} = StartPosition
	end

	RLBase.state_space(env::WindyGridWorldEnv) = Base.OneTo(length(LinearInds))
	RLBase.action_space(env::WindyGridWorldEnv) = Base.OneTo(length(Actions))

	function (env::WindyGridWorldEnv)(a::Int)
		p = env.position + Wind[env.position[2]] + Actions[a]
		p = CartesianIndex(min(max(p[1], 1), NX), min(max(p[2], 1), NY))
		env.position = p
		nothing
	end

	RLBase.state(env::WindyGridWorldEnv) = LinearInds[env.position]
	RLBase.is_terminated(env::WindyGridWorldEnv) = env.position == Goal
	RLBase.reward(env::WindyGridWorldEnv) = env.position == Goal ? 0.0 : -1.0

	RLBase.reset!(env::WindyGridWorldEnv) = env.position = StartPosition
end

# ╔═╡ 807e29b8-51d1-11eb-05bd-87574d1e6154
using Plots

# ╔═╡ feaaa9bc-51cf-11eb-1664-9f23f5cdc759
md"""
# Example 6.5: Windy Gridworld

First, let's define this environment by implementing the interfaces defined in `RLBase`.
"""

# ╔═╡ 412e68f4-51d1-11eb-2fe9-590a4f9d0cd3
world = WindyGridWorldEnv()

# ╔═╡ 63aec388-51d1-11eb-0d45-413471e91ea5
agent = Agent(
    policy=QBasedPolicy(
        learner=TDLearner(
            approximator=TabularQApproximator(
                ;n_state=length(state_space(world)),
                n_action=length(action_space(world)),
                opt=Descent(0.5)
                ),
            method=:SARSA
        ),
        explorer=EpsilonGreedyExplorer(0.1)
    ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ 6e6a7f7e-51d1-11eb-296a-bb55d6d3a66c
hook = StepsPerEpisode()

# ╔═╡ 748e0e98-51d1-11eb-059f-5392e547fbfe
run(agent, world, StopAfterStep(8000),hook)

# ╔═╡ 840462b2-51d1-11eb-3bf4-f33179eed650
plot([i for (i, x) in enumerate(hook.steps) for _ in 1:x])

# ╔═╡ Cell order:
# ╟─feaaa9bc-51cf-11eb-1664-9f23f5cdc759
# ╠═d9097a02-51d0-11eb-304e-cb6b5940b4c7
# ╠═412e68f4-51d1-11eb-2fe9-590a4f9d0cd3
# ╠═63aec388-51d1-11eb-0d45-413471e91ea5
# ╠═6e6a7f7e-51d1-11eb-296a-bb55d6d3a66c
# ╠═748e0e98-51d1-11eb-059f-5392e547fbfe
# ╠═807e29b8-51d1-11eb-05bd-87574d1e6154
# ╠═840462b2-51d1-11eb-3bf4-f33179eed650
