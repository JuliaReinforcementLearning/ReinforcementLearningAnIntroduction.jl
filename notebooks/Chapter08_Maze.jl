### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 4a40583e-5993-11eb-3121-0319695416d3
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
end

# ╔═╡ 98ff088e-5994-11eb-1b32-d928c49e8466
using Flux

# ╔═╡ a0b5cd0e-5994-11eb-2572-8334d710e797
using Plots

# ╔═╡ cd235348-5994-11eb-09f1-ad1bd636ccc4
using Statistics

# ╔═╡ 52a3fc2a-5992-11eb-27b4-0da25f26e08d
md"""
# Chapter 8.2 Dyna: Integrated Planning, Acting, and Learning

To demonstrate the flexibility of `ReinforcementLearning.jl`, the `DynaAgent` is also included and we'll explore its performance in this notebook.
"""

# ╔═╡ 50dea4e0-5993-11eb-1cd3-c3e6a79800a4
md"""
## The Maze Environment

In this chapter, the authors introduced a specific maze environment. So let's define it by implementing the interfaces in `ReinforcementLearning.jl`.
"""

# ╔═╡ a75c916a-5993-11eb-1928-ef4651464d8e
const LRUD = [
    CartesianIndex(0, -1),  # left
    CartesianIndex(0, 1),   # right
    CartesianIndex(-1, 0),  # up
    CartesianIndex(1, 0),   # down
]

# ╔═╡ b8c9bd9a-5993-11eb-3acc-6532b72f9bb2
begin
	mutable struct MazeEnv <: AbstractEnv
		walls::Set{CartesianIndex{2}}
		position::CartesianIndex{2}
		start::CartesianIndex{2}
		goal::CartesianIndex{2}
		NX::Int
		NY::Int
	end

	function MazeEnv()
		walls = Set([
			[CartesianIndex(i, 3) for i = 2:4]
			CartesianIndex(5, 6)
			[CartesianIndex(j, 8) for j = 1:3]
		])
		start = CartesianIndex(3, 1)
		goal = CartesianIndex(1, 9)
		MazeEnv(walls, start, start, goal, 6, 9)
	end
	
	function (env::MazeEnv)(a::Int)
		p = env.position + LRUD[a]
		if p == env.goal
			env.position = env.goal
		elseif !(p ∈ env.walls)
			env.position = CartesianIndex(min(max(p[1], 1), env.NX), min(max(p[2], 1), env.NY))
		end
		nothing
	end
end

# ╔═╡ c289a410-5993-11eb-2601-cdb3102ee32e
RLBase.action_space(env::MazeEnv) = Base.OneTo(length(LRUD))

# ╔═╡ 35054552-5994-11eb-0541-47fc7dba1724
RLBase.reward(env::MazeEnv) = Float64(env.position == env.goal)

# ╔═╡ 46a51362-5994-11eb-3645-87be0a146644
RLBase.is_terminated(env::MazeEnv) = env.position == env.goal

# ╔═╡ 494d0848-5994-11eb-0845-857771b8ec2d
RLBase.reset!(env::MazeEnv) = env.position = env.start

# ╔═╡ 0b6f59b2-5994-11eb-2255-b96582fb8d2e
begin
	import Base: *
	
	function extend(p::CartesianIndex{2}, n::Int)
		x, y = Tuple(p)
		[CartesianIndex(n * (x - 1) + i, n * (y - 1) + j) for i = 1:n for j = 1:n]
	end

	function remap(p::CartesianIndex{2}, n::Int)
		x, y = Tuple(p)
		CartesianIndex((x - 1) * n + 1, (y - 1) * n + 1)
	end

	function *(env::MazeEnv, n::Int)
		walls = Set{CartesianIndex{2}}(ww for w in env.walls for ww in extend(w, n))
		start, position, goal = remap(env.start, n), remap(env.position, n), remap(env.goal, n)
		NX, NY = env.NX * n, env.NY * n
		MazeEnv(walls, position, start, goal, NX, NY)
	end
end

# ╔═╡ bdabb1bc-5993-11eb-2b28-2d49b0e98b68
RLBase.state_space(env::MazeEnv) = Base.OneTo(env.NX * env.NY)

# ╔═╡ 4422f2e6-5994-11eb-1e9e-33ffb5cbcbdb
RLBase.state(env::MazeEnv) = (env.position[2] - 1) * env.NX + env.position[1]

# ╔═╡ 524b4b96-5994-11eb-07c9-212ded5607d7
x = MazeEnv()

# ╔═╡ 5d4f1c16-5994-11eb-1206-0bb9f1b95d70
md"""
## Figure 8.2
"""

# ╔═╡ 77c8c1e6-5994-11eb-03d8-85cd80e3f4b0
function plan_step(n)
	env = MazeEnv()
	ns = length(state_space(env))
	na = length(action_space(env))

	agent = DynaAgent(
		policy=QBasedPolicy(
			learner=TDLearner(
				approximator=TabularQApproximator(
					n_state=ns,
					n_action=na, 
					opt=Descent(0.1)
					),
				γ=0.1,
				method=:SARS
				),
			explorer=EpsilonGreedyExplorer(0.1;is_break_tie=true)
			),
		model=ExperienceBasedSamplingModel(),
		trajectory=VectorSARTTrajectory(),
		plan_step=n
	)

	hook = StepsPerEpisode()
	run(agent, env, StopAfterEpisode(50),hook)
	hook.steps
end

# ╔═╡ c85451a2-5994-11eb-1fa3-c1f9fc128a3f
begin
	fig_8_2 = plot(legend=:topright, xlabel="Episodes", ylabel="Steps per episode")
	for n in [0, 5, 50]
		plot!(fig_8_2, mean(plan_step(n) for _ in 1:30), label="plan_step = $n")
	end
	fig_8_2
end

# ╔═╡ e5b3dbfe-5995-11eb-3c89-659c945e3d79
md"""
## Figure 8.4
"""

# ╔═╡ 19c4c11c-5996-11eb-1d02-4b044c55ed30
function cumulative_dyna_reward(model, walls, nstep1, change, nstep2)
    env = MazeEnv(
		walls,
		CartesianIndex(6, 4),
		CartesianIndex(6, 4),
		CartesianIndex(1, 9),
		6,
		9
	)
	ns = length(state_space(env))
	na = length(action_space(env))
    agent = DynaAgent(
        policy=QBasedPolicy(
            learner=TDLearner(
                approximator=TabularQApproximator(
					n_state=ns,
					n_action=na, 
					opt=Descent(1.)
					),
                γ=0.95,
                method=:SARS
                ),
            explorer=EpsilonGreedyExplorer(0.1;is_break_tie=true)
            ),
        model=model,
        trajectory=VectorSARTTrajectory(),
        plan_step=10
	)
    
    hook = StepsPerEpisode()
    run(agent, env, StopAfterStep(nstep1;is_show_progress=false),hook)
    change(env.walls)
    run(agent, env, StopAfterStep(nstep2;is_show_progress=false),hook)
	
	cumulative_reward = []
	for (i, n) in enumerate(hook.steps)
		for _ in 1:n
			push!(cumulative_reward, i)
		end
	end
	for _ in (nstep1+nstep2):-1:length(cumulative_reward)
		push!(cumulative_reward, length(hook.steps))
	end
	cumulative_reward
end

# ╔═╡ 3e68d0a0-5997-11eb-28b0-9fe9f480212e
walls() = Set([CartesianIndex(4, j) for j in 1:8])

# ╔═╡ 5040c99a-5997-11eb-3b1f-ff6a7e7fc122
function change_walls(walls)
    pop!(walls, CartesianIndex(4,1))
    push!(walls, CartesianIndex(4,9))
end

# ╔═╡ d3463894-59a0-11eb-2bcf-0dc30ae600dc
begin
	fig_8_4 = plot(legend=:topleft, xlabel="Time steps", ylabel="Cumulative reward")
	plot!(fig_8_4, mean(cumulative_dyna_reward(ExperienceBasedSamplingModel(), walls(), 1000, change_walls, 2000) for _ in 1:30), label="Dyna-Q")
	plot!(fig_8_4, mean(cumulative_dyna_reward(TimeBasedSamplingModel(;n_actions=4), walls(), 1000, change_walls, 2000) for _ in 1:30), label="Dyna-Q+")
	fig_8_4
end

# ╔═╡ af1b3f58-59a2-11eb-2cdc-073dd3f3fbc4
md"""
## Figure 8.5
"""

# ╔═╡ 3b4c6844-59a1-11eb-3e95-8d9ebfbe393b
new_walls() = Set([CartesianIndex(4, j) for j in 2:9])

# ╔═╡ 4fb4ac62-59a1-11eb-3fe1-ebeea8006edb
function new_change_walls(walls)
    pop!(walls, CartesianIndex(4,9))
end

# ╔═╡ b1d8fc48-59a1-11eb-27f0-cdc20954dbb3
begin
	fig_8_5 = plot(legend=:topleft, ylabel="Cumulative reward", xlabel="Time steps")
	plot!(fig_8_5, mean(cumulative_dyna_reward(ExperienceBasedSamplingModel(), new_walls(), 3000, new_change_walls, 3000) for _ in 1:50), label="dyna-Q")
	plot!(fig_8_5, mean(cumulative_dyna_reward(TimeBasedSamplingModel(n_actions=4, κ = 1e-3), new_walls(), 3000, new_change_walls, 3000) for _ in 1:50), label="dyna-Q+")
	fig_8_5
end

# ╔═╡ bb2fa194-59a2-11eb-1b5b-39fe96f94dd9
md"""
## Example 8.4
"""

# ╔═╡ 23b5f902-59a3-11eb-1af3-09163b251737
function run_once(model, ratio=1)
    env = MazeEnv() * ratio
    ns = length(state_space(env))
    na = length(action_space(env))
    agent = DynaAgent(
        policy=QBasedPolicy(
            learner=TDLearner(
                approximator=TabularQApproximator(
					n_state=ns,
					n_action=na,
					opt=Descent(0.5)),
                γ=0.95,
                method=:SARS
                ),
            explorer=EpsilonGreedyExplorer(0.1;is_break_tie=true)
            ),
        model=model,
        trajectory=VectorSARTTrajectory(),
        plan_step=5
    )
    hook = StepsPerEpisode()
    run(agent, env, (args...) -> length(hook.steps) > 0 && hook.steps[end] <= 14 * ratio * 1.2,hook)
    model.sample_count
end

# ╔═╡ 721bf66c-59a3-11eb-3dc2-b965593cf427
begin
	p = plot(legend=:topleft)
	plot!(mean([run_once(ExperienceBasedSamplingModel(), ratio) for ratio in 1:6] for _ in 1:5), label="Dyna", yscale=:log10)
	plot!(mean([run_once(PrioritizedSweepingSamplingModel(), ratio) for ratio in 1:6] for _ in 1:5), label="Prioritized", yscale=:log10)
	p
end

# ╔═╡ Cell order:
# ╟─52a3fc2a-5992-11eb-27b4-0da25f26e08d
# ╠═4a40583e-5993-11eb-3121-0319695416d3
# ╟─50dea4e0-5993-11eb-1cd3-c3e6a79800a4
# ╠═a75c916a-5993-11eb-1928-ef4651464d8e
# ╠═b8c9bd9a-5993-11eb-3acc-6532b72f9bb2
# ╠═bdabb1bc-5993-11eb-2b28-2d49b0e98b68
# ╠═c289a410-5993-11eb-2601-cdb3102ee32e
# ╠═35054552-5994-11eb-0541-47fc7dba1724
# ╠═4422f2e6-5994-11eb-1e9e-33ffb5cbcbdb
# ╠═46a51362-5994-11eb-3645-87be0a146644
# ╠═494d0848-5994-11eb-0845-857771b8ec2d
# ╠═0b6f59b2-5994-11eb-2255-b96582fb8d2e
# ╠═524b4b96-5994-11eb-07c9-212ded5607d7
# ╟─5d4f1c16-5994-11eb-1206-0bb9f1b95d70
# ╠═98ff088e-5994-11eb-1b32-d928c49e8466
# ╠═77c8c1e6-5994-11eb-03d8-85cd80e3f4b0
# ╠═a0b5cd0e-5994-11eb-2572-8334d710e797
# ╠═cd235348-5994-11eb-09f1-ad1bd636ccc4
# ╠═c85451a2-5994-11eb-1fa3-c1f9fc128a3f
# ╟─e5b3dbfe-5995-11eb-3c89-659c945e3d79
# ╠═19c4c11c-5996-11eb-1d02-4b044c55ed30
# ╠═3e68d0a0-5997-11eb-28b0-9fe9f480212e
# ╠═5040c99a-5997-11eb-3b1f-ff6a7e7fc122
# ╠═d3463894-59a0-11eb-2bcf-0dc30ae600dc
# ╟─af1b3f58-59a2-11eb-2cdc-073dd3f3fbc4
# ╠═3b4c6844-59a1-11eb-3e95-8d9ebfbe393b
# ╠═4fb4ac62-59a1-11eb-3fe1-ebeea8006edb
# ╠═b1d8fc48-59a1-11eb-27f0-cdc20954dbb3
# ╟─bb2fa194-59a2-11eb-1b5b-39fe96f94dd9
# ╠═23b5f902-59a3-11eb-1af3-09163b251737
# ╠═721bf66c-59a3-11eb-3dc2-b965593cf427
