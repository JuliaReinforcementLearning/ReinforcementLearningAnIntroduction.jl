### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 741ec564-51c8-11eb-178d-3d5e49c0e174
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
end

# ╔═╡ 9f8815e8-51c8-11eb-2f5c-259a9a7b82b4
md"""
# Chapter 6.2 Random Walk
"""

# ╔═╡ 5550a880-51ca-11eb-3569-0ddbdd915759
md"""
In this section, we'll use the `RandomWalk1D` env provided in `ReinforcementLearning`.
"""

# ╔═╡ 9d9d5a20-51ca-11eb-318a-67b23ba3fab9
env = RandomWalk1D(;rewards=0.0=>1.0)

# ╔═╡ ad5961f2-51ca-11eb-0c9f-2faf60ba2d8f
NS, NA = length(state_space(env)), length(action_space(env))

# ╔═╡ 65bdcf50-51c9-11eb-1d53-d31f41af92b3
md"""
As is explained in the book, the true values of state **A** to **E** are:
"""

# ╔═╡ 7df2251c-51c9-11eb-3827-331375dcf781
true_values = [i/6 for i in 1:5]

# ╔═╡ 96ebb024-51c9-11eb-17cc-478517c1c285
md"""
To estimate the state values, we'll use the `VBasedPolicy` with a random action generator.
"""

# ╔═╡ 45d32c2a-51ca-11eb-1bf3-8b61db88186a
create_TD_agent(α) = Agent(
    policy=VBasedPolicy(
        learner = TDLearner(
            approximator=TabularApproximator(fill(0.5, NS), Descent(α)),
            method=:SRS,
            γ=1.0,
            n=0,
        ),
        mapping = (env, V) -> rand(1:NA)
    ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ d3975284-51ca-11eb-09bf-f98c05a416a1
begin
	p_6_2_left = plot(;legend=:bottomright)
	for i in [1, 10, 100]
		agent = create_TD_agent(0.1)
		run(agent, env, StopAfterEpisode(i))
		plot!(
			p_6_2_left,
			agent.policy.learner.approximator.table[2:end - 1],
			label="episode = $i"
		)
	end
	plot!(p_6_2_left, true_values, label="true value")
	p_6_2_left
end

# ╔═╡ 8507c710-51cb-11eb-3a43-75f591e687b6
md"""
To calculate the RMS error, we need to define such a hook first.
"""

# ╔═╡ ad9768f2-51cb-11eb-080c-7723ec21f95d
Base.@kwdef struct RecordRMS <: AbstractHook
    rms::Vector{Float64} = []
end

# ╔═╡ b61c59a4-51cb-11eb-3ba9-c743ebc5bd93
(f::RecordRMS)(::PostEpisodeStage, agent, env) = push!(
	f.rms,
	sqrt(mean((agent.policy.learner.approximator.table[2:end - 1] - true_values).^2))
)

# ╔═╡ d950e14e-51cb-11eb-1215-37a2ee16c07b
md"""
Now let's take a look at the performance of `TDLearner` under different α.
"""

# ╔═╡ fbae209e-51cb-11eb-04f6-f7edeaac6085
begin 
	p_6_2_right = plot()

	for α in [0.05, 0.1, 0.15]
		rms = []
		for _ in 1:100
			agent = create_TD_agent(α)
			hook = RecordRMS()
			run(agent, env, StopAfterEpisode(100),hook)
			push!(rms, hook.rms)
		end
		plot!(p_6_2_right, mean(rms), label ="TD alpha=$α", linestyle=:dashdot)
	end
	p_6_2_right
end

# ╔═╡ 3b38abf8-51cc-11eb-2da8-31a5d185393b
md"""
Then we can compare the differences between `TDLearner` and `MonteCarloLearner`.
"""

# ╔═╡ 7eae5996-51cc-11eb-1c64-5f8f6c59bd8a
create_MC_agent(α) = Agent(
    policy=VBasedPolicy(
        learner=MonteCarloLearner(
            approximator=TabularApproximator(fill(0.5, NS), Descent(α)),
            kind=EVERY_VISIT
        ),
        mapping = (env, V) -> rand(1:NA)
    ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ eaa74d56-51cc-11eb-1e18-bd7ffda238d6
for α in [0.01, 0.02, 0.03, 0.04]
    rms = []
    for _ in 1:100
        agent = create_MC_agent(α)
        hook = RecordRMS()
        run(agent, env, StopAfterEpisode(100),hook)
        push!(rms, hook.rms)
    end
    plot!(p_6_2_right, mean(rms), label ="MC alpha=$α")
end

# ╔═╡ 0ad60a5e-51cd-11eb-21ef-6f970a4d9f9c
p_6_2_right

# ╔═╡ 76db1be0-51cd-11eb-1b66-f7823c7b7ef2
begin
	fig_6_2 = plot()

	rms = []
	for _ in 1:100
		agent = create_TD_agent(0.1)
		hook = RecordRMS()
		run(agent, env, StopAfterEpisode(100),hook)
		push!(rms, hook.rms)
	end
	plot!(fig_6_2, mean(rms), label ="TD alpha=0.1", linestyle=:dashdot)


	rms = []
	for _ in 1:100
		agent = create_MC_agent(0.1)
		hook = RecordRMS()
		run(agent, env, StopAfterEpisode(100),hook)
		push!(rms, hook.rms)
	end
	plot!(fig_6_2, mean(rms), label ="MC alpha=0.1")

	fig_6_2
end

# ╔═╡ 501c914a-51ce-11eb-3268-df4813644c08
md"""

!!! warning

	Some of you might have noticed that the above figure is not the same with the one on the book of Figure 6.2. Actually we are not doing **BATCH TRAINING** here, because we're emptying the `trajectory` at the end of each episode. We leave it as an exercise for readers to practice developing new customized algorithms with `ReinforcementLearning.jl`. 😉
"""

# ╔═╡ Cell order:
# ╟─9f8815e8-51c8-11eb-2f5c-259a9a7b82b4
# ╠═741ec564-51c8-11eb-178d-3d5e49c0e174
# ╟─5550a880-51ca-11eb-3569-0ddbdd915759
# ╠═9d9d5a20-51ca-11eb-318a-67b23ba3fab9
# ╠═ad5961f2-51ca-11eb-0c9f-2faf60ba2d8f
# ╟─65bdcf50-51c9-11eb-1d53-d31f41af92b3
# ╠═7df2251c-51c9-11eb-3827-331375dcf781
# ╟─96ebb024-51c9-11eb-17cc-478517c1c285
# ╠═45d32c2a-51ca-11eb-1bf3-8b61db88186a
# ╠═d3975284-51ca-11eb-09bf-f98c05a416a1
# ╟─8507c710-51cb-11eb-3a43-75f591e687b6
# ╠═ad9768f2-51cb-11eb-080c-7723ec21f95d
# ╠═b61c59a4-51cb-11eb-3ba9-c743ebc5bd93
# ╟─d950e14e-51cb-11eb-1215-37a2ee16c07b
# ╠═fbae209e-51cb-11eb-04f6-f7edeaac6085
# ╟─3b38abf8-51cc-11eb-2da8-31a5d185393b
# ╠═7eae5996-51cc-11eb-1c64-5f8f6c59bd8a
# ╠═eaa74d56-51cc-11eb-1e18-bd7ffda238d6
# ╠═0ad60a5e-51cd-11eb-21ef-6f970a4d9f9c
# ╠═76db1be0-51cd-11eb-1b66-f7823c7b7ef2
# ╟─501c914a-51ce-11eb-3268-df4813644c08
