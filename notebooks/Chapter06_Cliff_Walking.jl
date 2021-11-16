### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 03c25682-4e46-11eb-1b97-33699faa080f
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Plots
	using Flux
	using Statistics
end

# ╔═╡ 1290b1fa-4e4d-11eb-111b-37ade59c8f30
md"""
# Chapter06 Temporal-Difference Learning (Cliff Walking)
"""

# ╔═╡ 0f4a2ba0-4e4d-11eb-0134-5d4fb9897b33
md"""
In Example 6.6, a gridworld example of **Cliff Walking** is introduced to compare the difference between on-policy (SARSA) and off-policy (Q-learning). Although there's a package of [GridWorlds.jl](https://github.com/JuliaReinforcementLearning/GridWorlds.jl) dedicated to 2-D environments, we decide to write an independent implementation here as a showcase.
"""

# ╔═╡ 63a013f0-4e4e-11eb-2737-ff5104b993f2
begin
	const NX = 4
	const NY = 12
	const Start = CartesianIndex(4, 1)
	const Goal = CartesianIndex(4, 12)
	const LRUD = [
		CartesianIndex(0, -1),  # left
		CartesianIndex(0, 1),   # right
		CartesianIndex(-1, 0),  # up
		CartesianIndex(1, 0),   # down
	]
	const LinearInds = LinearIndices((NX, NY))
end

# ╔═╡ 89c9010e-4e4e-11eb-2d97-05ee771fd5f4
function iscliff(p::CartesianIndex{2})
    x, y = Tuple(p)
    x == 4 && y > 1 && y < NY
end

# ╔═╡ eb5e16fc-4e4e-11eb-0ce7-5b5510c7924a
# take a look at the wordmap
heatmap((!iscliff).(CartesianIndices((NX, NY))); yflip = true)

# ╔═╡ 4e8e67ac-4e51-11eb-0742-ad46e685a921
begin
	Base.@kwdef mutable struct CliffWalkingEnv <: AbstractEnv
		position::CartesianIndex{2} = Start
	end
	function (env::CliffWalkingEnv)(a::Int)
		x, y = Tuple(env.position + LRUD[a])
		env.position = CartesianIndex(min(max(x, 1), NX), min(max(y, 1), NY))
	end
end

# ╔═╡ af4da300-4e51-11eb-3821-0522243eea6a
RLBase.state(env::CliffWalkingEnv) = LinearInds[env.position]

# ╔═╡ 94e72e6e-4e51-11eb-25e4-81c253969de6
RLBase.state_space(env::CliffWalkingEnv) = Base.OneTo(length(LinearInds))

# ╔═╡ be36d904-4e51-11eb-1ead-1d89513b68e5
RLBase.action_space(env::CliffWalkingEnv) = Base.OneTo(length(LRUD))

# ╔═╡ e087f574-4e51-11eb-1abf-795bdc85f7b1
RLBase.reward(env::CliffWalkingEnv) = env.position == Goal ? 0.0 : (iscliff(env.position) ? -100.0 : -1.0)

# ╔═╡ f97f149a-4e51-11eb-22d9-f9ec9ed837a8
RLBase.is_terminated(env::CliffWalkingEnv) = env.position == Goal || iscliff(env.position)

# ╔═╡ 8e26fbea-4e51-11eb-3ae8-312009b7c7c1
RLBase.reset!(env::CliffWalkingEnv) = env.position = Start

# ╔═╡ 3d8ee702-4e52-11eb-031a-bf64267362d3
world = CliffWalkingEnv()

# ╔═╡ 30b3604c-4e52-11eb-277f-d5c354e9ec66
md"""
Now we have a workable environment. Next we create several factories to generate different policies for comparison.
"""

# ╔═╡ fd5eb880-4e52-11eb-0c30-618e1e9f8f17
begin
	NS = length(state_space(world))
	NA = length(action_space(world))
end

# ╔═╡ 5da833fc-4e52-11eb-3f96-1d129eb12a00
create_agent(α, method) = Agent(
	policy = QBasedPolicy(
		learner=TDLearner(
			approximator=TabularQApproximator(
				;n_state=NS,
				n_action=NA,
				opt=Descent(α),
			),
			method=method,
			γ=1.0,
			n=0
		),
		explorer=EpsilonGreedyExplorer(0.1)
	),
	trajectory=VectorSARTTrajectory()	
)

# ╔═╡ 95bf52b0-4e53-11eb-1740-2fff3b8a9fbe
function repeated_run(α, method, N, n_episode, is_mean=true)
	env = CliffWalkingEnv()
	rewards = []
	for _ in 1:N
		h = TotalRewardPerEpisode(;is_display_on_exit=false)
		run(
			create_agent(α, method),
			env, 
			StopAfterEpisode(n_episode;is_show_progress=false),
			h
		)
		push!(rewards, is_mean ? mean(h.rewards) : h.rewards)
	end
	mean(rewards)
end

# ╔═╡ 33ab57f2-4e55-11eb-3b53-89492ca4b5f6
begin
	p = plot(legend=:bottomright, xlabel="Episodes", ylabel="Sum of rewards during episode")
	plot!(p, repeated_run(0.5, :SARS, 1000, 500, false), label="QLearning")
	plot!(p, repeated_run(0.5, :SARSA, 1000, 500, false), label="SARSA")
	p
end

# ╔═╡ a9f81264-4e5b-11eb-2634-cb2179ea9e6b
begin
	A = 0.1:0.05:0.95
	fig_6_3 = plot(;legend=:bottomright, xlabel="α", ylabel="Sum of rewards per episode")

	plot!(fig_6_3, A, [repeated_run(α, :SARS, 100, 100) for α in A], linestyle=:dash ,markershape=:rect, label="Interim Q")
	plot!(fig_6_3, A, [repeated_run(α, :SARSA, 100, 100) for α in A], linestyle=:dash, markershape=:dtriangle, label="Interim SARSA")
	plot!(fig_6_3, A, [repeated_run(α, :ExpectedSARSA, 100, 100) for α in A], linestyle=:dash, markershape=:cross, label="Interim ExpectedSARSA")

	plot!(fig_6_3, A, [repeated_run(α, :SARS, 10, 5000) for α in A], linestyle=:solid ,markershape=:rect, label="Asymptotic interim Q")
	plot!(fig_6_3, A, [repeated_run(α, :SARSA, 10, 5000) for α in A], linestyle=:solid, markershape=:dtriangle, label="Asymptotic SARSA")
	plot!(fig_6_3, A, [repeated_run(α, :ExpectedSARSA, 10, 5000) for α in A], linestyle=:solid, markershape=:cross, label="Asymptotic ExpectedSARSA")
	fig_6_3
end

# ╔═╡ Cell order:
# ╟─1290b1fa-4e4d-11eb-111b-37ade59c8f30
# ╠═03c25682-4e46-11eb-1b97-33699faa080f
# ╟─0f4a2ba0-4e4d-11eb-0134-5d4fb9897b33
# ╠═63a013f0-4e4e-11eb-2737-ff5104b993f2
# ╠═89c9010e-4e4e-11eb-2d97-05ee771fd5f4
# ╠═eb5e16fc-4e4e-11eb-0ce7-5b5510c7924a
# ╠═4e8e67ac-4e51-11eb-0742-ad46e685a921
# ╠═af4da300-4e51-11eb-3821-0522243eea6a
# ╠═94e72e6e-4e51-11eb-25e4-81c253969de6
# ╠═be36d904-4e51-11eb-1ead-1d89513b68e5
# ╠═e087f574-4e51-11eb-1abf-795bdc85f7b1
# ╠═f97f149a-4e51-11eb-22d9-f9ec9ed837a8
# ╠═8e26fbea-4e51-11eb-3ae8-312009b7c7c1
# ╠═3d8ee702-4e52-11eb-031a-bf64267362d3
# ╟─30b3604c-4e52-11eb-277f-d5c354e9ec66
# ╠═fd5eb880-4e52-11eb-0c30-618e1e9f8f17
# ╠═5da833fc-4e52-11eb-3f96-1d129eb12a00
# ╠═95bf52b0-4e53-11eb-1740-2fff3b8a9fbe
# ╠═33ab57f2-4e55-11eb-3b53-89492ca4b5f6
# ╠═a9f81264-4e5b-11eb-2634-cb2179ea9e6b
