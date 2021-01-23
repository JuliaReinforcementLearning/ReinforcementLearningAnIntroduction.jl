### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 41d2dbc6-5d31-11eb-12d5-d9c5db4e22c0
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
end

# ╔═╡ 11ec86dc-5d31-11eb-2d63-ddafcd73800d
md"""
# Chapter12 Random Walk
"""

# ╔═╡ 53376968-5d31-11eb-1c73-d9d50607cab9
N = 21

# ╔═╡ 539f2eea-5d31-11eb-1969-25432dee8823
true_values = -1:0.1:1

# ╔═╡ 56dfb4a8-5d31-11eb-217e-d75aa440a47d
Base.@kwdef struct RecordRMS <: AbstractHook
    rms::Vector{Float64}=[]
end

# ╔═╡ 5c1854ca-5d31-11eb-013b-a1aea66007e1
(h::RecordRMS)(::PostEpisodeStage, agent, env) = push!(h.rms, sqrt(mean((agent.policy.learner.approximator.table[2:end-1] - true_values[2:end-1]).^2)))

# ╔═╡ 82f0ec74-5d31-11eb-0bbc-979653a27bf4
function create_agent_env(α, λ)
    env = RandomWalk1D(N=21)
    ns, na =  length(state_space(env)),  length(action_space(env))
    agent = Agent(
        policy=VBasedPolicy(
            learner=TDλReturnLearner(
                approximator=TabularVApproximator(;n_state=ns, opt=Descent(α)),
                γ=1.0,
                λ=λ
            ),
            mapping = (env, V) -> rand(1:na)
        ),
        trajectory=VectorSARTTrajectory()
    )
    agent, env
end

# ╔═╡ 863aba54-5d31-11eb-21d1-ab6ec717c0c4
function records(α, λ, nruns=10)
    rms = []
    for _ in 1:nruns
        hook = RecordRMS()
        run(create_agent_env(α, λ)..., StopAfterEpisode(10, is_show_progress=false),hook)
        push!(rms, mean(hook.rms))
    end
    mean(rms)
end

# ╔═╡ 8a1b50b4-5d31-11eb-26bd-03f9dbaf549d
begin
	As = [0:0.1:1, 0:0.1:1, 0:0.1:1, 0:0.1:1, 0:0.1:1, 0:0.05:0.5, 0:0.02:0.2, 0:0.01:0.1]
	Λ = [0., 0.4, .8, 0.9, 0.95, 0.975, 0.99, 1.]
	p = plot(legend=:topright)
	for (A, λ) in zip(As, Λ)
		plot!(p, A, [records(α, λ) for α in A], label="lambda = $λ")
	end
	ylims!(p, (0.25, 0.55))
	p
end

# ╔═╡ Cell order:
# ╟─11ec86dc-5d31-11eb-2d63-ddafcd73800d
# ╠═41d2dbc6-5d31-11eb-12d5-d9c5db4e22c0
# ╠═53376968-5d31-11eb-1c73-d9d50607cab9
# ╠═539f2eea-5d31-11eb-1969-25432dee8823
# ╠═56dfb4a8-5d31-11eb-217e-d75aa440a47d
# ╠═5c1854ca-5d31-11eb-013b-a1aea66007e1
# ╠═82f0ec74-5d31-11eb-0bbc-979653a27bf4
# ╠═863aba54-5d31-11eb-21d1-ab6ec717c0c4
# ╠═8a1b50b4-5d31-11eb-26bd-03f9dbaf549d
