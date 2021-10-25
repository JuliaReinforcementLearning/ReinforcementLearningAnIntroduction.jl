### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 94075d98-522a-11eb-00bd-8709e96862e9
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Statistics
	using Flux
	using Plots
end

# ╔═╡ 285f8200-522a-11eb-0260-a3c27fa0762a
md"""
# Chapter 7.2 n-step Sarsa
"""

# ╔═╡ b299f0f4-522a-11eb-040a-e1a9cebe840e
env = RandomWalk1D(N=21)

# ╔═╡ c39deeb6-522a-11eb-1e1e-5193f0ad9fe0
ns, na = length(state_space(env)), length(action_space(env))

# ╔═╡ c7c57a3e-522a-11eb-0fda-91c3aff791d7
true_values = -1:0.1:1

# ╔═╡ dd9e49c6-522a-11eb-2a68-0fc5a404a0cd
md"""
Again, we first define a hook to calculate RMS
"""

# ╔═╡ efe41c64-522a-11eb-20d9-f5b14b5c138c
begin
	struct RecordRMS <: AbstractHook
		rms::Vector{Float64}
		RecordRMS() = new([])
	end
	(f::RecordRMS)(::PostEpisodeStage, agent, env) = push!(f.rms, sqrt(mean((agent.policy.learner.approximator.table[2:end-1] - true_values[2:end-1]).^2)))
end

# ╔═╡ f7443b06-522a-11eb-3801-5525ecada397
function run_once(α, n)
    env = RandomWalk1D(N=21)
    agent = Agent(
        policy=VBasedPolicy(
            learner=TDLearner(
                approximator=TabularVApproximator(;n_state=ns, opt=Descent(α)), 
                method=:SRS,
                n=n
                ),
            mapping= (env, V) -> rand(1:na)
            ),
        trajectory=VectorSARTTrajectory()
    )
    hook = RecordRMS()
    run(agent, env, StopAfterEpisode(10; is_show_progress=false), hook)
    mean(hook.rms)
end

# ╔═╡ 0b1b0448-522b-11eb-0264-9131eabe2525
begin
	A = 0.:0.05:1.0
	p = plot(xlabel="α", ylabel="Average RMS error")
	for n in [2^i for i in 0:9]
		avg_rms = Float64[]
		for α in A
			rms = []
			for _ in 1:100
				push!(rms, run_once(α, n))
			end
			push!(avg_rms, mean(rms))
		end
		plot!(p, A, avg_rms, label="n = $n")
	end

	ylims!(p, 0.25, 0.55)
	p
end

# ╔═╡ Cell order:
# ╟─285f8200-522a-11eb-0260-a3c27fa0762a
# ╠═94075d98-522a-11eb-00bd-8709e96862e9
# ╠═b299f0f4-522a-11eb-040a-e1a9cebe840e
# ╠═c39deeb6-522a-11eb-1e1e-5193f0ad9fe0
# ╠═c7c57a3e-522a-11eb-0fda-91c3aff791d7
# ╟─dd9e49c6-522a-11eb-2a68-0fc5a404a0cd
# ╠═efe41c64-522a-11eb-20d9-f5b14b5c138c
# ╠═f7443b06-522a-11eb-3801-5525ecada397
# ╠═0b1b0448-522b-11eb-0264-9131eabe2525
