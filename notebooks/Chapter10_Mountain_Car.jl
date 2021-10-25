### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ c2e4a0d0-5cae-11eb-199f-3b8bc7ca4867
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
	using SparseArrays
end

# ╔═╡ 65614b3c-5cae-11eb-0613-93393f90cfc0
md"""
# Chapter 10 Mountain Car
"""

# ╔═╡ 054805aa-5caf-11eb-1f4a-cd04a590a023
md"""
The `MountainCarEnv` is already provided in `ReinforcementLearning.jl`. So we can use it directly here. Note that by default this environment will terminate at the maximum step of `200`. While in the example on the book there's no such restriction.
"""

# ╔═╡ db653a64-5cae-11eb-0a6d-cd071416fc50
env = MountainCarEnv()

# ╔═╡ dc826048-5cae-11eb-3141-1b589c25852a
S = state_space(env)

# ╔═╡ 02252420-5caf-11eb-2352-251cf8b60517
md"""
First let's define a `Tiling` structure to encode the state.
"""

# ╔═╡ e1e987aa-5cae-11eb-1f24-69fefb87d4e0
begin
	struct Tiling{N,Tr<:AbstractRange}
		ranges::NTuple{N,Tr}
		inds::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
	end

	Tiling(ranges::AbstractRange...) = Tiling(
		ranges,
		LinearIndices(Tuple(length(r) - 1 for r in ranges))
	)

	Base.length(t::Tiling) = reduce(*, (length(r) - 1 for r in t.ranges))

	function Base.:-(t::Tiling, xs)
		Tiling((r .- x for (r, x) in zip(t.ranges, xs))...)
	end

	encode(range::AbstractRange, x) = floor(Int, div(x - range[1], step(range)) + 1)

	encode(t::Tiling, xs) = t.inds[CartesianIndex(Tuple(map(encode,  t.ranges, xs)))]
end

# ╔═╡ a0f8a23c-5caf-11eb-01d2-1706e563dac1
begin
	ntilings = 8
	ntiles = 8
	tiling = Tiling(
		(
			range(r.left, step=(r.right-r.left)/ntiles, length=ntiles+2)
			for r in S
		)...
	)
	offset = map(x-> x.right - x.left, S) ./ (ntiles * ntilings)
	tilings = [tiling - offset .* (i-1) for i in 1:ntilings]
end

# ╔═╡ d62c2d40-5caf-11eb-36e4-5bb8ee95c24e
md"""
The rest parts are simple, we initialize `agent` and `env`, then roll out experiments:
"""

# ╔═╡ 214a577a-5cb0-11eb-356b-ab06ad454eef
function create_env_agent(α=2e-4, n=0)
    env = StateTransformedEnv(
        MountainCarEnv(;max_steps=10000),
        state_mapping=s -> sparse(map(t -> encode(t, s), tilings), 1:8, ones(8), 81, 8) |> vec
    )

    agent = Agent(
        policy=QBasedPolicy(
            learner=TDLearner(
                approximator=LinearQApproximator(
                    n_state=81*8,
                    n_action=3,
                    opt = Descent(α)
                    ),
                method=:SARSA,
                n=n
                ),
            explorer=GreedyExplorer()
            ),
        trajectory=VectorSARTTrajectory(;state=Vector{Int})
    )

    env, agent
end

# ╔═╡ 3901d7b4-5cb0-11eb-2cc3-9ff34896f465
X = range(S[1].left, stop=S[1].right, length=40)

# ╔═╡ 3c0e7bea-5cb0-11eb-3190-598e56c887ff
Y = range(S[2].left, stop=S[2].right, length=40)

# ╔═╡ 230ee9ae-5cb0-11eb-0fe5-6fd8f5482af4
function show_approximation(n)
    env, agent = create_env_agent()
    run(agent, env, StopAfterEpisode(n))
    [
		agent.policy.learner.approximator(env.state_mapping([p, v])) |> maximum
        for p in X, v in Y
	]
end

# ╔═╡ 4ba11496-5cb0-11eb-202c-7f2baba059bf
n = 10

# ╔═╡ 4905a328-5cb0-11eb-03f1-0f6f27350ade
plot(X, Y, -show_approximation(n), linetype=:wireframe,
	xlabel="Position", ylabel="Velocity", zlabel="cost-to-go", title="Episode $n")

# ╔═╡ 68779ca2-5cb0-11eb-11e4-9fdb4ea0989f
begin
	fig_10_2 = plot(legend=:topright, xlabel="Episode", ylabel="Avg. steps per episode")
	n_runs = 5  # quite slow here, need revisit
	for α in [0.1/8, 0.2/8, 0.5/8]
		avg_steps_per_episode = zeros(501)
		for _ in 1:n_runs
			local env, agent = create_env_agent(α)
			hook = StepsPerEpisode()
			run(agent, env, StopAfterEpisode(500; is_show_progress=false),hook)
			avg_steps_per_episode .+= hook.steps
		end
		plot!(fig_10_2, avg_steps_per_episode[1:end-1] ./ n_runs, yscale=:log10, label="α=$α")
	end
	fig_10_2
end

# ╔═╡ 23790cca-04cc-4c50-984c-1d8ed4a0d11d
function run_once(α, n;is_reduce=true, n_episode=50)
	env, agent = create_env_agent(α, n)
	hook = StepsPerEpisode()
	run(agent, env, StopAfterEpisode(n_episode; is_show_progress=false),hook)
	is_reduce ? mean(hook.steps) : hook.steps
end

# ╔═╡ 79e4a002-5cb0-11eb-049a-db8707d6c400
begin
	fig_10_3 = plot(xlabel="Episode", ylabel="Avg. steps per episode")
	plot!(fig_10_3, mean(run_once(0.5/8, 1; is_reduce=false, n_episode=500)[1:end-1] for _ in 1:10), yscale=:log10, label="n=1")
	plot!(fig_10_3, mean(run_once(0.3/8, 8; is_reduce=false, n_episode=500)[1:end-1] for _ in 1:10), yscale=:log10, label="n=8")
	fig_10_3
end

# ╔═╡ 832c5ce0-5cb0-11eb-06f6-4f9ad5b707a2
begin
	fig_10_4 = plot(legend=:topright, xlabel="α × number of tilings ($ntilings)", ylabel="Avg. steps per episode")
	for (A, n) in [(0.4:0.1:1.7, 1), (0.3:0.1:1.6, 2), (0.2:0.1:1.4, 4), (0.2:0.1:0.9, 8), (0.2:0.1:0.7, 16)]
		plot!(fig_10_4, A, [mean(run_once(α/8, n) for _ in 1:5) for α in A], label="n = $n")
	end
	fig_10_4
end

# ╔═╡ Cell order:
# ╟─65614b3c-5cae-11eb-0613-93393f90cfc0
# ╠═c2e4a0d0-5cae-11eb-199f-3b8bc7ca4867
# ╟─054805aa-5caf-11eb-1f4a-cd04a590a023
# ╠═db653a64-5cae-11eb-0a6d-cd071416fc50
# ╠═dc826048-5cae-11eb-3141-1b589c25852a
# ╟─02252420-5caf-11eb-2352-251cf8b60517
# ╠═e1e987aa-5cae-11eb-1f24-69fefb87d4e0
# ╠═a0f8a23c-5caf-11eb-01d2-1706e563dac1
# ╟─d62c2d40-5caf-11eb-36e4-5bb8ee95c24e
# ╠═214a577a-5cb0-11eb-356b-ab06ad454eef
# ╠═3901d7b4-5cb0-11eb-2cc3-9ff34896f465
# ╠═3c0e7bea-5cb0-11eb-3190-598e56c887ff
# ╠═230ee9ae-5cb0-11eb-0fe5-6fd8f5482af4
# ╠═4ba11496-5cb0-11eb-202c-7f2baba059bf
# ╠═4905a328-5cb0-11eb-03f1-0f6f27350ade
# ╠═68779ca2-5cb0-11eb-11e4-9fdb4ea0989f
# ╠═23790cca-04cc-4c50-984c-1d8ed4a0d11d
# ╠═79e4a002-5cb0-11eb-049a-db8707d6c400
# ╠═832c5ce0-5cb0-11eb-06f6-4f9ad5b707a2
