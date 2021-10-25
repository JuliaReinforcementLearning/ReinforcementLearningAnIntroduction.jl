### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 9cd5e7cc-5c6c-11eb-3485-011143087d8d
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots, Measures
	using SparseArrays
end

# ╔═╡ 41c57942-5c6c-11eb-3c99-9345b6668a1a
md"""
# Chapter 9 On-policy Prediction with Approximation

In this notebook, we'll focus on the linear approximation methods.
"""

# ╔═╡ be5a557c-5c6c-11eb-035e-39e94df41852
md"""
## Figure 9.1

We've discussed the `RandomWalk1D` environment before. In previous example, the state space is relatively small (`1:7`). Here we expand it into `1:1000` and see how the `LinearVApproximator` will work here.
"""

# ╔═╡ 6a0881f0-5c6d-11eb-143e-0196833abc05
ACTIONS = collect(Iterators.flatten((-100:-1, 1:100)))

# ╔═╡ 7ee0867c-5c6d-11eb-11b4-a7858177564f
NA = length(ACTIONS)

# ╔═╡ 7aae4986-5c6d-11eb-09b0-fd883165bc72
NS = 1002

# ╔═╡ 8fc27a60-5c6d-11eb-05ff-dbdcd106b853
md"""
First, let's roll out a large experiment to calculate the *true* state values of each state:
"""

# ╔═╡ c4d3a538-5c6d-11eb-3369-9bd67cc10bcd
TRUE_STATE_VALUES = begin
    env = RandomWalk1D(N=NS, actions=ACTIONS)
    agent = Agent(
        policy=VBasedPolicy(
            learner=TDLearner(
                approximator=TabularVApproximator(;n_state=NS,opt=Descent(0.01)),
                method=:SRS,
                ),
            mapping=(env,V) -> rand(action_space(env))
            ),
        trajectory=VectorSARTTrajectory()
    )
    run(agent, env, StopAfterEpisode(10^5))
    agent.policy.learner.approximator.table
end

# ╔═╡ df244c4e-5c6d-11eb-23e2-bf718f61180e
plot(TRUE_STATE_VALUES[2:end-1])

# ╔═╡ e0f292ce-5c6d-11eb-0fc8-7fd0cb86d6b5
md"""
Next, we define a preprocessor to map adjacent states into groups.
"""

# ╔═╡ 2e40e8dc-5c6e-11eb-264a-f99d9f5d62e0
N_GROUPS = 10

# ╔═╡ 248d425e-5c6e-11eb-2b34-fff856ca079c
begin
	Base.@kwdef struct GroupMapping
		n::Int
		n_groups::Int = N_GROUPS
		n_per_group::Int=div(n, N_GROUPS)
	end
	function (p::GroupMapping)(x::Int)
		if x == 1
			res = 1
		elseif x == p.n
			res = p.n_groups + 2
		else
			res = div(x - 2, p.n_per_group) + 2
		end
		res
	end
end

# ╔═╡ 484bb1a8-5c6e-11eb-3aaa-ebb84d7c8785
plot([GroupMapping(;n=NS)(i) for i in 1:NS], legend=nothing)

# ╔═╡ be5e114c-5c6e-11eb-007d-8b22a431ee4f
md"""
To count the frequency of each state, we need to write a hook.
"""

# ╔═╡ ed416b12-5c6e-11eb-1d57-b3d730907456
begin
	struct CountStates <: AbstractHook
		counts::Vector{Int}
		CountStates(n) = new(zeros(Int, n))
	end
	(f::CountStates)(::PreActStage, agent, env, action) = f.counts[state(env.env)] += 1
end

# ╔═╡ 091cad38-5c6f-11eb-3f4e-893c28972111
md"""
Now let's kickoff our experiment:
"""

# ╔═╡ 11cf3a2c-5c6f-11eb-231a-99660c29621e
agent_1 = Agent(
    policy=VBasedPolicy(
        learner=MonteCarloLearner(
            approximator=TabularVApproximator(n_state=N_GROUPS+2,opt=Descent(2e-5)),
            kind=EVERY_VISIT,  # this is very important!
            ),
        mapping=(env,V) -> rand(action_space(env))
        ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ 198327e2-5c6f-11eb-26c4-19272b3374ae
env_1 = StateTransformedEnv(
    RandomWalk1D(N=NS, actions=ACTIONS),
    state_mapping=GroupMapping(n=NS)
)

# ╔═╡ 248e79a2-5c6f-11eb-30f5-9d417d286927
hook=CountStates(NS)

# ╔═╡ 2dcedb7e-5c6f-11eb-2026-abb9b378ee89
run(agent_1, env_1, StopAfterEpisode(10^5),hook)

# ╔═╡ 3fa64b5a-5c6f-11eb-2421-799d630e5460
begin
	fig_9_1 = plot(legend=:topleft, ylabel="Value scale", xlabel="State", right_margin = 1.5cm)
	fig_9_1_right = twinx(fig_9_1)
	plot!(fig_9_1, hook.counts./sum(hook.counts), color=:gray, label="state distribution")
	plot!(fig_9_1_right, agent_1.policy.learner.approximator.(env_1.state_mapping(s) for s in 2:NS-1), label="MC Learner", legend=:bottomright)
	plot!(fig_9_1_right, TRUE_STATE_VALUES[2:end-1], label="true values",legend=:bottomright, ylabel="Distribution scale")
end

# ╔═╡ 23060d86-5c70-11eb-2faa-a3851e3b5d2f
md"""
## Figure 9.2
"""

# ╔═╡ b19224a0-5c71-11eb-0582-2337e78a5ea9
agent_2 = Agent(
    policy=VBasedPolicy(
        learner=TDLearner(
            approximator=TabularVApproximator(n_state=N_GROUPS+2,opt=Descent(2e-4)),
            method=:SRS,
            ),
        mapping=(env,V) -> rand(action_space(env))
        ),
    trajectory=VectorSARTTrajectory()
)

# ╔═╡ e4ac979e-5c71-11eb-2ef3-8b5d88e38091
run(agent_2, env_1, StopAfterEpisode(10^5))

# ╔═╡ f12833c0-5c71-11eb-0c30-dd7f5c670a66
begin
	fig_9_2_left = plot(legend=:bottomright, xlabel="State")
	plot!(fig_9_2_left, agent_2.policy.learner.approximator.(env_1.state_mapping(s) for s in 2:NS-1), label="TD Learner", legend=:bottomright)
	plot!(fig_9_2_left, TRUE_STATE_VALUES[2:end-1], label="true values",legend=:bottomright)
	fig_9_2_left
end

# ╔═╡ c0afcbc6-5c72-11eb-3d31-69434487eda7
md"""
### Figure 9.2 right
"""

# ╔═╡ b797372c-5c72-11eb-3f71-5d57efd29d38
begin
	struct RecordRMS <: AbstractHook
		rms::Vector{Float64}
		RecordRMS() = new([])
	end
	function (f::RecordRMS)(::PostEpisodeStage, agent, env)
		push!(f.rms, sqrt(mean((agent.policy.learner.approximator.(env.state_mapping.(2:(NS-1))) - TRUE_STATE_VALUES[2:end-1]).^2)))
	end
end

# ╔═╡ 42af660e-5c73-11eb-3b38-bf3bb6bc6800
n_groups = 20

# ╔═╡ 804c0536-5c72-11eb-2294-93bb2512ed7a
function run_once(n, α)
    env = StateTransformedEnv(
		RandomWalk1D(N=NS, actions=ACTIONS),
		state_mapping=GroupMapping(n=NS)
	)
    agent = Agent(
        policy=VBasedPolicy(
            learner=TDLearner(
                approximator=TabularVApproximator(;
					n_state=n_groups+2,
					opt=Descent(α)
					),
                method=:SRS,
                n=n
                ),
            mapping=(env,V) -> rand(action_space(env))
            ),
        trajectory=VectorSARTTrajectory()
    )

    hook = RecordRMS()
    run(agent, env, StopAfterEpisode(10),hook)
    mean(hook.rms)
end

# ╔═╡ 54a7601e-5c73-11eb-2372-513196c58e8f
begin
	A = [0., 0.03, 0.06, 0.1:0.1:1...]
	fig_9_2_right = plot(legend=:bottomright, ylim=[0.25,0.55])
	for n in [2^i for i in 0:9]
		plot!(
			fig_9_2_right,
			A,
			mean(
				[run_once(n, α) for α in A] 
				for _ in 1:100
				),
			label="n = $n")
	end
	fig_9_2_right
end

# ╔═╡ be968ab2-5c74-11eb-321f-0186685a329a
md"""
## Figure 9.5
"""

# ╔═╡ dfcef90a-5c74-11eb-04e1-f3320eba9bf8
begin
	struct FourierPreprocessor
		order::Int
	end
	(fp::FourierPreprocessor)(s::Number) = [cos(i * π * s) for i = 0:fp.order]
end

# ╔═╡ 7901bf16-5c75-11eb-13e5-0dd84f259c0a
begin
	struct PolynomialPreprocessor
		order::Int
	end
	(pp::PolynomialPreprocessor)(s::Number) = [s^i for i = 0:pp.order]
end

# ╔═╡ 87c528bc-5c75-11eb-2f2f-adf254afda01
function run_once_MC(preprocessor, order, α)
    env = StateTransformedEnv(
        RandomWalk1D(N=NS, actions=ACTIONS),
        state_mapping=preprocessor
    )
    agent = Agent(
        policy=VBasedPolicy(
            learner=MonteCarloLearner(
                approximator=RLZoo.LinearVApproximator(;n=order+1,opt=Descent(α)),
                kind=EVERY_VISIT,
                ),
            mapping=(env,V) -> rand(1:NA)
            ),
        trajectory=VectorSARTTrajectory(;state=Vector{Float64})
    )

    hook=RecordRMS()
    run(agent, env, StopAfterEpisode(5000;is_show_progress=false),hook)
    hook.rms
end

# ╔═╡ c52bcb44-5c74-11eb-0e2b-fbb72e8edad8
begin
	
	fig_9_5 = plot(legend=:topright)

	for order in [5, 10, 20]
	plot!(
		fig_9_5, 
		mean(
			run_once_MC(
				x -> FourierPreprocessor(order)(x/NS),
				order,
				0.00005
			)
			for _ in 1:5
		),
		label="Fourier $order", 
		linestyle=:dash
	)

	plot!(
		fig_9_5, 
		mean(
			run_once_MC(
				x -> PolynomialPreprocessor(order)(x/NS),
				order,
				0.0001
			)
			for _ in 1:5
		),
		label="Polynomial $order", 
		linestyle=:solid
	)
	end

	fig_9_5
end

# ╔═╡ 08d133a0-5c77-11eb-1fbb-ed6b8da42d9f
md"""
## Figure 9.10

Implementing the tile encoding in Julia is quite easy！😀
"""

# ╔═╡ 2ef2aa46-5c77-11eb-1eec-13ad13061214
begin
	struct Tiling{N,Tr<:AbstractRange}
		ranges::NTuple{N,Tr}
		inds::LinearIndices{N,NTuple{N,Base.OneTo{Int}}}
	end
	
	Tiling(ranges...) =Tiling(
		ranges,
		LinearIndices(Tuple(length(r) - 1 for r in ranges))
	)
end

# ╔═╡ 587ab40c-5c78-11eb-1776-2bfa1cf6f608
Base.length(t::Tiling) = reduce(*, (length(r) - 1 for r in t.ranges))

# ╔═╡ 592ac4a0-5c78-11eb-3d28-f7b178f4b94f
encode(range::AbstractRange, x) = floor(Int, div(x - range[1], step(range)) + 1)

# ╔═╡ 5c0304ee-5c78-11eb-2394-8fc17938918c
encode(t::Tiling, xs) = t.inds[CartesianIndex(Tuple(map(encode,  t.ranges, xs)))]

# ╔═╡ 3c773ea6-5c78-11eb-1a09-0f1fc560386d
t = Tiling(range(1, step=200, length=7))	

# ╔═╡ 925141b4-5c78-11eb-208c-13e289d11f66
tt = [Tiling((range(1-4*(i-1), step=200, length=7))) for i in 1:50]

# ╔═╡ 7ab5686e-5c78-11eb-1067-a3127da36994
function run_once_MC_tiling(preprocessor, α, n)
    env = StateTransformedEnv(
        RandomWalk1D(N=NS, actions=ACTIONS),
        state_mapping=preprocessor
    )
    agent = Agent(
        policy=VBasedPolicy(
            learner=MonteCarloLearner(
                approximator=RLZoo.LinearVApproximator(;n=n,opt=Descent(α)),
                kind=EVERY_VISIT,
                ),
            mapping=(env,V) -> rand(1:NA)
            ),
        trajectory=VectorSARTTrajectory(;state=Vector{Float64})
    )

    hook=RecordRMS()
    run(agent, env, StopAfterEpisode(10000;is_show_progress=true),hook)
    hook.rms
end

# ╔═╡ 8b90595a-5c78-11eb-356b-df8d22646ed1
begin
	fig_9_10 = plot(xlabel="Episodes", ylabel="RMS error")
	
	plot!(
		fig_9_10,
		run_once_MC_tiling(
			x -> sparse([encode(t, x) for t in tt], 1:50, ones(50), 7, 50) |> vec,
			1e-4/50,
			7*50
		),
		label="50 tilings"
	)
	
	plot!(
		fig_9_10,
		run_once_MC_tiling(
			x -> Flux.onehot(encode(t, x), 1:7),
			1e-4,
			7
		),
		label = "one tiling"
	)
	
	fig_9_10
end

# ╔═╡ 248e1648-5c7a-11eb-0a7f-2767d27c80b6
md"""
Feel free to make a PR if you can improve the speed of generating this figure. ❤
"""

# ╔═╡ Cell order:
# ╟─41c57942-5c6c-11eb-3c99-9345b6668a1a
# ╠═9cd5e7cc-5c6c-11eb-3485-011143087d8d
# ╟─be5a557c-5c6c-11eb-035e-39e94df41852
# ╠═6a0881f0-5c6d-11eb-143e-0196833abc05
# ╠═7ee0867c-5c6d-11eb-11b4-a7858177564f
# ╠═7aae4986-5c6d-11eb-09b0-fd883165bc72
# ╟─8fc27a60-5c6d-11eb-05ff-dbdcd106b853
# ╠═c4d3a538-5c6d-11eb-3369-9bd67cc10bcd
# ╠═df244c4e-5c6d-11eb-23e2-bf718f61180e
# ╟─e0f292ce-5c6d-11eb-0fc8-7fd0cb86d6b5
# ╠═2e40e8dc-5c6e-11eb-264a-f99d9f5d62e0
# ╠═248d425e-5c6e-11eb-2b34-fff856ca079c
# ╠═484bb1a8-5c6e-11eb-3aaa-ebb84d7c8785
# ╟─be5e114c-5c6e-11eb-007d-8b22a431ee4f
# ╠═ed416b12-5c6e-11eb-1d57-b3d730907456
# ╟─091cad38-5c6f-11eb-3f4e-893c28972111
# ╠═11cf3a2c-5c6f-11eb-231a-99660c29621e
# ╠═198327e2-5c6f-11eb-26c4-19272b3374ae
# ╠═248e79a2-5c6f-11eb-30f5-9d417d286927
# ╠═2dcedb7e-5c6f-11eb-2026-abb9b378ee89
# ╠═3fa64b5a-5c6f-11eb-2421-799d630e5460
# ╟─23060d86-5c70-11eb-2faa-a3851e3b5d2f
# ╠═b19224a0-5c71-11eb-0582-2337e78a5ea9
# ╠═e4ac979e-5c71-11eb-2ef3-8b5d88e38091
# ╠═f12833c0-5c71-11eb-0c30-dd7f5c670a66
# ╟─c0afcbc6-5c72-11eb-3d31-69434487eda7
# ╠═b797372c-5c72-11eb-3f71-5d57efd29d38
# ╠═42af660e-5c73-11eb-3b38-bf3bb6bc6800
# ╠═804c0536-5c72-11eb-2294-93bb2512ed7a
# ╠═54a7601e-5c73-11eb-2372-513196c58e8f
# ╟─be968ab2-5c74-11eb-321f-0186685a329a
# ╠═dfcef90a-5c74-11eb-04e1-f3320eba9bf8
# ╠═7901bf16-5c75-11eb-13e5-0dd84f259c0a
# ╠═87c528bc-5c75-11eb-2f2f-adf254afda01
# ╠═c52bcb44-5c74-11eb-0e2b-fbb72e8edad8
# ╟─08d133a0-5c77-11eb-1fbb-ed6b8da42d9f
# ╠═2ef2aa46-5c77-11eb-1eec-13ad13061214
# ╠═587ab40c-5c78-11eb-1776-2bfa1cf6f608
# ╠═592ac4a0-5c78-11eb-3d28-f7b178f4b94f
# ╠═5c0304ee-5c78-11eb-2394-8fc17938918c
# ╠═3c773ea6-5c78-11eb-1a09-0f1fc560386d
# ╠═925141b4-5c78-11eb-208c-13e289d11f66
# ╠═7ab5686e-5c78-11eb-1067-a3127da36994
# ╠═8b90595a-5c78-11eb-356b-df8d22646ed1
# ╟─248e1648-5c7a-11eb-0a7f-2767d27c80b6
