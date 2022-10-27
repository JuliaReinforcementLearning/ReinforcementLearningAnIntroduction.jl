### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 973169b2-5cb5-11eb-2c27-8115e5bfec1f
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
end

# ╔═╡ 662d102a-5cb6-11eb-1f92-1b7f561b048a
md"""
## The Baird Count Environment
"""

# ╔═╡ 11ba9900-5cb5-11eb-0d89-0f51f890b091
begin
	const DASH_SOLID = (:dashed, :solid)

	Base.@kwdef mutable struct BairdCounterEnv <: AbstractEnv
		current::Int = rand(1:7)
	end

	RLBase.state_space(env::BairdCounterEnv) = Base.OneTo(7)
	RLBase.action_space(env::BairdCounterEnv) = Base.OneTo(length(DASH_SOLID))

	function (env::BairdCounterEnv)(a)
		if DASH_SOLID[a] == :dashed
			env.current = rand(1:6)
		else
			env.current = 7
		end
		nothing
	end

	RLBase.reward(env::BairdCounterEnv) = 0.
	RLBase.is_terminated(env::BairdCounterEnv) = false
	RLBase.state(env::BairdCounterEnv) = env.current
	RLBase.reset!(env::BairdCounterEnv) = env.current = rand(1:6)
end

# ╔═╡ 915ff55a-5cb6-11eb-09bc-2f489a33a737
md"""
## Off Policy
"""

# ╔═╡ b5711c62-5cb6-11eb-2b70-91051ff959d3
# Base.@kwdef struct OffPolicy{P,B} <: AbstractPolicy
#     π_target::P
#     π_behavior::B
# end

# ╔═╡ ba6d1efc-5cb6-11eb-28e5-f73f80f44ad7
# (π::OffPolicy)(env) = π.π_behavior(env)

# ╔═╡ 447be6a0-5cbf-11eb-2daf-9b1b54e403f1
begin
	
	# const VectorWSARTTrajectory = Trajectory{<:NamedTuple{(:weight, SART...)}}

	# function VectorWSARTTrajectory(;weight=Float64, state=Int, action=Int, reward=Float32, terminal=Bool)
	# 	VectorTrajectory(;weight=Float64, state=state, action=action, reward=reward, terminal=terminal)
	# end
	
	# function RLBase.update!(
	# 	p::OffPolicy,
	# 	t::VectorWSARTTrajectory,
	# 	e::AbstractEnv,
	# 	s::AbstractStage
	# )
	# 	update!(p.π_target, t, e, s)
	# end

	# function RLBase.update!(
	# 	t::VectorWSARTTrajectory,
	# 	p::OffPolicy,
	# 	env::AbstractEnv,
	# 	s::PreActStage,
	# 	a
	# )
	# 	push!(t[:state], state(env))
	# 	push!(t[:action], a)

	# 	w = prob(p.π_target, s, a) / prob(p.π_behavior, s, a)
	# 	push!(t[:weight], w)
	# end

	# function RLBase.update!(
	# 	t::VectorWSARTTrajectory,
	# 	p::OffPolicy{<:QBasedPolicy{<:TDLearner}},
	# 	env::AbstractEnv,
	# 	s::PreEpisodeStage,
	# )
	# 	empty!(t)
	# end

	# function RLBase.update!(
	# 	t::VectorWSARTTrajectory,
	# 	p::OffPolicy{<:QBasedPolicy{<:TDLearner}},
	# 	env::AbstractEnv,
	# 	s::PostEpisodeStage,
	# )
	# 	action = rand(action_space(env))

	# 	push!(trajectory[:state], state(env))
	# 	push!(trajectory[:action], action)
	# 	push!(t[:weight], 1.0)
	# end
end

# ╔═╡ f01b15c2-5cb6-11eb-1d41-21f632c01a2f
md"""
## Figure 11.2
"""

# ╔═╡ 45c5fdfe-5cb8-11eb-115e-93fcb138261d
world = BairdCounterEnv()

# ╔═╡ c9f52da8-5cb5-11eb-35eb-ad54e81df746
begin
	Base.@kwdef struct RecordWeights <: AbstractHook
		weights::Vector{Vector{Float64}}=[]
	end

	(h::RecordWeights)(::PostActStage, agent, env) = push!(
		h.weights,
		agent.policy.π_target.learner.approximator.weights |> deepcopy
	)
end

# ╔═╡ 30e3d956-5cb8-11eb-05be-ab2d4a182b4e
NW = 8

# ╔═╡ 6396bcb2-5cb8-11eb-2c4a-dd32adf35da6
INIT_WEIGHT = ones(8)

# ╔═╡ 7f294128-5cb8-11eb-3b3f-8b74678e407f
INIT_WEIGHT[7] = 10

# ╔═╡ 5ed1b616-5cb6-11eb-3678-f9795f222f8a
STATE_MAPPING = zeros(NW, length(state_space(world)))

# ╔═╡ 9df6e892-5cb8-11eb-28d3-098d7aff3db0
begin
	for i in 1:6
		STATE_MAPPING[i, i] = 2
		STATE_MAPPING[8, i] = 1
	end
	STATE_MAPPING[7, 7] = 1
	STATE_MAPPING[8, 7] = 2
end

# ╔═╡ bc09d846-5cb8-11eb-0638-7d7a435e5fc4
STATE_MAPPING

# ╔═╡ bff74902-5cb8-11eb-2736-638fcd9a1b5b
π_b = x -> rand() < 6/7 ? 1 : 2

# ╔═╡ 181f9792-5cb9-11eb-2575-6995ea64416d
π_t = VBasedPolicy(
	learner=TDLearner(
		approximator=RLZoo.LinearApproximator(INIT_WEIGHT, Descent(0.01)),
		γ=0.99,
		n=0,
		method=:SRS
	),
	mapping = (env, V) -> 2
)

# ╔═╡ 7ea3658e-5cb9-11eb-297c-1561db9958eb
prob_b = [6/7, 1/7]

# ╔═╡ 9ba05ebc-5cb9-11eb-1a41-cbf495eedbe5
prob_t = [0., 1.]

# ╔═╡ aa855324-5cb9-11eb-1d44-8b3584e93983
md"""
Well, I must admit it is a little tricky here.
"""

# ╔═╡ a1b4b438-5cb9-11eb-0017-4594a2929891
RLBase.prob(::typeof(π_b), s, a::Integer) = prob_b[a]

# ╔═╡ bc514e16-5cb9-11eb-0452-87f09f1bca1a
RLBase.prob(::typeof(π_t), s, a::Integer) = prob_t[a]

# ╔═╡ c165ae40-5cb9-11eb-197c-e10a30939e4e
agent = Agent(
    policy=OffPolicy(
        π_target=π_t,
        π_behavior=π_b
    ),
    trajectory=VectorWSARTTrajectory(state=Any)
)

# ╔═╡ 76cc56d0-5cba-11eb-34b2-771d4a8cb6c6
new_env = StateTransformedEnv(
    BairdCounterEnv(),
    ;state_mapping=s -> @view STATE_MAPPING[:, s]
)

# ╔═╡ a7a82a90-5cba-11eb-0743-2364ec991a79
hook = RecordWeights()

# ╔═╡ b0f5f690-5cba-11eb-1480-b300a5d8aa7e
run(agent, new_env, StopAfterStep(1000),hook)

# ╔═╡ c0f640ea-5cba-11eb-13da-27969e97d064
begin
	p = plot(legend=:topleft)
	for i in 1:length(INIT_WEIGHT)
		plot!(p, [w[i] for w in hook.weights])
	end
	p
end

# ╔═╡ Cell order:
# ╠═973169b2-5cb5-11eb-2c27-8115e5bfec1f
# ╠═662d102a-5cb6-11eb-1f92-1b7f561b048a
# ╠═11ba9900-5cb5-11eb-0d89-0f51f890b091
# ╟─915ff55a-5cb6-11eb-09bc-2f489a33a737
# ╠═b5711c62-5cb6-11eb-2b70-91051ff959d3
# ╠═ba6d1efc-5cb6-11eb-28e5-f73f80f44ad7
# ╠═447be6a0-5cbf-11eb-2daf-9b1b54e403f1
# ╟─f01b15c2-5cb6-11eb-1d41-21f632c01a2f
# ╠═45c5fdfe-5cb8-11eb-115e-93fcb138261d
# ╠═c9f52da8-5cb5-11eb-35eb-ad54e81df746
# ╠═30e3d956-5cb8-11eb-05be-ab2d4a182b4e
# ╠═6396bcb2-5cb8-11eb-2c4a-dd32adf35da6
# ╠═7f294128-5cb8-11eb-3b3f-8b74678e407f
# ╠═5ed1b616-5cb6-11eb-3678-f9795f222f8a
# ╠═9df6e892-5cb8-11eb-28d3-098d7aff3db0
# ╠═bc09d846-5cb8-11eb-0638-7d7a435e5fc4
# ╠═bff74902-5cb8-11eb-2736-638fcd9a1b5b
# ╠═181f9792-5cb9-11eb-2575-6995ea64416d
# ╠═7ea3658e-5cb9-11eb-297c-1561db9958eb
# ╠═9ba05ebc-5cb9-11eb-1a41-cbf495eedbe5
# ╟─aa855324-5cb9-11eb-1d44-8b3584e93983
# ╠═a1b4b438-5cb9-11eb-0017-4594a2929891
# ╠═bc514e16-5cb9-11eb-0452-87f09f1bca1a
# ╠═c165ae40-5cb9-11eb-197c-e10a30939e4e
# ╠═76cc56d0-5cba-11eb-34b2-771d4a8cb6c6
# ╠═a7a82a90-5cba-11eb-0743-2364ec991a79
# ╠═b0f5f690-5cba-11eb-1480-b300a5d8aa7e
# ╠═c0f640ea-5cba-11eb-13da-27969e97d064
