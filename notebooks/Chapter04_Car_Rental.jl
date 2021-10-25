### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 92081fb8-5d90-11eb-2078-ddbf87421051
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
	using Distributions
end

# ╔═╡ 3254385c-5d90-11eb-2c99-07a3a40ae467
md"""
Again, we'll describe the car rental problem with a distributional model.
"""

# ╔═╡ a1a5176c-5d90-11eb-2ad6-2f108e1531e2
begin
	const PoissonUpperBound = 10
	const MaxCars= 20
	const MaxMoves = 5
	const CostPerMove = 2
	const CarRentalCartesianIndices = CartesianIndices((0:MaxCars,0:MaxCars))
	const CarRentalLinearIndices = LinearIndices((0:MaxCars,0:MaxCars))
	const Actions = -MaxMoves:MaxMoves
	const RequestDist_1 = Poisson(3)
	const RequestDist_2 = Poisson(4)
	const ReturnDist_1 = Poisson(3)
	const ReturnDist_2 = Poisson(2)

	decode_state(s::Int) = Tuple(CarRentalCartesianIndices[s])
	encode_state(s1::Int, s2::Int) = CarRentalLinearIndices[CartesianIndex(s1+1, s2+1)]
	decode_action(a::Int) = a - MaxMoves - 1
	encode_action(a::Int) = a + MaxMoves + 1

	function merge_prob(dist)
		merged = Dict()
		for (s′, r, p) in dist
			if haskey(merged, (s′, r))
				merged[(s′, r)] += p
			else
				merged[(s′, r)] = p
			end
		end
		[(r, false, s′) => p for ((s′, r), p) in merged]
	end

	function nextstep(state::Int, action::Int)
		(s1, s2), a = decode_state(state), decode_action(action)
		move = a > 0 ? min(a, s1) : max(a, -s2)
		reward = -CostPerMove*abs(move)
		s1′, s2′ = min(s1 - move, MaxCars), min(s2 + move, MaxCars)
		merge_prob(
			(
				encode_state(
					min(max(s1′-req_1, 0)+ret_1, MaxCars),
					min(max(s2′-req_2, 0)+ret_2, MaxCars)
					), 
				reward + (min(s1′, req_1) + min(s2′, req_2)) * 10,
				(
					pdf(RequestDist_1, req_1) * 
					pdf(RequestDist_2, req_2) * 
					pdf(ReturnDist_1, ret_1) *
					pdf(ReturnDist_2, ret_2)
				)
			)
			for req_1 in 0:PoissonUpperBound,
				req_2 in 0:PoissonUpperBound, 
				ret_1 in 0:PoissonUpperBound,
				ret_2 in 0:PoissonUpperBound
		)
	end

	struct CarRentalEnvModel <: AbstractEnvironmentModel
		dist
	end

	function CarRentalEnvModel()
		CarRentalEnvModel(
			Dict(
				(s,a) => nextstep(s,a)
				for s in 1:(MaxCars+1)^2 for a in 1:length(Actions)
			)
		)
	end

	(m::CarRentalEnvModel)(s, a) = m.dist[(s, a)]

	RLBase.state_space(m::CarRentalEnvModel) = Base.OneTo((MaxCars+1)^2)
	RLBase.action_space(m::CarRentalEnvModel) = Base.OneTo(length(Actions))
end

# ╔═╡ 20e3c8d6-5d92-11eb-1591-9911e62805bd
model = CarRentalEnvModel()

# ╔═╡ 0c099578-5d92-11eb-2a8b-2f4889026185
V = TabularVApproximator(n_state=length(state_space(model)), opt=Descent(1.0))

# ╔═╡ dda27350-5d94-11eb-1a82-c302513da719
p = TabularPolicy(;table=Dict(s=>1 for s in state_space(model)), n_action=length(action_space(model)))

# ╔═╡ 021f9b90-5d95-11eb-303a-ed2fb3de3d08
policy_iteration!(;V=V, π=p, model=model, γ=0.9, max_iter=300)

# ╔═╡ 8bbe08e6-5d98-11eb-0c02-4712c4659830
heatmap(0:MaxCars, 0:MaxCars, reshape([decode_action(p(x)) for x in state_space(model)], 1+MaxCars,1+MaxCars))

# ╔═╡ 91d90bf6-5d98-11eb-23a5-4dfc2839a707
heatmap(0:MaxCars, 0:MaxCars, reshape(V.table, 1+MaxCars,1+MaxCars))

# ╔═╡ Cell order:
# ╟─3254385c-5d90-11eb-2c99-07a3a40ae467
# ╠═92081fb8-5d90-11eb-2078-ddbf87421051
# ╠═a1a5176c-5d90-11eb-2ad6-2f108e1531e2
# ╠═20e3c8d6-5d92-11eb-1591-9911e62805bd
# ╠═0c099578-5d92-11eb-2a8b-2f4889026185
# ╠═dda27350-5d94-11eb-1a82-c302513da719
# ╠═021f9b90-5d95-11eb-303a-ed2fb3de3d08
# ╠═8bbe08e6-5d98-11eb-0c02-4712c4659830
# ╠═91d90bf6-5d98-11eb-23a5-4dfc2839a707
