### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 604afcfc-5d9d-11eb-0e2d-4971e8c87824
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
	using Distributions
end

# ╔═╡ be402764-5e02-11eb-0978-bb9f65076f60
md"""
Again, we use a environment model to describe the **Grid World** in **Chapter 4.2**.
"""

# ╔═╡ f2de50d6-5e02-11eb-0c19-2756e3a8fe68
begin
	isterminal(s::CartesianIndex{2}) = s == CartesianIndex(1,1) || s == CartesianIndex(4,4) 

	function nextstep(s::CartesianIndex{2}, a::CartesianIndex{2})
		s′ = s + a
		if isterminal(s) || s′[1] < 1 || s′[1] > 4 || s′[2] < 1 || s′[2] > 4
			s′ = s
		end
		r = isterminal(s) ? 0. : -1.0
		[(r, isterminal(s′), LinearIndices((4,4))[s′]) => 1.0]
	end

	const ACTIONS = [
		CartesianIndex(-1, 0),
		CartesianIndex(1,0),
		CartesianIndex(0, 1),
		CartesianIndex(0, -1)
	]

	struct GridWorldEnvModel <: AbstractEnvironmentModel
		cache
	end
	
	GridWorldEnvModel() = GridWorldEnvModel(
		Dict(
			(s, a) => nextstep(CartesianIndices((4,4))[s], ACTIONS[a])
			for s in 1:16 for a in 1:4
		)
	)
	
	(m::GridWorldEnvModel)(s, a) = m.cache[(s,a)]
	
	RLBase.state_space(m::GridWorldEnvModel) = Base.OneTo(16)
	RLBase.action_space(m::GridWorldEnvModel) = Base.OneTo(4)
end

# ╔═╡ f02b3240-5e03-11eb-3009-8fa372d61d3d
V = TabularVApproximator(n_state=16, opt=Descent(1.0))

# ╔═╡ fe9a730e-5e03-11eb-26e2-8b94166c0332
p = TabularRandomPolicy(table=Dict(s => fill(0.25, 4) for s in 1:16))

# ╔═╡ f27da2fa-5e04-11eb-268f-45865581a8c9
model = GridWorldEnvModel()

# ╔═╡ 4b8ecf0c-5e04-11eb-38a1-839f6cb37e78
policy_evaluation!(V=V, π=p, model=model, γ=1.0)

# ╔═╡ 6484f504-5e04-11eb-1286-9ba6803d9f2b
reshape(V.table, 4,4)

# ╔═╡ Cell order:
# ╠═604afcfc-5d9d-11eb-0e2d-4971e8c87824
# ╟─be402764-5e02-11eb-0978-bb9f65076f60
# ╠═f2de50d6-5e02-11eb-0c19-2756e3a8fe68
# ╠═f02b3240-5e03-11eb-3009-8fa372d61d3d
# ╠═fe9a730e-5e03-11eb-26e2-8b94166c0332
# ╠═f27da2fa-5e04-11eb-268f-45865581a8c9
# ╠═4b8ecf0c-5e04-11eb-38a1-839f6cb37e78
# ╠═6484f504-5e04-11eb-1286-9ba6803d9f2b
