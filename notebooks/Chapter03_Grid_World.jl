### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ bd94028c-5d8e-11eb-22b6-5fa384999fdb
begin
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
end

# ╔═╡ fbf4f518-5d8e-11eb-1611-2b658a468f57
md"""
To describe the **Grid World** in **Example 3.5**, we'll create a distributional environment model. Here the *distributional* means, given a state-action paire, we can predict the possible next state, reward, termination info and the corresponding probability.
"""

# ╔═╡ 5ff21d54-5d8f-11eb-0e8f-95ca0eabc01f
begin
	function nextstep(s::CartesianIndex{2}, a::CartesianIndex{2})
		if s == CartesianIndex(1, 2)
			r, s′ =  10., CartesianIndex(5, 2)
		elseif s == CartesianIndex(1, 4)
			r, s′ = 5., CartesianIndex(3, 4)
		else
			s′ = s + a
			if 1 ≤ s′[1] ≤ 5 && 1 ≤ s′[2] ≤ 5
				r = 0.
			else
				r = -1.
				s′ = s
			end
		end
		[(r, false, LinearIndices((5,5))[s′]) => 1.0]
	end

	ACTIONS = (
		CartesianIndex(-1, 0),
		CartesianIndex(1,0),
		CartesianIndex(0, 1),
		CartesianIndex(0, -1)
	)

	struct GridWorldModel <: AbstractEnvironmentModel
	end

	function (m::GridWorldModel)(s, a)
		nextstep(CartesianIndices((5,5))[s], ACTIONS[a])
	end

	RLBase.state_space(m::GridWorldModel) = Base.OneTo(5*5)
	RLBase.action_space(m::GridWorldModel) = Base.OneTo(length(ACTIONS))
end

# ╔═╡ 81bcb7a8-5d8f-11eb-070b-97e0199e6c4e
V = TabularVApproximator(;n_state=25, opt=Descent(1.0))

# ╔═╡ 86d47256-5d8f-11eb-27ae-ffa516974fe2
policy_evaluation!(
    V = V,
    π=RandomPolicy(Base.OneTo(4)),
    model=GridWorldModel(),
    γ=0.9
)

# ╔═╡ 8b4d1bc8-5d8f-11eb-1e5b-ddbaa49c216b
table = reshape(V.table, 5, 5)

# ╔═╡ 90c88768-5d8f-11eb-2a17-0dcdf7e0772a
heatmap(table,yflip=true)

# ╔═╡ Cell order:
# ╠═bd94028c-5d8e-11eb-22b6-5fa384999fdb
# ╟─fbf4f518-5d8e-11eb-1611-2b658a468f57
# ╠═5ff21d54-5d8f-11eb-0e8f-95ca0eabc01f
# ╠═81bcb7a8-5d8f-11eb-070b-97e0199e6c4e
# ╠═86d47256-5d8f-11eb-27ae-ffa516974fe2
# ╠═8b4d1bc8-5d8f-11eb-1e5b-ddbaa49c216b
# ╠═90c88768-5d8f-11eb-2a17-0dcdf7e0772a
