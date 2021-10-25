### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ f5cc0f04-5d99-11eb-3abe-bf3fccdac9e6
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using ReinforcementLearning
	using Flux
	using Statistics
	using Plots
	using Distributions
end

# ╔═╡ 28cb401c-5d9a-11eb-2098-1939ea55d918
begin
	const pₕ = 0.4
	const WinCapital = 100

	decode_state(s::Int) = s - 1
	encode_state(s::Int) = s + 1

	function nextstep(s::Int, a::Int)
		s = decode_state(s)
		a = min(s, a)
		if s == WinCapital || s==0
			[(0., false,encode_state(s))=>1.0]
		else
			[
				((s+a >= WinCapital ? 1.0 : 0.), false, encode_state(min(s+a, WinCapital))) => pₕ,
			 	(0., false, encode_state(max(s-a, 0))) => 1-pₕ
			]
		end
	end

	struct GamblerProblemEnvModel <: AbstractEnvironmentModel
		cache
	end
	
	function GamblerProblemEnvModel()
		GamblerProblemEnvModel(
			Dict(
				(s,a) => nextstep(s,a)
				for s in 1:(WinCapital+1) for a in 1:WinCapital
			)
		)
	end
	
	RLBase.state_space(m::GamblerProblemEnvModel) = Base.OneTo(WinCapital+1)
	RLBase.action_space(m::GamblerProblemEnvModel) = Base.OneTo(WinCapital)
	
	(m::GamblerProblemEnvModel)(s, a) = m.cache[(s,a)]
end

# ╔═╡ 7cfd0a76-5d9c-11eb-35dc-176affb90ad3
V = TabularVApproximator(;n_state=1+WinCapital,opt=Descent(1))

# ╔═╡ bcede678-5d9c-11eb-05d0-fb25818336cf
RLZoo.value_iteration!(V=V, model=GamblerProblemEnvModel(), γ=1.0, max_iter=1000)

# ╔═╡ 005c156e-5d9e-11eb-11be-1b69be12b13b
plot(V.table[2:end-1])

# ╔═╡ Cell order:
# ╠═f5cc0f04-5d99-11eb-3abe-bf3fccdac9e6
# ╠═28cb401c-5d9a-11eb-2098-1939ea55d918
# ╠═7cfd0a76-5d9c-11eb-35dc-176affb90ad3
# ╠═bcede678-5d9c-11eb-05d0-fb25818336cf
# ╠═005c156e-5d9e-11eb-11be-1b69be12b13b
