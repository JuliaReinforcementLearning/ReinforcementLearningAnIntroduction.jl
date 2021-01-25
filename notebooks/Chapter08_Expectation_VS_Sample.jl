### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ d80342ae-522b-11eb-19c0-cf931fb4ae81
using Statistics

# ╔═╡ 9c2ef32a-522d-11eb-29e6-29da29e4dec7
using Plots

# ╔═╡ 88b47ee6-522d-11eb-1a98-ed4dd43dfd11
Base.@kwdef mutable struct SampleAvg
    t::Int = 0
    avg::Float64 = 0.0
end

# ╔═╡ 8c94a810-522d-11eb-2024-01ca19bc1984
function (s::SampleAvg)(x)
    s.t += 1
    s.avg += (x - s.avg) / s.t
    s.avg
end

# ╔═╡ 8fde026e-522d-11eb-2aa1-f3e28c238ea6
function run_once(b)
    rms = Float64[]
    distribution = randn(b)
    expectation = mean(distribution)
    sample_avg = SampleAvg()
    
    for i in 1:2*b
        avg = sample_avg(distribution[rand(1:b)])
        push!(rms, abs(avg - expectation))
    end
    rms
end

# ╔═╡ 969b48aa-522d-11eb-2039-61916ddf87d4
begin
	n_runs = 1000
	p = plot(legend=:topright)

	for b in [2, 10, 100, 1000]
		rms = mean(run_once(b) for _ in 1:n_runs)
		xs = (1:2*b) ./ b
		plot!(p, xs, rms, label="b=$b")
	end

	p
end

# ╔═╡ Cell order:
# ╠═d80342ae-522b-11eb-19c0-cf931fb4ae81
# ╠═88b47ee6-522d-11eb-1a98-ed4dd43dfd11
# ╠═8c94a810-522d-11eb-2024-01ca19bc1984
# ╠═8fde026e-522d-11eb-2aa1-f3e28c238ea6
# ╠═9c2ef32a-522d-11eb-29e6-29da29e4dec7
# ╠═969b48aa-522d-11eb-2039-61916ddf87d4
