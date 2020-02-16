export SampleAvg,
       CachedSampleAvg,
       CachedSum,
       discount_rewards,
       discount_rewards!

import StatsBase: countmap

"extend the countmap in StatsBase to support general iterator"
function countmap(iter)
    res = Dict{eltype(iter),Int}()
    for x in iter
        if haskey(res, x)
            res[x] += 1
        else
            res[x] = 1
        end
    end
    res
end

Base.@kwdef mutable struct SampleAvg
    t::Int = 0
    avg::Float64 = 0.0
end

function (s::SampleAvg)(x)
    s.t += 1
    s.avg += (x - s.avg) / s.t
    s.avg
end

Base.@kwdef struct CachedSampleAvg{T}
    cache::Dict{T,SampleAvg} = Dict{T,SampleAvg}()
end

function (c::CachedSampleAvg)(k, x)
    if !haskey(c.cache, k)
        c.cache[k] = SampleAvg()
    end
    c.cache[k](x)
end

Base.Base.@kwdef struct CachedSum{T}
    cache::Dict{T,Float64} = Dict{T,Float64}()
end

function (c::CachedSum)(k, x)
    c.cache[k] = get!(c.cache, k, 0.0) + x
end

function discount_rewards!(new_rewards, rewards, γ)
    new_rewards[end] = rewards[end]
    for i = (length(rewards) - 1):-1:1
        new_rewards[i] = rewards[i] + new_rewards[i+1] * γ
    end
    new_rewards
end

discount_rewards(rewards, γ) = discount_rewards!(similar(rewards), rewards, γ)

discount_rewards_reduced(rewards, γ) = foldr((r, g) -> r + γ * g, rewards)

