export TimeBasedSampleModel

import StatsBase: sample

"""
    TimeBasedSampleModel(nactions::Int, κ::Float64 = 1e-4)
"""
mutable struct TimeBasedSampleModel <: AbstractEnvironmentModel
    experiences::Dict{
        Any,
        Dict{Any,NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}}},
    }
    nactions::Int
    κ::Float64
    t::Int
    last_visit::Dict{Tuple{Any,Any},Int}
    TimeBasedSampleModel(nactions::Int, κ::Float64 = 1e-4) =
        new(
            Dict{
                Any,
                Dict{
                    Any,
                    NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}},
                },
            }(),
            nactions,
            κ,
            0,
            Dict{Tuple{Any,Any},Int}(),
        )
end

function RLBase.extract_experience(t::AbstractTrajectory, m::TimeBasedSampleModel)
    if length(t) > 0
        get_trace(t, :state)[end],
        get_trace(t, :action)[end],
        get_trace(t, :reward)[end],
        get_trace(t, :terminal)[end],
        get_trace(t, :next_state)[end]
    else
        nothing
    end
end

function RLBase.update!(m::TimeBasedSampleModel, transition::Tuple)
    s, a, r, d, s′ = transition
    if haskey(m.experiences, s)
        m.experiences[s][a] = (reward = r, terminal = d, nextstate = s′)
    else
        m.experiences[s] = Dict{
            Any,
            NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}},
        }(a => (reward = r, terminal = d, nextstate = s′))
    end
    m.t += 1
    m.last_visit[(s, a)] = m.t
end

function sample(m::TimeBasedSampleModel)
    s = rand(keys(m.experiences))
    a = rand(1:m.nactions)
    r, d, s′ = get(m.experiences[s], a, (0.0, false, s))
    r += m.κ * sqrt(m.t - get(m.last_visit, (s, a), 0))
    s, a, r, d, s′
end
