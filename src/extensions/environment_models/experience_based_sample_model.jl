export ExperienceBasedSampleModel, sample

import StatsBase: sample

"""
    ExperienceBasedSampleModel() -> ExperienceBasedSampleModel

Generate a transition based on previous experiences.
"""
mutable struct ExperienceBasedSampleModel <: AbstractEnvironmentModel
    experiences::Dict{
        Any,
        Dict{Any,NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}}},
    }
    sample_count::Int
    ExperienceBasedSampleModel() =
        new(
            Dict{
                Any,
                Dict{
                    Any,
                    NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}},
                },
            }(),
            0,
        )
end

function RLBase.extract_experience(t::AbstractTrajectory, m::ExperienceBasedSampleModel)
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

RLBase.update!(m::ExperienceBasedSampleModel, ::Nothing) = nothing

function RLBase.update!(m::ExperienceBasedSampleModel, transition::Tuple)
    s, a, r, d, s′ = transition
    if haskey(m.experiences, s)
        m.experiences[s][a] = (reward = r, terminal = d, nextstate = s′)
    else
        m.experiences[s] = Dict{
            Any,
            NamedTuple{(:reward, :terminal, :nextstate),Tuple{Float64,Bool,Any}},
        }(a => (reward = r, terminal = d, nextstate = s′))
    end
end

function sample(model::ExperienceBasedSampleModel)
    s = rand(keys(model.experiences))
    a = rand(keys(model.experiences[s]))
    model.sample_count += 1
    s, a, model.experiences[s][a]...
end
