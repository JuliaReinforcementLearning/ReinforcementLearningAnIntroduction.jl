export DeterministicDistributionModel

"""
    DeterministicDistributionModel(table::Array{Vector{NamedTuple{(:nextstate, :reward, :prob),Tuple{Int,Float64,Float64}}}, 2})

Store all the transformations in the `table` field.
"""
struct DeterministicDistributionModel <: AbstractEnvironmentModel
    table::Array{
        Vector{NamedTuple{(:nextstate, :reward, :prob),Tuple{Int,Float64,Float64}}},
        2,
    }
end

RLBase.get_observation_space(m::DeterministicDistributionModel) = DiscreteSpace(size(m.table, 1))
RLBase.get_action_space(m::DeterministicDistributionModel) = DiscreteSpace(size(m.table, 2))

(m::DeterministicDistributionModel)(s::Int, a::Int) = m.table[s, a]