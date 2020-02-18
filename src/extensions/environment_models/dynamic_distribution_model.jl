export DynamicDistributionModel

"""
    DynamicDistributionModel(f::Tf, ns::Int, na::Int) -> DynamicDistributionModel{Tf}

Use a general function `f` to store the transformations. `ns` and `na` are the number of states and actions.
"""
struct DynamicDistributionModel{Tf<:Function} <: AbstractEnvironmentModel
    f::Tf
    ns::Int
    na::Int
end

RLBase.get_observation_space(m::DynamicDistributionModel) = DiscreteSpace(m.ns)
RLBase.get_action_space(m::DynamicDistributionModel) = DiscreteSpace(m.na)

(m::DynamicDistributionModel)(s, a) = m.f(s, a)