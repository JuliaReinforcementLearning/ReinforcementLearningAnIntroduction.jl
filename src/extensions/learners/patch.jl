function RLBase.update!(π::Union{AbstractLearner, OffPolicy, QBasedPolicy{<:TDLearner{<:AbstractApproximator,:ExpectedSARSA}}}, t::AbstractTrajectory)
    experience = extract_experience(t, π)
    isnothing(experience) || update!(π, experience)
end

function RLBase.update!(π::OffPolicy{<:VBasedPolicy}, transitions::NamedTuple)
    # ??? define a `get_batch_prob` function for efficiency
    weights = [
        get_prob(π.π_target, (state = s,), a) / get_prob(π.π_behavior, (state = s,), a)
        for (s, a) in zip(transitions.states, transitions.actions)
    ]  # TODO: implement iterate interface for (SubArray of) CircularArrayBuffer
    experience = merge(transitions, (weights = weights,))
    update!(π.π_target, experience)
end

extract_experience(t::AbstractTrajectory, π::OffPolicy) =
    extract_experience(t, π.π_target)

extract_experience(trajectory::AbstractTrajectory, p::VBasedPolicy) =
    extract_experience(trajectory, p.learner)