export GradientBanditLearner

using Flux: softmax, onehot

"""
    GradientBanditLearner(;approximator::A, optimizer::O, baseline::B)
"""
Base.@kwdef mutable struct GradientBanditLearner{A,O,B} <: AbstractLearner
    approximator::A
    optimizer::O
    baseline::B
end

(learner::GradientBanditLearner)(s::Int) = s |> learner.approximator |> softmax
(learner::GradientBanditLearner)(obs) = learner(get_state(obs))

RLBase.update!(learner::GradientBanditLearner, ::Nothing) = nothing

function RLBase.update!(learner::GradientBanditLearner, experience::Tuple)
    s, a, r = experience
    probs = s |> learner.approximator |> softmax
    r̄ = learner.baseline isa Number ? learner.baseline : learner.baseline(r)
    errors = (r - r̄) .* (onehot(a, 1:length(probs)) .- probs)
    update!(learner.approximator, s => apply!(learner.optimizer, s, errors))
end

function RLCore.extract_experience(t::AbstractTrajectory, ::GradientBanditLearner)
    if length(t) > 0
        (
            get_trace(t, :state)[end],
            get_trace(t, :action)[end],
            get_trace(t, :reward)[end],
        )
    else
        nothing
    end
end