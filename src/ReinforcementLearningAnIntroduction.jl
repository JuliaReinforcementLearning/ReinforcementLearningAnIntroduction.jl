module ReinforcementLearningAnIntroduction

using Reexport

const RLIntro = ReinforcementLearningAnIntroduction
export RLIntro

@reexport using ReinforcementLearningBase
@reexport using ReinforcementLearningCore

# !!! just a quick and dirty fix for
# https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl/issues/17
function RLBase.update!(model::AbstractEnvironmentModel, buffer::AbstractTrajectory)
    transitions = extract_experience(buffer, model)
    isnothing(transitions) || update!(model, transitions)
end

include("environments/environments.jl")
include("extensions/extensions.jl")

end # module
