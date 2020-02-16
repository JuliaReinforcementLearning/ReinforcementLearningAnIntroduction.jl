module ReinforcementLearningAnIntroduction

using Reexport

const RLIntro = ReinforcementLearningAnIntroduction
export RLIntro

@reexport using ReinforcementLearningBase
@reexport using ReinforcementLearningCore

include("environments/environments.jl")
include("extensions/extensions.jl")

end # module
