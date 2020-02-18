export FourierPreprocessor, PolynomialPreprocessor, TilingPreprocessor

"""
    FourierPreprocessor(order::Int)
Transform a scalar to a vector of `order+1` Fourier bases.
"""
struct FourierPreprocessor <: AbstractPreprocessor
    order::Int
end

(p::FourierPreprocessor)(s::Number) = [cos(i * π * s) for i = 0:p.order]

"""
    PolynomialPreprocessor(order::Int)
Transform a scalar to vector of maximum `order` polynomial.
"""
struct PolynomialPreprocessor <: AbstractPreprocessor
    order::Int
end

(p::PolynomialPreprocessor)(s::Number) = [s^i for i = 0:p.order]

"""
    TilingPreprocessor(tilings::Vector{<:Tiling})
Use each `tilings` to encode the state and return a vector.
"""
struct TilingPreprocessor{Tt<:Tiling} <: AbstractPreprocessor
    tilings::Vector{Tt}
end

(p::TilingPreprocessor)(s::Union{<:Number,<:Array}) = [encode(t, s) for t in p.tilings]
