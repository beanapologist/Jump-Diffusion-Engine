# Jump-Diffusion-Engine
Universal SDE jump diffusion engine for dissipative systems. Turn chaos into stable oscil order. 

## Core Equation

dΔ = [Λ(t) − f(Δ)] dt  +  σ dW  +  J dN

Λ(t) = ε₀ + A·sin(ωt)        
the source, steady and singing
f(Δ) = kΔ + gΔ²/(K²+Δ²)      
the sink, linear then saturating
Δ*   : Λ = f(Δ),  f′(Δ*) > 0  
the bowl, the stable held place



