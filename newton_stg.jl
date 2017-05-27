using NLPModels

include("steihaug.jl")

function newton_stg(nlp::AbstractNLPModel; η1 = 0.25, η2 = 0.75, σ1 = 0.5, σ2 = 2.0, atol = 1e-6, rtol = 1e-4)
    k = 0
    k_max = 1000
    tempo = 0
    tempo_inicial = time()
    tempo_max = 30
    saida = 0
    x = nlp.meta.x0
    f(x) = obj(nlp, x)
    g(x) = grad(nlp, x)
    H(x) = hess_op(nlp, x)
    fx = f(x)
    gx = g(x)
    B = H(x)
    Δ = min(max(0.1*norm(gx), 1), 100)
       ϵ = atol + rtol*norm(gx)
    while norm(gx) > ϵ
        d = steihaug(Δ, gx, B)
        ared = fx - f(x + d)
        pred = - dot(d,gx) - 0.5*dot(B*d,d)
        ρ = ared/pred
	if ρ > η1
            x = x + d 
        end
	if ρ < η1
            Δ = σ1*Δ
        elseif ρ > η2
            Δ = σ2*Δ
        end
        fx = f(x)
        gx = g(x)
        B = H(x)
	if k >= k_max
            saida = 1
            break
        end  
        tempo = time() - tempo_inicial
	if tempo >= tempo_max
            saida = 2
            break
        end
        k = k + 1
    end
    return x, fx, norm(gx), k, saida, tempo
end
