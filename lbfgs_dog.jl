using Krylov, LinearOperators, NLPModels

include("dogleg.jl")

function lbfgs_dog(nlp::AbstractNLPModel; η1 = 0.25, η2 = 0.75, σ1 = 0.5, σ2 = 2.0, atol = 1e-6, rtol = 1e-4)
    k_max = 1000
    tempo_max = 30
    saida = 0
    tempo = 0.0
    tempo_inicial = time()
    f(x) = obj(nlp, x)
    g(x) = grad(nlp, x)
    x = nlp.meta.x0
    fx = f(x)
    gx = g(x)
    n = nlp.meta.nvar
    Δ = min(max(0.1*norm(gx), 1), 100)
    tol = atol + rtol*norm(gx)
    k = 0
    B = LBFGSOperator(n)
    while norm(gx) > tol
        x_ant = x
        d = dogleg(gx, B, Δ)
        ared = fx - f(x + d)
        pred = -dot(d,gx) - 0.5*dot(B*d,d)
        ρ = ared/pred
	if ρ >= η1
            x = x + d
        end
	if ρ < η1
            Δ = σ1*Δ
        elseif ρ > η2
            Δ = σ2*Δ
        end
	if k >= k_max
            saida = 1
            break
        end
        tempo = time() - tempo_inicial
	if tempo >= tempo_max
            saida = 2
            break
        end
        gx_ant = gx
        gx = g(x)
        fx = f(x)
        s = gx - gx_ant
        y = x - x_ant
        push!(B, y, s)
        k = k + 1
    end
    return x, fx, norm(gx), k, saida, tempo
end
