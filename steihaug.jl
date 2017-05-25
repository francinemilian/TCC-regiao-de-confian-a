function steihaug(Δ, g, B; ϵ = 1e-6, iter_mult = 10)
    n = length(g) 
    r = g
    d = -r
    z = rj = p =zeros(n)
    iter = 0
    maxiter = iter_mult*n
    if norm(r) < ϵ
        p = z
        return p
    end
    while iter <= maxiter
        bd = B*d        
        dot_dbd = dot(bd, d)
        if dot_dbd <= 0
            a = dot(d, d)
            b = dot(d, z)
            c = dot(z, z) - Δ^2
            τ = (-b + sqrt(b^2 - a*c))/a
            p = z + τ*d
            return p
            break
        end
        dot_r = dot(r, r)
        α = dot_r/dot_dbd
        zj = z + α*d
        if norm(zj) >= Δ
            a = dot(d, d)
            b = dot(d, z)
            c = dot(z, z) - Δ^2
            τ = (-b + sqrt(b^2 - a*c))/a      
            p = z + τ*d
            return p
            break
        end
        rj = r + α*bd
        if norm(rj) < ϵ
            p = zj
            return p
        end       
        dot_rj = dot(rj, rj)       
        β = dot_rj/dot_r
        d = -rj + β*d
        r = rj
        z = zj
        iter = iter + 1
    end
end
