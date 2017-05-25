function dogleg(g, B, Δ)
    bg = B*g
    dot_g = dot(g, g)
    dot_gB = dot(bg, g)
    du = -(dot_g/dot_gB)*g
    if norm(du) > Δ
        return -(Δ/norm(g))*g
    end
    dn,_ = cg(B,-g)
    if norm(dn) <= Δ
        return dn
    end
    y = dn - du
    a = dot(y, y)
    b = dot(y, du)
    c = dot(du, du) - Δ^2
    α = (-b + sqrt(b^2 - a*c))/a
    d = du + α*(dn - du)
    return d
end
