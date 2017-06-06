using Krylov

function newton_puro()
    f(x) = 0.5*(x[2]-x[1]^2)^2+0.5*(1-x[1])^2
    g(x) = [2*x[1]^3-2*x[1]*x[2]-1+x[1], -x[1]^2+x[2]]
    h(x) = [6*x[1]^2-2*x[2]+1  -2*x[1]; -2*x[1]  1]
    x = [-1.2, 1.0]
    gx = g(x)
    b = h(x)
    k = 0
        
    while norm(gx) > 1e-6
        d,_ = cg(b, - gx)
        print(x, d, k)
        x = x + d
        gx = g(x)
        b = h(x)
        k = k + 1
    end
    return x, d, k
end

function cauchy_quad()
    f(x) = 3*x[1]^2+3*x[1]*x[2]+2*x[2]^2 + x[1] + x[2]
    g(x) = [6*x[1]+3*x[2]+1, 3*x[1]+4*x[2]+1]
    h(x) = [6 3; 3 4]
    x = [0,0]
    gx = g(x)
    b = h(x)
    k = 0
    d = -gx
    while norm(gx) > 1e-6
        d = - gx
        print(k, x, d)
        p1 = dot(d, d)
        p2 = dot(b*d, d)
        t = p1/p2
        x = x + t*d
        gx = g(x)
        b = h(x)
        k = k + 1
    end
    return x, d, k 
end
