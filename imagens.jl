using Plots
gr(size=(800, 600))

function gera_imagem1()

    f(x) = 0.5*(x[2]-x[1]^2)^2+0.5*(1-x[1])^2

    x = linspace(-1.5, 1.5, 100)
    y = linspace(-2.5, 2.5, 100)
    contour(x, y, (x,y)->f([x;y]))

    x0=[-1.2,1.0];
    x1=[-0.0297872,-1.36851];
    x2=[0.245646,-0.0155215];
    x3=[0.900623,0.382126];
    x4=[0.954109,0.907463];
    x5=[0.999739,0.997396];
    
    X = [x0[1] x1[1] x2[1] x3[1] x4[1] x5[1];
         x0[2] x1[2] x2[2] x3[2] x4[1] x5[2]]
    plot!(X[1,:], X[2,:], c=:blue, l=:solid, label="d")
    scatter!(X[1,:], X[2,:], markersize=2, c=:red, label="x")
    

    
    png("newton")
end

gera_imagem1()

function gera_imagem2()

    f(x) = 3*x[1]^2+3*x[1]*x[2]+2*x[2]^2 + x[1] + x[2]

    x = linspace(-0.2, 0.2, 100)
    y = linspace(-0.3, 0.2, 100)
    contour(x, y, (x,y)->f([x;y]))
    t = linspace(0, 2pi, 100)
    
    x0=[0,0];
    x1=[-0.125,-0.125];
    x2=[-0.0625,-0.1875];
    x3=[-0.0703125,-0.195313];
    x4=[-0.0664063,-0.199219];
    x5=[-0.0668945,-0.199707];
    x6=[-0.0666504,-0.199951];
    x7=[-0.0666809,-0.199982];
    x8=[-0.0666656,-0.199997];
    x9=[-0.0666676,-0.199999];
    x10=[-0.0666666,-0.2];
    x11=[-0.0666667,-0.2]
    
    X = [x0[1] x1[1] x2[1] x3[1] x4[1] x5[1] x6[1] x7[1] x8[1] x9[1] x10[1] x11[1];
         x0[2] x1[2] x2[2] x3[2] x4[2] x5[2] x6[2] x7[2] x8[2] x9[2] x10[2] x11[2]] 
    
    plot!(X[1,:], X[2,:], c=:blue, l=:solid, label="d")
    scatter!(X[1,:], X[2,:], markersize=2, c=:red, label="x")
    
     png("grad")
end

gera_imagem2()
