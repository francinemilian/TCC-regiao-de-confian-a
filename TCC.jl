using CUTEst, BenchmarkProfiles, Plots

include("RegiaoDeConfianca.jl")

problems = CUTEst.select(max_con=0, only_free_var = true)

metodos = [lbfgs_dog, newton_stg]

sort!(problems)

prob = length(problems)
met = length(metodos)

TE = -ones(prob, met)
AV = -ones(prob, met)

	for (j,m) in enumerate(metodos)
		open("$m.txt", "w") do file 
		str = @sprintf("%8s  %10s  %10s  %7s  %7s  %7s  %10s\n","Problema", "f(x)", "||g(x)||","iter","saida","avaliações","tempo")
		print(str)
		print(file,str)
		for (i,p) in enumerate(problems)
			nlp = CUTEstModel(p)
			c = nlp.counters
			try
				x, fx, ngx, iter, s, t = m(nlp)
				if isnan(ngx)
					s=3
				end
				av = c.neval_obj+c.neval_grad+c.neval_hess+c.neval_hprod
				str = @sprintf("%8s  %10.1e  %10.1e  %7d  %7d  %7d  %10.8f\n",p,fx,ngx,iter,s,av,t)
				print(str)
				print(file,str)
				if s == 0
					TE[i,j] = t
					AV[i,j] = av
				end
				reset!(nlp)
			catch
				s = 3
				srt = @printf("%-7s  %s\n", p, s)
				print(str)
				print(file,str)
				reset!(nlp)
			finally
				finalize(nlp)
			end
		end
	end
end

performance_profile(TE, ["LBFGS-Dogleg", "Newton-Steihaug"])
png("perprof_t")

performance_profile(AV, ["LBFGS-Dogleg", "Newton-Steihaug"])
png("perprof_av")
