# STRONG ASSUMPTION: only underflows, never overflows of magnitude

function solve_standardqp(A::Matrix,b::Vector,c::Vector,Q::Matrix, tol=1e-8, maxit=100; verbose=false, genLatex=false, slack_var=[])

	###########################
	# compute round tresholds #
	###########################

	# Needed to avoid noise of secondary gradients

    m,n = size(A)

	min_deg_Q = minimum(map(x->min_degree(x), Q))
	min_deg_c = minimum(map(x->min_degree(x), c))

	min_deg_A = minimum(map(x->min_degree(x), A))
	min_deg_b = minimum(map(x->min_degree(x), b))

	max_deg_Q = maximum(map(x->degree(x), Q))
	max_deg_c = maximum(map(x->degree(x), c))

	max_deg_A = maximum(map(x->degree(x), A))
	max_deg_b = maximum(map(x->degree(x), b))

	levels_dual = maximum([max_deg_Q-min_deg_Q, max_deg_c-min_deg_c, max_deg_Q-min_deg_c, max_deg_c-min_deg_Q])+1
	levels_prim = maximum([max_deg_A-min_deg_A, max_deg_b-min_deg_b, max_deg_A-min_deg_b, max_deg_b-min_deg_A])+1

	n_levels_max = max(levels_prim, levels_dual)
	
	trash_deg_b = max_deg_b-n_levels_max # max_deg_b-levels_prim
	trash_deg_c = max_deg_c-n_levels_max # max_deg_c-levels_dual
	trash_deg_xs = 2
	
	#####################
	# garbage variables #
	#####################
	
	var_to_show = setdiff(1:n, slack_var);
	
	flag = false;
	
	#################
	# Aux variables #
	#################

    iter = 0
	show = false
	show_more = false
	r = Matrix(undef, 0, 3); # just for genLatex purposes
	
	rb_den = norm(b);
	rb_den += magnitude(rb_den);
	rc_den = norm(c);
	rc_den += magnitude(rc_den);
	
	#linear = (all(x->x==0, Q)) ? true : false
	linear = false
	
	###############
	# Standardize #
	###############
	
	n_levels = 1
	
	#################
    # initial value #
	#################
    
    x,λ,s = starting_point(A,b,c,Q, tol)

	x = map(x->principal(x), x)
	s = map(x->principal(x), s)
	λ = map(x->principal(x), λ)

	x_deg = map(x->degree(x), x)
	s_deg = map(x->degree(x), s)
	
	if genLatex
		println("\t\\textbf{iter} & \$\\bm{\\mu}\$ & \$\\bm{x}\$ & \$\\bm{f(x)}\$\\\\");
		println("\t\\hline");
		print("\t$(iter) & \$"); print_latex(mean(x.*s)); print("\$ & \$"); print_latex(x[var_to_show]); print("\$ & \$"); print_latex(0.5*(x'*Q*x)+dot(c, x)); println("\$ \\\\");
		println("\t\\hline");
    elseif verbose
        print(iter); print(" "); print(mean(x.*s)); print(" "); print(norm([A'*λ + s + Q*x - c; A*x - b; x.*s])/norm([b;c])); print(" "); println("0., 0."); 
    end

    for iter=1:maxit
	
        ##############
		# solve 10.1 #
		##############

		rb  = b-A*x
        rc  = c+Q*x-A'*λ-s
        rxs = -x.*s
		
		rb = denoise(rb, tol)
		rc = denoise(rc, tol)
		
		
		rb -= retrieve_infinitesimals(rb, trash_deg_b)
		rc -= retrieve_infinitesimals(rc, trash_deg_c)
		# in NAQP a minimum meaningful degree for rxs (i.e., μ) does not exists
		# To avoid numerical instabilities, two monosemia are kept for each entry in rxs
		rxs -= parametric_retrieve_infinitesimals(rxs, trash_deg_xs)

		#=
		rb -= parametric_retrieve_infinitesimals(rb, n_levels)
		rc -= parametric_retrieve_infinitesimals(rc, n_levels)
		rxs -= parametric_retrieve_infinitesimals(rxs, n_levels)
		=#

		if show_more
			print("rb: "); println(rb); #println(norm(rb))
			println("")
			print("rc: "); println(rc); #println(norm(rc))
			println("")
			print("rxs: "); println(rxs)
			println("")
		end
		
		f3 = fact3(A,Q,x,s)
		
        λ_aff,x_aff,s_aff = solve3(f3,rb,rc,rxs)		
		
		if n_levels == n_levels_max
			x_aff -= parametric_retrieve_infinitesimals(x_aff, n_levels_max)
			s_aff -= parametric_retrieve_infinitesimals(s_aff, n_levels_max)
			λ_aff -= parametric_retrieve_infinitesimals(λ_aff, n_levels_max)
		else
			x_aff = map(x->principal(x), x_aff)
			s_aff = map(x->principal(x), s_aff)
			λ_aff = map(x->principal(x), λ_aff)
		end
		
		x_aff = denoise(x_aff, tol) 
		s_aff = denoise(s_aff, tol)
		λ_aff = denoise(λ_aff, tol)

		###########################
        # calculate α_aff, μ_aff  #
		###########################

        α_aff_pri  = alpha_max(x,x_aff,1.0, n)
        α_aff_dual = alpha_max(s,s_aff,1.0, n)
		
		α_aff_pri  -= retrieve_infinitesimals(α_aff_pri, -1)
		α_aff_dual -= retrieve_infinitesimals(α_aff_dual, -1)
		
		!linear && ((α_aff_pri <= α_aff_dual) ? α_aff_dual = α_aff_pri : α_aff_pri = α_aff_dual)
		
		# not used rxs because some info in it is cut out (optimization to avoid double calculus is possible)
		μ = mean(x.*s)
        
		target_x = denoise(x+α_aff_pri*x_aff, tol)
		target_s = denoise(s+α_aff_dual*s_aff, tol)
		target = denoise(target_x.*target_s./n, tol)
		μ_aff = sum(target)
		
        σ = (μ_aff/μ)^3 #(μ==0) ? σ = 0 : σ = (μ_aff/μ)^3 # 
		σ -= retrieve_infinitesimals(σ, -1)
		
		if show_more
			print("x_aff: "); println(x_aff)
			println("")
			print("s_aff: "); println(s_aff)
			println("")
			print("λ_aff: "); println(λ_aff)
			println("")
			println("")
			print("α_aff_pri: "); println(α_aff_pri)
			print("α_aff_dual: "); println(α_aff_dual)
			println("")
			print("μ_aff: "); println(μ_aff)
			println("")
		end
			
		##############
        # solve 10.7 #
		##############

        rb = zeros(m)
        rc = zeros(n)
        rxs = denoise(σ*μ.-α_aff_pri*α_aff_dual*x_aff.*s_aff, tol)
		# Same choice of -2 as in previous rxs
		rxs -= parametric_retrieve_infinitesimals(rxs, trash_deg_xs)
		
		if show_more
			print("rxs: "); println(rxs)
			println("")
		end

        λ_cc,x_cc,s_cc = solve3(f3,rb,rc,rxs)
		
		if n_levels == n_levels_max
			x_cc -= parametric_retrieve_infinitesimals(x_cc, n_levels_max)
			s_cc -= parametric_retrieve_infinitesimals(s_cc, n_levels_max)
			λ_cc -= parametric_retrieve_infinitesimals(λ_cc, n_levels_max)
		else
			x_cc = map(x->principal(x), x_cc)
			s_cc = map(x->principal(x), s_cc)
			λ_cc = map(x->principal(x), λ_cc)
		end
		
		x_cc = denoise(x_cc, tol) 
		s_cc = denoise(s_cc, tol)
		λ_cc = denoise(λ_cc, tol)
		
		if show_more
			print("x_cc: "); println(x_cc)
			println("")
			print("s_cc: "); println(s_cc)
			println("")
			print("λ_cc: "); println(λ_cc)
			println("")
		end
		
		##############################
        # compute direction and step #
		##############################

        dx = x_aff+x_cc
        dλ = λ_aff+λ_cc
        ds = s_aff+s_cc

		if n_levels == n_levels_max
			dx -= parametric_retrieve_infinitesimals(dx, n_levels_max)
			ds -= parametric_retrieve_infinitesimals(ds, n_levels_max)
			dλ -= parametric_retrieve_infinitesimals(dλ, n_levels_max)
		else
			dx = map(x->principal(x), dx)
			ds = map(x->principal(x), ds)
			dλ = map(x->principal(x), dλ)
		end

        α_pri = min(0.99*alpha_max(x,dx,Inf, n),1)
        α_dual = min(0.99*alpha_max(s,ds,Inf, n),1)
		
		α_pri  -= retrieve_infinitesimals(α_pri, -1)
		α_dual -= retrieve_infinitesimals(α_dual, -1)
		
		!linear && ((α_pri <= α_dual) ? α_dual = α_pri : α_pri = α_dual)
		
		if show
			print("dx: "); println(dx)
			println("")
			print("ds: "); println(ds)
			println("")
			print("dλ: "); println(dλ)
			println("")
			println("")
			print("α_pri: "); println(α_pri)
			print("α_dual: "); println(α_dual)
			println("")
		end
		
		###############################
        # compute x^k+1, λ^k+1, s^k+1 #
		###############################

        x = x+α_pri*dx
        λ = λ+α_dual*dλ
        s = s+α_dual*ds
		
		x = denoise(x, tol)		
		s = denoise(s, tol)
		λ = denoise(λ, tol)
		
		###############
        # termination #
		###############

		cost_fun = dot(c,x)+0.5*x'*Q*x

		# 10*tol is needed to avoid instabilities when computing the norm of a vector of BANs with ld(x) = 1e-8
		# A more precise choice of the threshold comes from Theorem 4, think about adding it
		r1 = norm(denoise(A*x-b, 10*tol))
		r2 = norm(denoise(A'*λ+s-c-Q*x, 10*tol))
		r3 = denoise(dot(x,s)/n, 10*tol)
		
		r1 -= retrieve_infinitesimals(r1, max_deg_b-n_levels)
		r2 -= retrieve_infinitesimals(r2, max_deg_c-n_levels)
		
		r1 /= rb_den #(1+norm(b))
		r2 /= rc_den #(1+norm(Q*x+c)) #(1+norm(c)) #
		r3 /= (magnitude(cost_fun)+abs(cost_fun)) #(1+abs(cost_fun)) # 

		r3 -= parametric_retrieve_infinitesimals(r3, trash_deg_xs)

		r1 -= retrieve_infinitesimals(r1, -n_levels)
		r2 -= retrieve_infinitesimals(r2, -n_levels)
		r3 -= retrieve_infinitesimals(r3, -n_levels)

        if genLatex
			print("\t$(iter) & \$"); print_latex(mean(x.*s)); print("\$ & \$"); print_latex(x[var_to_show]); print("\$ & \$"); print_latex(cost_fun); println("\$ \\\\");
			println("\t\\hline");
			r = [r
				 r1 r2 r3];
		elseif verbose
            print(iter); print(" "); print(mean(x.*s)); print(" "); print(norm([A'*λ + s - c - Q*x; A*x - b; x.*s])/norm([b;c])); print(" "); print(α_pri); print(" "); println(α_dual); 
        end
		
		if show
			println("")
			print("x: "); println(x)
			println("")
			print("x_std: "); println(map(x->standard_part(x), x))
			println("")
			print("s: "); println(s)
			println("")
			print("λ: "); println(λ)
			println("")
			print("f(x): "); println(cost_fun)
		end
		
		if show
			println("")
			print("r1: "); println(r1)
			print("r2: "); println(r2)
			print("r3: "); println(r3)
			println("");
			print("μ: "); println(mean(x.*s));
			println("")
		end

        if (typeof(r1)<:Real) ? r1 < tol : all(z->abs(z) < tol, r1.num) 
		
            #r2 = norm(A'*λ+s-c)/(1+norm(c))			

            if (typeof(r2)<:Real) ? r2 < tol : all(z->abs(z) < tol, r2.num) 

                #cx = dot(c,x)
                #r3 = abs(cx-dot(b,λ))/(1+abs(cx))

                if (typeof(r3)<:Real) ? r3 < tol : all(z->abs(z) < tol, r3.num) 
				
					if n_levels == n_levels_max
					
						#=
						x -= parametric_retrieve_infinitesimals(x, levels_prim)
						s -= parametric_retrieve_infinitesimals(s, levels_dual)
						λ -= parametric_retrieve_infinitesimals(λ, levels_dual)
						=#

						flag = true;
						if show
							println("")
							print("OPTIMAL SOLUTION X: "); println(x); println("")
							print("OPTIMAL SOLUTION S: "); println(s); println("")
							print("OPTIMAL SOLUTION λ: "); println(λ); println("")
							println("")
						end

						return x,λ,s,flag,iter,r
						
					else
						# careful, maybe a too big cut
						# TODO use the theoretical threshold x,s<sqrt(n*tol)
						x = denoise(x, tol*n)
						s = denoise(s, tol*n)
						λ = denoise(λ, tol*n)
						
						# infinitesimal noise addition (it is infinitesimal w.r.t. the active entry)
						N = map(x->x==0, x) # mask inactive/active entries of x/s
						B = .~N
						x[N] += map(x->η^(1-x),x_deg[N])
						s[B] += map(x->η^(1-x),s_deg[B])

						x_deg[N] .-= 1
						s_deg[B] .-= 1
						
						n_levels += 1
					end
                end
            end
        end
    end

    return x,λ,s,flag,iter,r
end

@inline function parametric_retrieve_infinitesimals(x, d)
	return map(z->retrieve_infinitesimals(z, degree(z)-d), x)
end
