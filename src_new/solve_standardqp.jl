# STRONG AsUMPTION: only underflows, never overflows of magnitude

# mettere denoise e tagliare rispetto al degree i gradienti

function solve_standardqp(A::Matrix,b::Vector,c::Vector,Q::Matrix, tol=1e-8, maxit=100; verbose=false, genLatex=false, slack_var=[])

	###########################
	# compute round tresholds #
	###########################

	# Needed to avoid noise of secondary gradients

    m,n = size(A)
	
	#####################
	# garbage variables #
	#####################
	
	var_to_show = setdiff(1:n, slack_var);
	
	flag = false;
	
	#################
	# Aux variables #
	#################

    iter = 0
	show = true
	show_more = false
	r = Matrix(undef, 0, 3); # just for genLatex purposes
	
	rb_den = norm(b);
	rb_den += magnitude(rb_den);
	rc_den = norm(c);
	rc_den += magnitude(rc_den);
	
	#linear = (all(x->x==0, Q)) ? true : false
	linear = false
	
	level = 1
	max_level = 2
	
	#################
    # initial value #
	#################
    
    x,λ,s = starting_point(A,b,c,Q, tol)

	x = map(z->principal(z), x)
	s = map(z->principal(z), s)
	λ = map(z->principal(z), λ)
	
	#max_deg_b = maximum(map(z->degree(z), b-A*x))-maximum(map(z->degree(z), b))
    #max_deg_c = maximum(map(z->degree(z), c+Q*x-A'*λ-s))-maximum(map(z->degree(z), c))
	
	min_deg_b = minimum(map(z->min_degree(z), b))-1 
	min_deg_c = minimum(map(z->min_degree(z), c))-1
	
	min_deg_r1 = min_deg_b - maximum(map(z->degree(z), b))
	min_deg_r2 = min_deg_c - maximum(map(z->degree(z), c))
	
	x_deg = map(x->degree(x), x)
	s_deg = map(x->degree(x), s)
	λ_deg = map(x->degree(x), λ)
	
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

		rb  = map(z->principal(z), denoise(b-A*x, tol))
        rc  = map(z->principal(z), denoise(c+Q*x-A'*λ-s, tol))
        rxs = map(z->principal(z), denoise(-x.*s, tol))
		
		rb -= retrieve_infinitesimals(rb, min_deg_b)
		rc -= retrieve_infinitesimals(rc, min_deg_c)
		

		if show_more
			print("rb: "); println(rb); #println(norm(rb))
			println("")
			print("rc: "); println(rc); #println(norm(rc))
			println("")
			print("rxs: "); println(rxs)
			println("")
		end
		
		f3 = fact3(A,Q,x,s, tol)
		
        λ_aff,x_aff,s_aff = solve3(f3,rb,rc,rxs)
		
		x_aff = map(z->principal(z), denoise(x_aff, tol/10)) # denoise only if gradient's entry is truly not impactful
		s_aff = map(z->principal(z), denoise(s_aff, tol/10))
		λ_aff = map(z->principal(z), denoise(λ_aff, tol/10))
		
		for i in eachindex(x_aff)
			x_aff[i] -= retrieve_infinitesimals(x_aff[i], x_deg[i]-1)
			s_aff[i] -= retrieve_infinitesimals(s_aff[i], s_deg[i]-1)
		end
		
		for i in eachindex(λ_aff)
			λ_aff[i] -= retrieve_infinitesimals(λ_aff[i], λ_deg[i]-1)
		end
		
		
		
		###########################
        # calculate α_aff, μ_aff  #
		###########################

        α_aff_pri  = principal(alpha_max(x,x_aff,1.0))
        α_aff_dual = principal(alpha_max(s,s_aff,1.0))
		
		!linear && ((α_aff_pri <= α_aff_dual) ? α_aff_dual = α_aff_pri : α_aff_pri = α_aff_dual)
		
		# not used rxs because some info in it is cut out (optimization to avoid double calculus is posible)
		μ = mean(x.*s)
        
		target_x = x+α_aff_pri*x_aff
		target_s = s+α_aff_dual*s_aff
		target = denoise(target_x.*target_s./n, tol)
		μ_aff = sum(target)
		
        σ = (μ_aff/μ)^3
		
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
		
		if show_more
			print("rxs: "); println(rxs)
			println("")
		end

        λ_cc,x_cc,s_cc = solve3(f3,rb,rc,rxs)
		
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

        dx = map(z->principal(z), denoise(x_aff+x_cc, tol/10)) # denoise only if gradient's entry is truly not impactful
        dλ = map(z->principal(z), denoise(λ_aff+λ_cc, tol/10))
        ds = map(z->principal(z), denoise(s_aff+s_cc, tol/10))
		
		for i in eachindex(dx)
			dx[i] -= retrieve_infinitesimals(dx[i], x_deg[i]-1)
			ds[i] -= retrieve_infinitesimals(ds[i], s_deg[i]-1)
		end
		
		for i in eachindex(dλ)
			dλ[i] -= retrieve_infinitesimals(dλ[i], λ_deg[i]-1)
		end

        α_pri = principal(min(0.99*alpha_max(x,dx,Inf),1))
        α_dual = principal(min(0.99*alpha_max(s,ds,Inf),1))
		
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

        x = denoise(x+α_pri*dx, tol)
        λ = denoise(λ+α_dual*dλ, tol)
        s = denoise(s+α_dual*ds, tol)
		
		###############
        # termination #
		###############

		cost_fun = dot(c,x)+0.5*x'*Q*x

		r1 = norm(denoise(A*x-b,tol))
		r2 = norm(denoise(A'*λ+s-c-Q*x,tol))
		r3 = dot(x,s)/n
		r1 /= rb_den
		r2 /= rc_den 
		# r3 /= (magnitude(cost_fun)+abs(cost_fun)) 
		
		#r1 = denoise(r1,tol)
		#r2 = denoise(r2,tol)
		r3 = denoise(r3,tol)
		
		r1 -= retrieve_infinitesimals(r1, 0-level)
		r2 -= retrieve_infinitesimals(r2, 1-level)
		r1 -= retrieve_infinitesimals(r1, min_deg_r1)
		r2 -= retrieve_infinitesimals(r2, min_deg_r2)
		r3 = principal(r3)
		r3 -= retrieve_infinitesimals(r3, 0-level)
		#r3 = standard_part(r3)
		
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

        if all(z->abs(z) < tol*10, r1.num) 
		
            #r2 = norm(A'*λ+s-c)/(1+norm(c))			

            if all(z->abs(z) < tol*10, r2.num) 

                #cx = dot(c,x)
                #r3 = abs(cx-dot(b,λ))/(1+abs(cx))

                if all(z->abs(z) < tol*10, r3.num) #r3 < tol*10 #
				
					if level == max_level

						flag = true;
						if show
							println("")
							print("OPTIMAL SOLUTION X: "); println(x); println("")
							print("OPTIMAL SOLUTION S: "); println(s); println("")
							print("OPTIMAL SOLUTION λ: "); println(λ); println("")
							println("")
						end

						return x,λ,s,flag,iter,r
					end
					
					x = denoise(x, 10*tol)
					s = denoise(s, 10*tol)
					λ = denoise(λ, 10*tol)
					
					N = map(x->x==0, x) # mask inactive/active entries of x/s
					B = .~N
					x[N] += map(x->η^(1-x),x_deg[N])
					s[B] += map(x->η^(1-x),s_deg[B])
					
					#x_deg[N] .-= 1
					#s_deg[B] .-= 1
					
					#L = map(x->x==0, λ)
					#λ_deg[L] .-= 1
					
					x_deg .-= 1
					s_deg .-= 1
					λ_deg .-= 1
					
					level += 1
					
                end
            end
        end
    end

    return x,λ,s,flag,iter,r
end
