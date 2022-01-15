function solve_standardqp(A,b,c,Q, tol=1e-8, maxit=100; verbose=false, genLatex=false, slack_var=[])

	###########################
	# compute round tresholds #
	###########################

    m,n = size(A)
	min_deg_Q = minimum(map(x->min_degree(x), Q))
	min_deg_A = minimum(map(x->min_degree(x), A))
	min_deg_c = minimum(map(x->min_degree(x), c))
	min_deg_b = minimum(map(x->min_degree(x), b))
	
	trash_deg_r1 = min(min_deg_A, min_deg_b)-1;
	trash_deg_r2 = min(min_deg_Q, min_deg_A, min_deg_c)-1;
	trash_deg = min(trash_deg_r2, min_deg_b);
	
	max_deg_Q = maximum(map(x->degree(x), Q))-1
	max_deg_c = maximum(map(x->degree(x), c))-1
	
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
	
	linear = (all(x->x==0, Q)) ? true : false
	
	###############
	# Standardize #
	###############
	
	Q_true = copy(Q)
	c_true = copy(c)
	
	_Q = map(x->retrieve_infinitesimals(x, max_deg_Q), Q)
	_c = map(x->retrieve_infinitesimals(x, max_deg_c), c)
	
	Q -= _Q
	c -= _c
	
	n_levels = 1
	
	#################
    # initial value #
	#################
    
    x,λ,s = starting_point(A,b,c,Q)

	if genLatex
		println("\t\\textbf{iter} & \$\\bm{\\mu}\$ & \$\\bm{x}\$ & \$\\bm{c^Tx}\$\\\\");
		println("\t\\hline");
		print("\t$(iter) & \$"); print_latex(mean(x.*s)); print("\$ & \$"); print_latex(x[var_to_show]); print("\$ & \$"); print_latex(0.5*(x'*Q_true*x)+dot(c_true, x)); println("\$ \\\\");
		println("\t\\hline");
    elseif verbose
        print(iter); print(" "); print(mean(x.*s)); print(" "); print(norm([A'*λ + s + Q_true*x - c_true; A*x - b; x.*s])/norm([b;c_true])); print(" "); println("0., 0."); 
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

		#rb -= retrieve_infinitesimals(rb, trash_deg_r1)
		rb -= retrieve_infinitesimals(rb, trash_deg)
		#rc -= retrieve_infinitesimals(rc, trash_deg_r2)
		rc -= retrieve_infinitesimals(rc, trash_deg)
		rxs -= retrieve_infinitesimals(rxs, trash_deg)

		if show_more
			print("rb: "); println(norm(rb))
			println("")
			print("rc: "); println(norm(rc))
			println("")
		end
		
		f3 = fact3(A,Q,x,s)
		
        λ_aff,x_aff,s_aff = solve3(f3,rb,rc,rxs)
		
		x_aff = denoise(x_aff, tol) 
		s_aff = denoise(s_aff, tol)
		λ_aff = denoise(λ_aff, tol)		
		
		x_aff -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), x_aff)
		s_aff -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), s_aff)
		λ_aff -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), λ_aff)

		###########################
        # calculate α_aff, μ_aff #
		###########################

        α_aff_pri  = alpha_max(x,x_aff,1.0, n)
        α_aff_dual = alpha_max(s,s_aff,1.0, n)
		
		α_aff_pri  -= retrieve_infinitesimals(α_aff_pri, -1)
		α_aff_dual -= retrieve_infinitesimals(α_aff_dual, -1)
		
		!linear && ((α_aff_pri <= α_aff_dual) ? α_aff_dual = α_aff_pri : α_aff_pri = α_aff_dual)
		
		μ = -mean(rxs)
        
		target_x = denoise(x+α_aff_pri*x_aff, tol)
		target_s = denoise(s+α_aff_dual*s_aff, tol)
		target_x[findall(x->x<0, target_x)] .*= -1 #.= 0 #
		target_s[findall(x->x<0, target_s)] .*= -1 #.= 0 #
		target = denoise(target_x.*target_s./n, tol)
		target[findall(x->x<0, target)] .*= -1 #.= 0 #
		μ_aff = sum(target)
		
        (μ==0) ? σ = 0 : σ = (μ_aff/μ)^3 # σ = (μ_aff/μ)^3
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
		
		rxs -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), rxs)

        λ_cc,x_cc,s_cc = solve3(f3,rb,rc,rxs)

		x_cc = denoise(x_cc, tol) 
		s_cc = denoise(s_cc, tol)
		λ_cc = denoise(λ_cc, tol)

		x_cc -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), x_cc)
		s_cc -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), s_cc)
		λ_cc -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), λ_cc)
		
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

		dx -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), dx)
		ds -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), ds)
		dλ -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), dλ)
		
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
		x[findall(x->x<0, x)] .*= -1 #.= 0 #
		
		s = denoise(s, tol)
		s[findall(x->x<0, s)] .*= -1 #.= 0 #
			
		λ = denoise(λ, tol)
		
		x -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), x)
		s -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), s)
		λ -= map(x->retrieve_infinitesimals(x, degree(x)-n_levels), λ)
		
		###############
        # termination #
		###############

		cost_fun = dot(c_true,x)+0.5*x'*Q_true*x

		r1 = norm(denoise(A*x-b, tol))
		r2 = norm(denoise(A'*λ+s-c-Q*x, tol))
		r3 = denoise(dot(x,s)/n, tol)

		r1 -= retrieve_infinitesimals(r1, trash_deg_r1)
		r2 -= retrieve_infinitesimals(r2, trash_deg_r2)
		r3 -= retrieve_infinitesimals(r3, trash_deg)

		r1 -= retrieve_infinitesimals(r1, -n_levels)
		r2 -= retrieve_infinitesimals(r2, -n_levels)
		r3 -= retrieve_infinitesimals(r3, -n_levels)

		r1 /= rb_den #(1+norm(b))
		r2 /= rc_den #(1+norm(Q*x+c)) #(1+norm(c)) #
		r3 /= (magnitude(cost_fun)+abs(cost_fun)) #(1+abs(cost_fun)) # 
		
		#Added because I think they are right, deeper investigation is needed
		r1 = principal(r1)
		r2 = principal(r2)
		r3 = principal(r3)

        if genLatex
			print("\t$(iter) & \$"); print_latex(mean(x.*s)); print("\$ & \$"); print_latex(x[var_to_show]); print("\$ & \$"); print_latex(cost_fun); println("\$ \\\\");
			println("\t\\hline");
			r = [r
				 r1 r2 r3];
		elseif verbose
            print(iter); print(" "); print(mean(x.*s)); print(" "); print(norm([A'*λ + s - c_true - Q_true*x; A*x - b; x.*s])/norm([b;c_true])); print(" "); print(α_pri); print(" "); println(α_dual); 
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
				
					if all(x->x==0, _Q) && all(x->x==0, _c)

						flag = true;
						if show
							println("")
							print("OPTIMAL SOLUTION X: "); println(x); println("")
							print("OPTIMAL SOLUTION S: "); println(s); println("")
							print("OPTIMAL SOLUTION λ: "); println(λ); println("")
							println("")
						end
						
						x -= retrieve_infinitesimals(x, -1)
						s -= retrieve_infinitesimals(s, -n_levels)
						λ -= retrieve_infinitesimals(λ, -n_levels)
						
						return x,λ,s,flag,iter,r
						
					else
						
						x = denoise(x, tol*100)
						x[findall(x->x<0, x)] .*= -1 #.= 0 #
						

						s = denoise(s, tol*100)
						s[findall(x->x<0, s)] .*= -1 #.= 0 #
							
						λ = denoise(λ, tol*100)
						
						noise = ones(length(x)).*(η^n_levels)
						x += noise
						s += noise
						
						#Q += _Q
						#c += _c
						
						max_deg_Q -= 1
						max_deg_c -= 1
						
						_Q = map(x->retrieve_infinitesimals(x, max_deg_Q), Q_true)
						_c = map(x->retrieve_infinitesimals(x, max_deg_c), c_true)
						
						Q = Q_true - _Q
						c = c_true - _c
						
						n_levels += 1
					end
                end
            end
        end
    end

    return x,λ,s,flag,iter,r
end
