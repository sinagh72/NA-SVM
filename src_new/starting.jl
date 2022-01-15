"""
obtain the starting point for PD method
(page 410 on Wright)
"""

function starting_point(A,b,c,Q, tol=1e-8)
    AA = A*A'
    # f = cholesky(AA)
    f = lu(AA)

	x = f\b
    x = denoise(A'*x, tol)
	
    λ = denoise(A*(c+Q*x), tol)
    λ = f\λ # no need of denoise because \ is already denoised

    s = denoise(A'*λ, tol)
    s = denoise(c+Q*x-s, tol)
	
	#=
	print("true_x: ")
	println(x)
	println("")
	print("true_s: ")
	println(s)
	println("")
	=#
	
	# this may change entries magnitude
    dx = max(-1.5*minimum(x),0.0)
    ds = max(-1.5*minimum(s),0.0)
	x = x.+dx
    s = s.+ds
	
	#=
	x[findall(x->x<0, x)] .*= -0.5
	s[findall(x->x<0, s)] .*= -0.5
	=#
	
	#=
	print("x': ")
	println(x)
	println("")
	print("s': ")
	println(s)
	println("")
	=#

    xs = dot(x,s)/2.0 # Think about the magnitude and to shrink it
	
 	#=
	print("xs: ")
	println(xs)
	println("")
	
	print("x.*s: ")
	println(x.*s)
	println("")
	=#
	
    dx = xs/sum(s)
    ds = xs/sum(x)

	
	x = x.+dx
    s = s.+ds
	
	
	#=
	mx = map(x->magnitude(x), x)./magnitude(dx)

    x = x+(dx.*mx)
    s = s+(ds./mx)
	=#
	#=
	print("x'': ")
	println(x)
	println("")
	print("s'': ")
	println(s)
	println("")
	error()
	=#
	
    return x,λ,s
end
