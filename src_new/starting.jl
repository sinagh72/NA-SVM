"""
obtain the starting point for PD method
case of coupled variables
page 229
"""

function starting_point(A, b, c ,Q, tol=1e-8)
    AA = denoise(A*A', tol)
    f = lu(AA)
	f = LU{eltype(AA),typeof(AA)}(denoise(f.factors, tol), f.ipiv, f.info)
	
	x = denoise(f\b, tol)
	x = denoise(A'*x, tol)
	
    λ = denoise(A*(c+Q*x), tol)
    λ = denoise(f\λ, tol)

    s = denoise(A'*λ, tol)
    s = denoise(c+Q*x-s, tol)
	
	# this may change entries magnitude
    dx = max(-1.5*minimum(x),0.0)
    ds = max(-1.5*minimum(s),0.0)
	x = x.+dx
    s = s.+ds

    xs = dot(x,s)/2.0
	
    dx = xs/sum(s)
    ds = xs/sum(x)

	
	x = x.+dx
    s = s.+ds
	
    return x,λ,s
end
