# Exercise 7 - HW3
#using Pkg
#Pkg.add("QuadGK")
#Pkg.add("SymPy")
#Pkg.add("MAT")
#Pkg.add("MatrixDepot")
using LinearAlgebra, QuadGK, SymPy, MatrixDepot, MAT, SparseArrays

function gramian_val(A, B, t, tol)
    n = size(A,1)
    err = 1;
    i = 0;
    P = spzeros(n,n);
    err_e = 0;
    while err > tol
        T = t^(i+1)/factorial(i+1);
        C = spzeros(n,n)
        for k = 0:i
            fact = (-1)^k*factorial(i)/(factorial(k)*factorial(i-k))
            C += fact*(A^k*B*A^(i-k));
        end
        P += T*C;
        err_e += 1/factorial(i);
        err = norm(B)*(exp(1)-err_e)
        i += 1;
    end
    return P
end

tau = 1;
tol = 1e-10;
A = matrixdepot("neumann", 30)
A = A-A'; # make A antisymmetric
A = A/(2*norm(A,1));
B = sprandn(size(A,1), size(A,1), 0.05);

myapp = @timed P = gramian_val(A, B, tau, tol);

# naive approach
f(t) = exp(collect(-t*A))*B*exp(collect(t*A));
naive = @timed P_naive = quadgk(f, 0, tau)

println("Time for naive is ", naive.time,
        " whereas for our algorithm is ", myapp.time)
