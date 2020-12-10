# Exercise 7 - HW3
using Pkg
Pkg.add("QuadGK")
Pkg.add("SymPy")
Pkg.add("MAT")
Pkg.add("MatrixDepot")
using LinearAlgebra, QuadGK, SymPy, MatrixDepot, MAT, SparseArrays

function gramian_val(A, B, t, tol)
    err = 1;
    i = 0;
    while err > tol
        T = t^(i+1)/factorial(i+1);
        for k = 0:i
            C += (-1)^k*factorial(i)/(factorial(k)*factorial(i-k))*(A^k*B*A^(i-k));
        end
        P += T*C;
        err_e += 1/factorial(i);
        err = norm(B)*(e-err_e)
        i += 1;
    end
    return P
end

tau = 1;
tol = 1e-10;
A = matrixdepot("neumann", 20^2)
A = A-A'; # make A antisymmetric
A = A/(2*norm(A,1));
B = sprandn(size(A,1), size(A,1), 0.05);

myapp = @timed P = commutator_val(A, B, tau, tol);

# naive approach
f = t->exp(-t*A)*B*exp(t*A);
naive = @timed P_naive = quadgk(f, 0, tau)

println("Time for naive is ", naive.time,
        " whereas for our algorithm is ", myapp.time)
