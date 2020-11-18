include("../Homework_1_Valerio/arnoldi_val.jl")
include("../Homework_1_Valerio/GS_val.jl")

function my_gmres_val(A,b,k)
    e1 = zeros(k+1,1);
    e1[1] = 1;
    Q, H = arnoldi(A,b,k)
    b_norm = norm(b)
    z = H\(b_norm .*e1);
    x = Q[:,1:k]*z;
    return x
end
