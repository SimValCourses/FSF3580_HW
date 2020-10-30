using LinearAlgebra

function PM(A,v,k)
    for i in 1:k
        v = A*v
        v = v/norm(v)
    end
    return v
end
