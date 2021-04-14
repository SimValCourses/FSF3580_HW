function DD = second_der(n,h)
e = ones(n,1);
DD = 1/h^2*spdiags([e -2*e e], -1:1, n, n);
%A(n,:) = 1/h^2*[zeros(1,n-2),1,-1];
%A(1,1) = -1/h^2;
if n < 50
    DD = full(DD);
else
    DDf = DD;
end
end


