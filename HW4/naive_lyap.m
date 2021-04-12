function X=naive_lyap(A,P)
%  Solves the Lyapunov equation
%    AX+XA'+P=0
%  using a naive approach, i.e. solves the linear system
%  
%  Bx = c,
%  where B = kron(I,A)+kron(A,I), x = vec(X), c = vec(-P)	
%

n = size(A,1);
W = -P;
I = eye(n);
B = kron(I,A)+kron(A,I);
c = W(:);

x = B\c;

X = reshape(x,n,length(x)/n);