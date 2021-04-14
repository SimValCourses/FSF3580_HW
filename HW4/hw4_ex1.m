nnv = [2:2:10];
res_bs = zeros(length(nnv),1);
t_bs = zeros(length(nnv),1);
res_naive = zeros(length(nnv),1);
t_naive = zeros(length(nnv),1);

for i = 1:length(nnv)
	n = nnv(i);
	A=randn(n); 
	P=randn(n);
	X0 = lyap(A,P);
	f1 = @() lyap0(A,P);
	t_bs(i) = timeit(f1);
	res_bs(i)=norm(A*X0+X0*A'+P);

	X1 = naive_lyap(A,P);
	f2 = @() naive_lyap(A,P);
	t_naive(i) = timeit(f2);
	res_naive(i)=norm(A*X1+X1*A'+P);
end

semilogy(nnv, t_bs, nnv, t_naive, nnv, nnv.^6, 'k--', nnv, nnv.^3, 'k-*')
