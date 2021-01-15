function Km = getUpsamplingKernel(nC, sig, upsamp)

xs = 1:nC;
xn = linspace(1, nC, nC * upsamp);
d0 = (xs' - xs).^2;
d1 = (xn' - xs).^2;
K0 = exp(-d0/sig);
K1 = exp(-d1/sig);
Km = K1 / (K0 + 0.001 * eye(nC));

end