function isort = sort_spike_trains(S)

% by Mattia, basically a wrapper of the sorting part of rastermap

[U, Sv, ~] = svd(S); % U has "eigenvectors" ("principal components") of S*St, Sv the "singular values"
[~, isort] = sort(U(:, 1), 'descend'); % sort by "first singular value"

if min(size(S, 1), size(S, 2)) < 100 
    iPC = 1 : min(size(S, 1), size(S, 2));
else
    iPC = 1 : 100;
end
S = U(:, iPC) * Sv(iPC, iPC); % reconstruct S using i "Principal Components"
[NN, nPC] = size(S);

nC = 30; % number of clusters
nn = floor(NN/nC); % number of spike trains assigned to one cluster
iclust = zeros(NN, 1); % initialize clusters
iclust(isort) = ceil((1 : NN) / nn); % assign cluster based on order of first component
iclust(iclust>nC) = nC;  % correct cluster assignment

sig = [linspace(nC / 10, 1, 100) 1 * ones(1, 50)];
for t = 1 : numel(sig)
    V = zeros(nPC, nC, 'single');
    for j = 1 : nC
        ix = iclust == j;
        V(:, j) = sum(S(ix, :), 1);
    end
    
    V = my_conv2(V, sig(t), 2);
    V = normc(V);
    
    cv = S * V;
    [~, iclust] = max(cv, [], 2);
end

% create kernels for upsampling
sigUp = 1;
upsamp = 100;
Km = getUpsamplingKernel(nC, sigUp, upsamp);
[~, iclustup] = max(cv * Km', [], 2);
iclustup = gather_try(iclustup);
[~, isort] = sort(iclustup);

end