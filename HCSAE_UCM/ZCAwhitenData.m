function patches = ZCAwhitenData(patches)
numpatches=size(patches,2);
 epsilon=0.1;
sigma = patches * patches' / numpatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';%
patches = ZCAWhite * patches;
save ZCAWhite.mat ZCAWhite;
end
