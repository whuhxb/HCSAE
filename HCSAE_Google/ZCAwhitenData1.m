function patches = ZCAwhitenData1(patches)
numpatches=size(patches,2);
 epsilon=0.1;
sigma = patches * patches' / numpatches;
[u, s, v] = svd(sigma);
ZCAWhite1 = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';%
patches = ZCAWhite1 * patches;
save ZCAWhite1.mat ZCAWhite1;
end