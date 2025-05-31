function patches = normalizeData1(patches)%这种归一化为全局的归一化，平均值还是按列求的，但是标准差就是全局的标准差
patches = bsxfun(@minus, patches, mean(patches));
% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));%patches(:)表示把矩阵展开成一列，类似于reshape，然后这里std对整个列求标准差。
patches = max(min(patches, pstd), -pstd) / pstd;%经过这一步，patches最大为pstd，最小为-pstd，再除以pstd，得到最大为1，最小为-1.
% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
end