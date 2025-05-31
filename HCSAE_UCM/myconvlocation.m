function convlocation=myconvlocation(image,patchDim)
[im,in,imagechannel]=size(image);
locate=zeros( patchDim*patchDim ,(im - patchDim + 1)*(in - patchDim + 1));
count=0;
for j=1:in - patchDim + 1
    for i=1:im - patchDim + 1
        b=[];
        a=im*(j-1)+i: im*(j-1)+i+patchDim-1;
        a=a';
        for t=1:patchDim
            b=[b;a+im*(t-1)];
        end
    count=count+1;
    locate(:,count)=b;
    end
end

convlocation=[];
count=0;
for i=1:imagechannel
convlocation=[convlocation;locate+im*in*(i-1)];
end