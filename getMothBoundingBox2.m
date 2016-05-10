
function  result_bbox = getMothBoundingBox2(cdata,view_id)

debug_flag = 1;

cdata = double(cdata);
x_deriv_filter = [1 0 -1];
y_deriv_filter = [1 0 -1]';


x_deriv = imfilter(cdata,x_deriv_filter,'replicate','corr','same');
y_deriv = imfilter(cdata,y_deriv_filter,'replicate','corr','same');

mag_grad = sqrt(x_deriv.*x_deriv + y_deriv.*y_deriv);

if(view_id == 1)
    tr = 1; br = 600; lc = 1; rc = 800;
    mag_grad_thresh = mag_grad > 150;
elseif(view_id == 2)
    tr = 100; br = 550; lc = 150; rc = 550; %top row, bottom row, left col, right col
    mag_grad = mag_grad(tr:br,lc:rc);
    mag_grad_thresh = and((mag_grad > 200),(mag_grad < 2000));
elseif(view_id == 4) %represents ty's new moth dataset
    tr = 1; br = 600; lc = 1; rc = 800;
    mag_grad_thresh = mag_grad > 35;
end

if debug_flag

    f = figure(10); clf;
    subplot(411); imagesc(cdata); colormap gray;
    subplot(412); imagesc(mag_grad_thresh); colormap gray;
end

cc = bwconncomp(mag_grad_thresh,8); %get connected components

%get rid of all cc less than some size

numPixels = cellfun(@numel,cc.PixelIdxList);

if(view_id == 1)
    idx = find(numPixels <=50);
elseif(view_id == 2)
    idx = find(numPixels <= 25); %set size of cc to remove
elseif(view_id == 4)
    idx = find(numPixels <= 10);
end

mag_grad_filtered = mag_grad_thresh;

for i=1:1:length(idx)
    mag_grad_filtered(cc.PixelIdxList{idx(i)}) = 0;
end

if debug_flag
subplot(413); imagesc(mag_grad_filtered); hold on;
end

[r,c] = find(mag_grad_filtered > 0);

r = r + tr - 1;
c = c + lc - 1;

min_r = min(r); max_r = max(r);
min_c = min(c); max_c = max(c);

result_bbox = [min_r,max_r,min_c,max_c];

if debug_flag
rectangle('Position',[min_c min_r max_c-min_c+1 max_r-min_r+1],'EdgeColor','r');
end
% padding = 30;
% 
% min_r = max(1,min_r-padding);
% max_r = min(600,max_r+padding);
% 
% min_c = max(1,min_c-padding);
% max_c = min(800,max_c+padding);
% 
% result = cdata(min_r:max_r,min_c:max_c); %crop out background (equivalently zoom in to Moth) 
% 
% subplot(414);
% imagesc(result); colormap gray;

end


