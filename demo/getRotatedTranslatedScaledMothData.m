
function [augmented_training_images,augmented_training_labels] = getRotatedTranslatedScaledMothData(im,result_bbox,training_label,num_samples,varargin)

for a=1:2:length(varargin)
eval(sprintf('%s = varargin{a+1};',lower(varargin{a})));
end

if(~exist('aug_type','var'))
    aug_type = 'none';
end

if(~exist('debug_flag','var'))
    debug_flag = false;
end

center_offset = [400 300 400 300 400 300 400 300];
w = 800; h=600;
crop_w = 600; crop_h = 400;


augmented_training_images = uint16(zeros(crop_h,crop_w,num_samples)); 
augmented_training_labels = zeros(1,8,num_samples);

%sample randomly a row from [1 to top_margin], and col from [1 to col_margin]
for j=1:1:num_samples
    
    valid_rotation = false;
    valid_scale = false;
    
    if(strcmp(aug_type,'tr')) %do random translations and rotations 
        
        while ~valid_rotation
        
        %first try to rotate about moth's center
        rand_rot_degrees = randi([-45 45],1,1); %random rotation in degrees
        rotated_im = imrotate(im,rand_rot_degrees,'bilinear','crop'); %preserve size
        
        centered_training_label = training_label - center_offset;
        centered_training_label = reshape(centered_training_label,[2 4]);

        R = [cos(-pi*rand_rot_degrees/180) -sin(-pi*rand_rot_degrees/180); sin(-pi*rand_rot_degrees/180) cos(-pi*rand_rot_degrees/180)];
        
        rotated_labels = R*centered_training_label;
        rotated_labels = reshape(rotated_labels,[1 8]);
        rotated_labels = rotated_labels + center_offset;
        
        %perform translation
        rotated_label_x = rotated_labels(1:2:end);
        rotated_label_y = rotated_labels(2:2:end); 
        
        rotated_bbox = [min(rotated_label_y) max(rotated_label_y) min(rotated_label_x) max(rotated_label_x)];
        rotated_bbox = round(rotated_bbox);
        
        min_r = rotated_bbox(1); max_r = rotated_bbox(2);
        min_c = rotated_bbox(3); max_c = rotated_bbox(4);
        
        valid_r = (min_r > 1) && (min_r < 600) && (max_r < 600) && (max_r > 1);
        valid_c = (min_c > 1) && (min_c < 800) && (max_c < 800) && (max_c > 1);
        
            if(valid_r && valid_c)
                valid_rotation = true;
            else
                display('problem?');
            end

        end
        
        [left_range,top_range] = computeCropRanges(rotated_bbox,w,h,crop_w,crop_h); %get crop ranges for 400 x 600 crop
        
        c = randi(left_range,1,1); %randomly pick a column
        r = randi(top_range,1,1); %randomly pick a row
   
        sample_training_im = rotated_im(r:r+crop_h-1,c:c+crop_w-1);
        sample_training_label = rotated_labels - [c-1 r-1 c-1 r-1 c-1 r-1 c-1 r-1];
        
    elseif strcmp(aug_type,'t') %do translations alone
        
        [left_range,top_range] = computeCropRanges(result_bbox,w,h,crop_w,crop_h); %get crop ranges for 400 x 600 crop
        
        c = randi(left_range,1,1); %randomly pick a column
        r = randi(top_range,1,1); %randomly pick a row
   
        sample_training_im = im(r:r+crop_h-1,c:c+crop_w-1);
        sample_training_label = training_label - [c-1 r-1 c-1 r-1 c-1 r-1 c-1 r-1];
    elseif strcmp(aug_type,'ts') %do translations and scale
        
        while ~valid_scale
        
        sf = .5 + rand(); %pick a random scale factor in [.5 1.5] 
        
        im_scaled = imresize(im,sf,'bicubic'); %randomly scale original image. Anti-aliasing is done internally.
        result_bbox_scaled = round(sf*result_bbox); 
        
        bbox_s_w = result_bbox_scaled(4) - result_bbox_scaled(3) + 1;
        bbox_s_h = result_bbox_scaled(2) - result_bbox_scaled(1) + 1;
        
        %need to know whether resized bbox is larger than the crop size of 400 x 600
        if( (bbox_s_w > crop_w) || (bbox_s_h > crop_h) )
            continue;
        else
            valid_scale = true;
        end
        
        %if SF > 1 then we want to compute potential crops with respect to larger sized image
        %if SF <=1 then we want to compute potential crops with respect to original image, not the smaller one
        
        if(sf > 1)
            [left_range,top_range] = computeCropRanges(result_bbox_scaled,size(im_scaled,2),size(im_scaled,1),crop_w,crop_h); %get crop ranges for 400 x 600 crop
            im_full = im_scaled;
        else %sf <=1
            [left_range,top_range] = computeCropRanges(result_bbox_scaled,w,h,crop_w,crop_h); %get crop ranges for 400 x 600 crop
            im_full = zeros(h,w);
            im_full(1:size(im_scaled,1),1:size(im_scaled,2)) = im_scaled;
            
        end
        
        c = randi(left_range,1,1); %randomly pick a column
        r = randi(top_range,1,1); %randomly pick a row
   
        sample_training_im = im_full(r:r+crop_h-1,c:c+crop_w-1);
        sample_training_label = sf*training_label - [c-1 r-1 c-1 r-1 c-1 r-1 c-1 r-1];
        
        end
        
    else %no transformation
        [left_range,top_range] = computeCropRanges(result_bbox,w,h,crop_w,crop_h); %get crop ranges for 400 x 600 crop

        %get centered crop
        c = floor( (left_range(1) + left_range(2))/2);
        r = floor( (top_range(1) + top_range(2))/2);
        sample_training_im = im(r:r+crop_h-1,c:c+crop_w-1);
        sample_training_label = training_label - [c-1 r-1 c-1 r-1 c-1 r-1 c-1 r-1];
    end
    

    if(debug_flag)
        figure(1);clf;imagesc(sample_training_im); colormap gray; hold on;    
        plot(sample_training_label(1),sample_training_label(2),'or','MarkerSize',12); hold on;
        plot(sample_training_label(3),sample_training_label(4),'og','MarkerSize',12); hold on;
        plot(sample_training_label(5),sample_training_label(6),'oc','MarkerSize',12); hold on;
        plot(sample_training_label(7),sample_training_label(8),'om','MarkerSize',12); hold on;
    end
    
    augmented_training_images(:,:,j) = sample_training_im;
    augmented_training_labels(:,:,j) = sample_training_label;
end

end