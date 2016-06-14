function [training_set,training_label_set,training_bbox] = loadInitialTrainingSet(training_idx,num_training,moth_images_path,cam1_state_pts,training_set,training_label_set,training_bbox)

%first store all num_training images in their basic form.
for i=1:1:num_training
    im = uint16(imread([moth_images_path,num2str(cam1_state_pts(training_idx(i),1)),'.png'])); 
    result_bbox = getMothBoundingBox2(im,1); %get tight bounding box on moth
    label = abs(cam1_state_pts(training_idx(i),2:9)); %absolute value which means we are using guesses for occluded parts as well
    
    im = double(im);
    %turn into RGB 8 bit?
    min_val = min(im(:));
    max_val = max(im(:));
   
    im_8bit = (im-min_val)./(max_val - min_val); %scale between 0 and 1
    im_8bit = floor(255*im_8bit); %0-255
        
%     im_8bit_resized = imresize(im_8bit,[224 224],'bilinear'); %NOT transposed 
    
    training_set(:,:,i) = im_8bit;
    training_label_set(:,i) = label'; %relative to 600 x 800
    training_bbox(:,i) = result_bbox';  
    
end

end
