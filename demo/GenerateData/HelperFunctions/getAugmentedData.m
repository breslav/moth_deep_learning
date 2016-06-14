function [training_data,training_label] = getAugmentedData(aug_type,i,training_set,training_label_set,training_bbox,training_data,training_label,label_scale,im_idx,num_training,vgg_16_bgr,batch_size)

    if (i == 1) %first batch
        
        %go through all images in training_set and use them without any random crop/rotations/scale
        for j=1:1:num_training
            label = training_label_set(:,j)'; 
            im_8_bit = training_set(:,:,j);
            bbox = training_bbox(:,j)';
            [aug_training_images, aug_training_labels] = getRotatedTranslatedScaledMothData(im_8_bit,bbox,label,1,'aug_type','none','debug_flag',false);
            
            [b,g,r] = convertImToCaffe(double(aug_training_images),vgg_16_bgr);
            
            %transpose channels to width x height
            training_data(j,1,:,:) = b';
            training_data(j,2,:,:) = g';
            training_data(j,3,:,:) = r';
            
            aug_training_labels = aug_training_labels.*label_scale; %labels relative to 224 x 224
            training_label(j,1,1,:) = aug_training_labels;
        end
        
        %populate remaining training samples by getting a random image from the original set and transforming it
        random_im_idx = im_idx(randperm(batch_size));
        
        
        for k=num_training+1:1:batch_size %fill in the rest of the batch with random images
            
            j = random_im_idx(k-num_training); %get random image idx
            
            label = training_label_set(:,j)'; 
            im_8_bit = training_set(:,:,j);
            bbox = training_bbox(:,j)';
            [aug_training_images, aug_training_labels] = getRotatedTranslatedScaledMothData(im_8_bit,bbox,label,1,'aug_type',aug_type,'debug_flag',false);
            
            [b,g,r] = convertImToCaffe(double(aug_training_images),vgg_16_bgr);
            
            %transpose channels to width x height
            training_data(k,1,:,:) = b';
            training_data(k,2,:,:) = g';
            training_data(k,3,:,:) = r';
            
            aug_training_labels = aug_training_labels.*label_scale; %labels relative to 224 x 224
            training_label(k,1,1,:) = aug_training_labels;       
        end
        
    else
        random_im_idx = im_idx(randperm(batch_size));
        
        for k=1:1:batch_size %fill in the rest of the batch with random images
            
            j = random_im_idx(k); %get random image idx
            
            label = training_label_set(:,j)'; 
            im_8_bit = training_set(:,:,j);
            bbox = training_bbox(:,j)';
            [aug_training_images, aug_training_labels] = getRotatedTranslatedScaledMothData(im_8_bit,bbox,label,1,'aug_type',aug_type,'debug_flag',false);
            
            [b,g,r] = convertImToCaffe(double(aug_training_images),vgg_16_bgr);
            
            %transpose channels to width x height
            training_data(k,1,:,:) = b';
            training_data(k,2,:,:) = g';
            training_data(k,3,:,:) = r';
            
            aug_training_labels = aug_training_labels.*label_scale; %labels relative to 224 x 224
            training_label(k,1,1,:) = aug_training_labels;       
        end
        
    end
    
end