%moth dataset augmentation

%need to create a dataset for moth images

addpath('Z:/Moth');

%load annotations
load('moth_example/SavedAnnotations/cam1_annotations_pts.mat');

%set train/testing split
orig_split = false; %boolean to determine whether we want to use the original training/testing split used in WACV16 experiments

if orig_split
    load('moth_example/training_testing_split_421.mat')
else
    cam1_state_pts = cell2mat(cam_state_save);
    training_idx = 1:1:400;
end
    
%training_idx / testing_idx are relative to the frames

num_training = length(training_idx);

vgg_16_bgr = [103.939, 116.779, 123.68];

label_scale = repmat([224/600 224/400],1,4); 

training_set = zeros(600,800,num_training); %store whole training set resized to 224 x 224, mean not yet subtracted
training_label_set = zeros(8,num_training);
training_bbox = zeros(4,num_training);

%SET AUGMENTATION TYPE
aug_type = 'ts'; 

%first store all num_training images in their basic form.
for i=1:1:num_training
    im = uint16(imread(['Cam1_Images/',num2str(cam1_state_pts(training_idx(i),1)),'.png'])); 
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
    
%now generate batches of training samples
num_batches = 200;
batch_size = 1000;

    
%create random im indices by taking indices in order, we use them later for random permutations
im_idx = mod(0:1:batch_size-1,num_training) + 1;


for i=1:1:num_batches
    
    training_data = zeros(batch_size, 3, 224, 224);
    training_label = zeros(batch_size, 1, 1, 8);
    
    if (i == 1) %first batch
        
        %go through all images in training_set and use them without any random crop/rotations
        for j=1:1:num_training
            label = training_label_set(:,j)'; 
            im_8_bit = training_set(:,:,j);
            bbox = training_bbox(:,j)';
            [aug_training_images, aug_training_labels] = getRotatedTranslatedMothData(im_8_bit,bbox,label,1,'aug_type','none','debug_flag',false);
            
            [b,g,r] = convertImToCaffe(double(aug_training_images),vgg_16_bgr);
            
            %transpose channels to width x height
            training_data(j,1,:,:) = b';
            training_data(j,2,:,:) = g';
            training_data(j,3,:,:) = r';
            
            aug_training_labels = aug_training_labels.*label_scale; %labels relative to 224 x 224
            training_label(j,1,1,:) = aug_training_labels;
        end
        
        
        random_im_idx = im_idx(randperm(batch_size));
        
        
        for k=num_training+1:1:batch_size %fill in the rest of the batch with random images
            
            j = random_im_idx(k-num_training); %get random image idx
            
            label = training_label_set(:,j)'; 
            im_8_bit = training_set(:,:,j);
            bbox = training_bbox(:,j)';
            [aug_training_images, aug_training_labels] = getRotatedTranslatedMothData(im_8_bit,bbox,label,1,'aug_type',aug_type,'debug_flag',false);
            
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
            [aug_training_images, aug_training_labels] = getRotatedTranslatedMothData(im_8_bit,bbox,label,1,'aug_type',aug_type,'debug_flag',false);
            
            [b,g,r] = convertImToCaffe(double(aug_training_images),vgg_16_bgr);
            
            %transpose channels to width x height
            training_data(k,1,:,:) = b';
            training_data(k,2,:,:) = g';
            training_data(k,3,:,:) = r';
            
            aug_training_labels = aug_training_labels.*label_scale; %labels relative to 224 x 224
            training_label(k,1,1,:) = aug_training_labels;       
        end
        
    end
    
    permuted_training_data = permute(training_data,[4 3 2 1]); %hdf5 needs transposed version?
    permuted_training_label = permute(training_label,[4 3 2 1]);

    h5create(['moth_example/moth_train_',num2str(i),'.hdf5'],'/data',[224,224,3,batch_size],'Datatype','double');
    h5write(['moth_example/moth_train_',num2str(i),'.hdf5'],'/data',permuted_training_data);
    h5create(['moth_example/moth_train_',num2str(i),'.hdf5'],'/label',[8,1,1,batch_size],'Datatype','double');
    h5write(['moth_example/moth_train_',num2str(i),'.hdf5'],'/label',permuted_training_label);

    
   
   
end