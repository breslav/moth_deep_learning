%moth dataset augmentation

%need to create a dataset for moth images

addpath('Z:/Moth');

%load train/testing data
load('moth_example/training_testing_split_421.mat')
load('moth_example/SavedAnnotations/cam1_annotations_pts.mat');

%training_idx / testing_idx are relative to the frames

num_training = length(training_idx);

num_samples = 500;

training_data = zeros(num_samples, 3, 224, 224);
training_label = zeros(num_samples, 1, 1, 8);


vgg_16_bgr = [103.939, 116.779, 123.68];

label_scale = repmat([224/600 224/400],1,4); 

training_set = zeros(600,800,num_training); %store whole training set resized to 224 x 224, mean not yet subtracted
training_label_set = zeros(8,num_training);
training_bbox = zeros(4,num_training);
%first store all ~200 training images in their basic form.
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
    training_label_set(:,i) = label'; %relative to 224 x 224 image
    training_bbox(:,i) = result_bbox';  
end
    
%now generate batches of training samples
num_batches = 200;
batch_size = 1000;
    
for i=1:1:num_batches
    
    if (i == 1) %first batch
        
        %go through all images in training_set and use them without any random crop/rotations
        for j=1:1:num_training
            label = training_label_set(:,j)'; 
            im_8_bit = training_set(:,:,j);
            bbox = training_bbox(:,j)';
            [aug_training_images, aug_training_labels] = getRotatedTranslatedMothData(im_8_bit,bbox,label,1,'aug_type','none','debug_flag',true);
        end        

    else
            [aug_training_images, aug_training_labels] = getRotatedTranslatedMothData(im_8_bit,bbox,label,1,'aug_type','random','debug_flag',false);
      
    end
    
    
    
    result_bbox = getMothBoundingBox2(im,1); %get tight bounding box on moth
    label = cam1_state_pts(training_idx(i),2:9);
    
    [data,labels] = getTranslatedMothData(im,result_bbox,num_samples,label);
    
    im = double(im);
    %turn into RGB 8 bit?
    min_val = min(im(:));
    max_val = max(im(:));
   
    for j=1:1:num_samples
        im_8bit = (double(data(:,:,j))-min_val)./(max_val - min_val); %scale between 0 and 1
        im_8bit = floor(255*im_8bit); %0-255
        
        b_channel = im_8bit - vgg_16_bgr(1); 
        g_channel = im_8bit - vgg_16_bgr(2);
        r_channel = im_8bit - vgg_16_bgr(3);

        b = imresize(b_channel,[224 224],'bilinear')'; %transpose to height x width
        g = imresize(g_channel,[224 224],'bilinear')';
        r = imresize(r_channel,[224 224],'bilinear')';
        
        training_data(j,1,:,:) = b;
        training_data(j,2,:,:) = g;
        training_data(j,3,:,:) = r;
        
        
        labels(:,:,j) = labels(:,:,j).*label_scale;
        training_label(j,1,1,:) = labels(:,:,j)';

    end
    
   
    permuted_training_data = permute(training_data,[4 3 2 1]); %hdf5 needs transposed version?
    permuted_training_label = permute(training_label,[4 3 2 1]);

    h5create(['moth_example/moth_train_',num2str(i),'.hdf5'],'/data',[224,224,3,num_samples],'Datatype','double');
    h5write(['moth_example/moth_train_',num2str(i),'.hdf5'],'/data',permuted_training_data);
    h5create(['moth_example/moth_train_',num2str(i),'.hdf5'],'/label',[8,1,1,num_samples],'Datatype','double');
    h5write(['moth_example/moth_train_',num2str(i),'.hdf5'],'/label',permuted_training_label);

    
   
   
end