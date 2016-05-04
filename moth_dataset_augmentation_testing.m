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
    testing_idx = 401:1:800;
end

num_testing = length(testing_idx);

num_samples = num_testing;

%storage in caffe format (num samples x num channels x H x W)
testing_data = zeros(num_samples, 3, 224, 224);
testing_label = zeros(num_samples, 1, 1, 8);

%bgr channel offsets, network specific
vgg_16_bgr = [103.939, 116.779, 123.68];

label_scale = repmat([224/600 224/400],1,4); 


for i=1:1:num_testing
    im = uint16(imread(['Cam1_Images/',num2str(cam1_state_pts(testing_idx(i),1)),'.png'])); 
    
    result_bbox = getMothBoundingBox2(im,1); %get tight bounding box on moth
    label = abs(cam1_state_pts(testing_idx(i),2:9));
    
    [data,labels] = getTranslatedMothData(im,result_bbox,num_samples,label); %get a bunch of crops
    
    %pre-processing
    im = double(im);
    %turn into RGB 8 bit?
    min_val = min(im(:));
    max_val = max(im(:));
   
    j=1; %just pick the first crop for test set
    im_8bit = (double(data(:,:,j))-min_val)./(max_val - min_val); %scale between 0 and 1
    im_8bit = floor(255*im_8bit); %0-255
        
    b_channel = im_8bit - vgg_16_bgr(1); 
    g_channel = im_8bit - vgg_16_bgr(2);
    r_channel = im_8bit - vgg_16_bgr(3);

    b = imresize(b_channel,[224 224],'bilinear')'; %transpose to height x width
    g = imresize(g_channel,[224 224],'bilinear')';
    r = imresize(r_channel,[224 224],'bilinear')';

    testing_data(i,1,:,:) = b;
    testing_data(i,2,:,:) = g;
    testing_data(i,3,:,:) = r;
    
    labels(:,:,j) = labels(:,:,j).*label_scale;
    testing_label(i,1,1,:) = labels(:,:,j)';
 
end

%for writing to hdf5 file we reverse dimension order.
permuted_testing_data = permute(testing_data,[4 3 2 1]); %hdf5 needs transposed version?
permuted_testing_label = permute(testing_label,[4 3 2 1]);

h5create(['moth_example/hdf5_data/moth_test_',num2str(1),'.hdf5'],'/data',[224,224,3,num_samples],'Datatype','double');
h5write(['moth_example/hdf5_data/moth_test_',num2str(1),'.hdf5'],'/data',permuted_testing_data);
h5create(['moth_example/hdf5_data/moth_test_',num2str(1),'.hdf5'],'/label',[8,1,1,num_samples],'Datatype','double');
h5write(['moth_example/hdf5_data/moth_test_',num2str(1),'.hdf5'],'/label',permuted_testing_label);
