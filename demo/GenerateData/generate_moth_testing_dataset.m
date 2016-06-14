%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Script Objectives:
%1: Load an initial set of testing data (images and their labels)
%2: Format training data so that it is compatible with Caffe and the VGG 16 network
%3: Write data to HDF5 files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%setup path variables
demo_path = 'moth_example/demo/';
moth_images_path = 'moth_example/Cam1_Images/';
moth_annotations_path = 'moth_example/SavedAnnotations/cam1_annotations_pts.mat';
moth_hdf5_path = 'moth_example/';

%add to path the helper functions
addpath(genpath(demo_path));

%load annotations
load(moth_annotations_path);

%create matrix of annotations
cam1_state_pts = cell2mat(cam_state_save);

%set image/data indices used for testing
testing_idx = 401:1:800;
num_testing = length(testing_idx);
num_samples = num_testing;

%allocate storage for testing data 
%storage in caffe format (num samples x num channels x H x W)
testing_data = zeros(num_samples, 3, 224, 224);
testing_label = zeros(num_samples, 1, 1, 8);

%bgr channel offsets, network specific
vgg_16_bgr = [103.939, 116.779, 123.68];
label_scale = repmat([224/600 224/400],1,4); 


for i=1:1:num_testing
    im = uint16(imread([moth_images_path,num2str(cam1_state_pts(testing_idx(i),1)),'.png'])); 
    
    result_bbox = getMothBoundingBox2(im,1); %get tight bounding box on moth
    label = abs(cam1_state_pts(testing_idx(i),2:9));
    
    im = double(im);
    %turn into RGB 8 bit
    min_val = min(im(:));
    max_val = max(im(:));
   
    im_8bit = (im-min_val)./(max_val - min_val); %scale between 0 and 1
    im_8bit = floor(255*im_8bit); %0-255
    
    %in this case we just get a 400 x 600 crop from the original test image
    [data,labels] = getRotatedTranslatedScaledMothData(im_8bit,result_bbox,label,1,'aug_type','t','debug_flag',false);
    
    im_8bit = double(data); %note the values are 8 bit but we still store in a double
    
    b_channel = im_8bit - vgg_16_bgr(1); 
    g_channel = im_8bit - vgg_16_bgr(2);
    r_channel = im_8bit - vgg_16_bgr(3);

    b = imresize(b_channel,[224 224],'bilinear')'; %transpose to height x width
    g = imresize(g_channel,[224 224],'bilinear')';
    r = imresize(r_channel,[224 224],'bilinear')';

    testing_data(i,1,:,:) = b;
    testing_data(i,2,:,:) = g;
    testing_data(i,3,:,:) = r;
    
    labels = labels.*label_scale;
    testing_label(i,1,1,:) = labels';
 
end

%for writing to hdf5 file we reverse dimension order.
permuted_testing_data = permute(testing_data,[4 3 2 1]); %hdf5 needs transposed version?
permuted_testing_label = permute(testing_label,[4 3 2 1]);

%create hdf5 files.
%NOTE: a single hdf5 file has both a /data field and a /label field, which store the data and labels respectively
%NOTE if you try to overwrite an hdf5 file it will throw an error so you need to delete it first (or change the name)
h5create([moth_hdf5_path,'moth_test_',num2str(1),'.hdf5'],'/data',[224,224,3,num_samples],'Datatype','double');
h5write([moth_hdf5_path,'moth_test_',num2str(1),'.hdf5'],'/data',permuted_testing_data);
h5create([moth_hdf5_path,'moth_test_',num2str(1),'.hdf5'],'/label',[8,1,1,num_samples],'Datatype','double');
h5write([moth_hdf5_path,'moth_test_',num2str(1),'.hdf5'],'/label',permuted_testing_label);
