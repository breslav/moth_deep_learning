%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Script Objectives:
%1: Load an initial set of training data (images and their labels)
%2: Create a larger set by performing transformations on the original training set (data augmentation)
%3: Format training data so that it is compatible with Caffe and the VGG 16 network
%4: Write data to HDF5 files
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

%set image/data indices used for training 
training_idx = 1:1:400;
num_training = length(training_idx);

%allocate storage for training data, labels, and bounding boxes (used for data augmentation)
training_set = zeros(600,800,num_training); 
training_label_set = zeros(8,num_training);
training_bbox = zeros(4,num_training);

%load initial training set
[training_set, training_label_set, training_bbox] = loadInitialTrainingSet(training_idx,num_training,moth_images_path,cam1_state_pts,training_set,training_label_set,training_bbox);

%create and initialize parameters for DATASET AUGMENTATION
%we create a larger training set by randomly sampling images from the initial training set and apply transformations to them

%SET AUGMENTATION TYPE ('t' is translation only, 'tr' is translation and rotation, 'ts' is translation and scale)
%translation data augmentation will involve taking a random 400 x 600 crop of the original 600 x 800 image and then resizing it to 224 x 224 for use with VGG 16 
aug_type = 't'; 
label_scale = repmat([224/600 224/400],1,4); %labels consistent with the 400 x 600 crop then need to be scaled relative to 224 x 224

%coefficients specific to the VGG 16 network. they are to be used when training involves using the pretrained VGG 16 network as a starting point
vgg_16_bgr = [103.939, 116.779, 123.68];

%specify the total number of training data one wishes to augment to. This will be represented as num_batches x batch_size. Each batch will be stored 
%in an individual hdf5 file.

num_batches = 20;
batch_size = 1000;

%create random image indices by taking indices in order, we use them later for random permutations
im_idx = mod(0:1:batch_size-1,num_training) + 1;

%create one batch at a time, with part of the first batch containing the original training data without transformation
for i=1:1:num_batches
    %allocate storage for a batch of training images and labels (Caffe format)
    training_data = zeros(batch_size, 3, 224, 224);
    training_label = zeros(batch_size, 1, 1, 8);
    
    %generate transformations of a subset of the initial training data and store it, along with consistent labels
    [training_data,training_label] = getAugmentedData(aug_type,i,training_set,training_label_set,training_bbox,training_data,training_label,label_scale,im_idx,num_training,vgg_16_bgr,batch_size);
    
    %allocate storage for writing to hdf5
    permuted_training_data = permute(training_data,[4 3 2 1]); %hdf5 needs transposed version
    permuted_training_label = permute(training_label,[4 3 2 1]);

    %create hdf5 files.
    %NOTE: a single hdf5 file has both a /data field and a /label field, which store the data and labels respectively
    %NOTE if you try to overwrite an hdf5 file it will throw an error so you need to delete it first (or change the name)
    h5create([moth_hdf5_path,'moth_train_',num2str(i),'.hdf5'],'/data',[224,224,3,batch_size],'Datatype','double');
    h5write([moth_hdf5_path,'moth_train_',num2str(i),'.hdf5'],'/data',permuted_training_data);
    h5create([moth_hdf5_path,'moth_train_',num2str(i),'.hdf5'],'/label',[8,1,1,batch_size],'Datatype','double');
    h5write([moth_hdf5_path,'moth_train_',num2str(i),'.hdf5'],'/label',permuted_training_label);
end